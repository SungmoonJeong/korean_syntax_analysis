# api/pipeline.py
#
# main.py(Streamlit)의 run_analysis/split_sentences와 utils/loader.py의 리소스 로더를
# Streamlit 런타임 없이 그대로 쓸 수 있게 옮겨온 모듈. 분석 파이프라인 로직 자체는
# 원본에서 단 한 줄도 바뀌지 않았다 — @st.cache_resource(세션 캐시)만 서버 프로세스
# 시작 시 1회 로딩(load_all_resources)으로 바뀌었다. main.py/utils/loader.py는
# 그대로 두고 이 파일만 새로 추가했다.

import os
import pickle
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
warnings.filterwarnings("ignore", message="apply_permutation is deprecated")

import torch
import torch.serialization as ts
from dotenv import load_dotenv
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from kiwipiepy import Kiwi
from openai import OpenAI
from supar import Parser
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import core.analyzer as analyzer
import services.glossing as glossing
import services.openai_client as openai_client
import visualization.renderer as renderer
from config import KIWI_USER_WORDS, POS_ENG_MAP, device_ce
from span_extraction.morpheme_span_extractor_klue_phase3 import (
    MorphemeSpanExtractorKLUE as MorphemeSpanExtractor,
)

# 어느 디렉터리에서 서버를 띄우든 프로젝트 루트 기준 상대경로가 항상 맞도록 고정
# (main.py/utils/loader.py는 프로세스 cwd에 의존하는 상대경로를 그대로 씀 — 새
# 코드라서 가능한 개선이며, 기존 파일 동작에는 영향 없음)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GLOSS_PKL_PATH = str(PROJECT_ROOT / "gloss_dict.pkl")
MODEL_PATH_CE = str(PROJECT_ROOT / "klue_roberta_ce_listwise_llrd")
TOKENIZER_PATH_CE = str(PROJECT_ROOT / "klue_roberta_ce_listwise_llrd")
SUPAR_MODEL_PATH = str(PROJECT_ROOT / "supar_morph_dp" / "model.pth")

TAU = 0.6
MARGIN = 0.0
USE_LLM = True

EXAMPLE_SENTENCES = [
    "나는 어제 도서관에서 책을 읽었다.",
    "그는 내일 비가 올 것이라고 말했다.",
    "작년 이맘때는 눈이 왔는데 올해는 벌써 덥다.",
]

load_dotenv(PROJECT_ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================================
# 리소스 로딩 — utils/loader.py와 동일한 로직, @st.cache_resource만 제거
# ============================================================================
@contextmanager
def allow_pickle_load():
    # PyTorch 2.6의 기본 safe-load 제약 완화 (supar pickle 호환) — utils/loader.py와 동일
    orig_load, orig_ts_load = torch.load, ts.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    def patched_ts_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_ts_load(*args, **kwargs)

    torch.load, ts.load = patched_load, patched_ts_load
    try:
        yield
    finally:
        torch.load, ts.load = orig_load, orig_ts_load


def _load_resources(supar_model_path: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with allow_pickle_load():
        parser = Parser.load(supar_model_path, device=device)
        print("SuPar device:", next(parser.model.parameters()).device)
    kiwi = Kiwi(integrate_allomorph=False)
    for word, tag, score in KIWI_USER_WORDS:
        kiwi.add_user_word(word, tag, score)
    return parser, kiwi


def _load_ce_model(path_model: str, path_tok: str):
    tok = AutoTokenizer.from_pretrained(path_tok)
    mdl = AutoModelForSequenceClassification.from_pretrained(path_model)
    return tok, mdl


# 서버 시작 시 load_all_resources()가 한 번만 채워 넣는 전역 슬롯
PARSER = None
KIWI = None
GLOSS_DICT: dict = {}
CE_TOK = None
CE_MODEL = None
AI_HANDLER: "openai_client.OpenAIHandler | None" = None


def load_all_resources() -> None:
    """무거운 리소스를 프로세스당 1회 로드한다 (FastAPI lifespan에서 호출)."""
    global PARSER, KIWI, GLOSS_DICT, CE_TOK, CE_MODEL, AI_HANDLER

    print("[pipeline] SuPar 파서 + Kiwi 로딩 중...")
    PARSER, KIWI = _load_resources(SUPAR_MODEL_PATH)

    print("[pipeline] OpenAI client 로딩 중...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    AI_HANDLER = openai_client.OpenAIHandler(client)

    print("[pipeline] gloss_dict.pkl 로딩 중...")
    try:
        with open(GLOSS_PKL_PATH, "rb") as f:
            GLOSS_DICT = pickle.load(f)
    except Exception as e:
        print(f"[pipeline] 사전 로드 실패: {e}")
        GLOSS_DICT = {}

    print("[pipeline] cross-encoder 모델 로딩 중...")
    CE_TOK, CE_MODEL = _load_ce_model(MODEL_PATH_CE, TOKENIZER_PATH_CE)
    CE_MODEL.to(torch.device(str(device_ce))).eval()

    print("[pipeline] 리소스 로딩 완료")


# ============================================================================
# 분석 파이프라인 — main.py의 split_sentences/run_analysis를 그대로 포팅
# ============================================================================
def split_sentences(text: str) -> list[str]:
    """마침표/물음표/느낌표 뒤 공백 기준으로 문장 분리"""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


def run_analysis(text: str) -> dict:
    """단일 문장을 분석하여 결과 dict 반환 (main.py:run_analysis와 동일 로직)"""
    tokens, xpos, spans_off, lemmas = analyzer.kiwi_morphs(KIWI, text)
    words_m, upos_m, arcs_m, rels_m = analyzer.parse_dep(PARSER, tokens)

    pos_eng_for_view = [
        v if (v := POS_ENG_MAP.get(t.split("+")[0], t)) != "품사 없음" else ""
        for t in xpos
    ]
    groups = analyzer.group_by_eojeol(text, tokens, spans_off)

    spans_m = MorphemeSpanExtractor(tokens, xpos, arcs_m, rels_m).extract()

    outs = glossing.gloss_sequence_from_tokens(
        sentence=text,
        tokens=tokens,
        lemmas=lemmas,
        xpos=xpos,
        gloss_dict=GLOSS_DICT,
        tau=TAU,
        margin=MARGIN,
        ce_tok=CE_TOK,
        ce_model=CE_MODEL,
        use_llm=USE_LLM,
        ai_handler=AI_HANDLER,
    )
    gloss_texts = [(o.get("gloss") or "") for o in outs]

    # compound rule로 묶인 구간 목록: {s, e, gloss, pos_eng}
    # 포함: compound_ngram(조사/어미), affix_compound(접사)
    compound_spans = []
    i_out = 0
    while i_out < len(outs):
        meta = outs[i_out].get("meta", {})
        rule = meta.get("rule", "")
        if rule in ("compound_ngram", "affix_compound"):
            span_len = meta.get("span", 1)
            if span_len > 1:
                o = outs[i_out]
                pe = o.get("pos_eng", "")
                compound_spans.append({
                    "s": i_out,
                    "e": i_out + span_len - 1,
                    "gloss": o.get("gloss") or "",
                    "pos_eng": pe if pe and pe != "품사 없음" else "",
                })
            i_out += span_len
        else:
            i_out += 1

    phrase_html, phrase_h = renderer.render_phrase_bars(
        tokens, spans_m, groups,
        gloss=gloss_texts,
        pos_tags=pos_eng_for_view,
        compound_spans=compound_spans,
        show_eojeol=True,
        theme="dark",
        density="cozy",
        scale=1.2,
    )

    # 상단 원문 문장에 구(Phrase) 단위 색상 적용
    sentence_html = renderer.render_colored_sentence(text, spans_off, spans_m)

    romanize = ""
    try:
        romanize = Transliter(academic).translit(text)
    except Exception:
        pass

    translation = AI_HANDLER.translate_to_english(text)

    return {
        "sentence":      text,
        "sentence_html": sentence_html,
        "html":          phrase_html,
        "html_h":        phrase_h,
        "romanize":      romanize,
        "translation":   translation,
        "tts_html":      renderer.render_tts_button(text),
    }


def analyze_text(text: str) -> list[dict]:
    """입력 텍스트를 문장 단위로 나눠 각각 분석 — main.py의 분석 실행 루프와 동일"""
    sentences = split_sentences(text.strip())
    results = []
    for sent in sentences:
        try:
            result = run_analysis(sent)
        except Exception as e:
            result = {
                "sentence":      sent,
                "sentence_html": sent,
                "html":          f"<p style='color:red;padding:16px'>분석 오류: {e}</p>",
                "html_h":        80,
                "romanize":      "",
                "translation":   "",
                "tts_html":      "",
            }
        results.append(result)
    return results
