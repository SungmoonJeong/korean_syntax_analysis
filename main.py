# main.py — Streamlit Entry Point
# 역할: UI 라우팅 + 분석 파이프라인 오케스트레이션

import re

from hangul_romanize import Transliter
from hangul_romanize.rule import academic

# KLUE-DP 기반 Span Extractor (기존 버전 morpheme_span_extractor_klue의 업데이트 버전)
from span_extraction.morpheme_span_extractor_klue_phase3 import MorphemeSpanExtractorKLUE as MorphemeSpanExtractor

import core.analyzer as analyzer
import services.glossing as glossing
import services.openai_client as openai_client
import streamlit as st
import utils.loader as loader
import visualization.renderer as renderer
from config import PAGE_LAYOUT, PAGE_TITLE, POS_ENG_MAP, device_ce

# ============================================================================
# 페이지 고유 상수
# ============================================================================
EXAMPLE_SENTENCES = [
    "나는 어제 도서관에서 책을 읽었다.",
    "그는 내일 비가 올 것이라고 말했다.",
    "작년 이맘때는 눈이 왔는데 올해는 벌써 덥다.",
]


# ============================================================================
# 유틸 함수
# ============================================================================
def split_sentences(text: str) -> list[str]:
    """마침표/물음표/느낌표 뒤 공백 기준으로 문장 분리"""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


def run_analysis(text, parser, kiwi, gloss_dict, ce_tok, ce_model,
                 ai_handler, gemini_handler, TAU, MARGIN, USE_LLM) -> dict:
    """단일 문장을 분석하여 결과 dict 반환"""
    # 고유명사(NNP) 판별: Kiwi 분석 전에 Gemini로 스팬을 뽑아 pretokenized로 고정.
    # 실패해도 빈 리스트라 원본 Kiwi 분석 그대로 진행됨(파이프라인 안 막힘).
    nnp_spans = gemini_handler.detect_nnp(text)
    tokens, xpos, spans_off, lemmas = analyzer.kiwi_morphs(kiwi, text, pretokenized=nnp_spans)
    words_m, upos_m, arcs_m, rels_m = analyzer.parse_dep(parser, tokens)

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
        gloss_dict=gloss_dict,
        tau=TAU,
        margin=MARGIN,
        ce_tok=ce_tok,
        ce_model=ce_model,
        use_llm=USE_LLM,
        ai_handler=ai_handler,
        kiwi=kiwi,
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

    translation = ai_handler.translate_to_english(text)

    return {
        "sentence":      text,
        "sentence_html": sentence_html,
        "html":          phrase_html,
        "html_h":        phrase_h,
        "romanize":      romanize,
        "translation":   translation,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

    st.markdown(renderer.get_app_styles(), unsafe_allow_html=True)
    st.markdown(renderer.render_header(), unsafe_allow_html=True)

    GLOSS_PKL_PATH = "gloss_dict.pkl"
    MODEL_PATH_CE  = "./klue_roberta_ce_listwise_llrd"
    TOKENIZER_PATH_CE = "./klue_roberta_ce_listwise_llrd"
    MARGIN  = 0.0
    TAU     = 0.6
    USE_LLM = True

    # ── 리소스 로드 ───────────────────────────────────────────────────────────
    client     = loader.load_openai_client()
    ai_handler = openai_client.OpenAIHandler(client)
    gemini_handler = loader.load_gemini_handler()

    try:
        gloss_dict = loader.load_gloss_pickle(GLOSS_PKL_PATH)
    except Exception as e:
        gloss_dict = {}
        st.warning(f"사전 로드 실패: {e}")

    parser, kiwi = loader.load_resources()
    ce_tok, ce_model = loader.load_ce_model_cached(
        MODEL_PATH_CE, TOKENIZER_PATH_CE, str(device_ce),
    )

    # ── 세션 상태 초기화 ──────────────────────────────────────────────────────
    defaults = {
        "results":     None,  # list[dict] | None
        "current_idx": 0,
        "input_text":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ══════════════════════════════════════════════════════════════════════════
    # 상단 고정 영역: 타이틀 + 입력 필드 + 버튼 + 예시 문장
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(renderer.render_title(), unsafe_allow_html=True)

    input_text = st.text_area(
        label="문장 입력",
        value=st.session_state.input_text,
        placeholder="분석할 문장을 입력하세요",
        height=120,
        label_visibility="collapsed",
    )
    # text_area 값 동기화 (key 없이 value로 제어)
    st.session_state.input_text = input_text

    # 글자수: text_area 바로 아래 오른쪽 정렬 (하단 느낌)
    char_count = len(st.session_state.input_text)
    st.markdown(
        f'<div class="char-count" style="margin-top:-12px; text-align:right;">'
        f'{char_count}자'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 분석하기 버튼
    _, btn_col = st.columns([5, 1])
    with btn_col:
        analyze_clicked = st.button(
            "분석하기",
            key="analyze_btn",
            disabled=not input_text.strip(),
            use_container_width=True,
            type="primary",
        )

    # 예시 문장 카드
    st.markdown('<div class="example-label">예시 문장</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLE_SENTENCES))
    for i, (col, sent) in enumerate(zip(ex_cols, EXAMPLE_SENTENCES)):
        with col:
            if st.button(sent, key=f"ex_{i}"):
                # rerun 전에 session_state만 바꾸면 다음 렌더에서 text_area value가 갱신됨
                st.session_state.input_text = sent
                st.session_state.results    = None
                st.rerun()

    # ── 분석 실행 ─────────────────────────────────────────────────────────────
    if analyze_clicked and input_text.strip():
        sentences = split_sentences(input_text.strip())
        st.session_state.current_idx = 0

        results = []
        with st.spinner("분석 중..."):
            for sent in sentences:
                try:
                    result = run_analysis(
                        sent, parser, kiwi, gloss_dict,
                        ce_tok, ce_model, ai_handler, gemini_handler,
                        TAU, MARGIN, USE_LLM,
                    )
                except Exception as e:
                    result = {
                        "sentence":      sent,
                        "sentence_html": sent,
                        "html":          f"<p style='color:red;padding:16px'>분석 오류: {e}</p>",
                        "html_h":        80,
                        "romanize":      "",
                        "translation":   "",
                    }
                results.append(result)
        st.session_state.results = results
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # 결과 영역: 분석 결과가 있을 때만 표시 (PDF 기획 3/4번 화면)
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.results:
        results = st.session_state.results
        idx     = st.session_state.current_idx
        total   = len(results)
        cur     = results[idx]

        # 복수 문장일 때 ◀ 카드 ▶ 3단 레이아웃, 단일이면 카드만
        if total > 1:
            col_prev, col_card, col_next = st.columns([1, 16, 1])
        else:
            col_card = st.container()
            col_prev = col_next = None

        # ── ◀ 이전 버튼 ───────────────────────────────────────────────────
        if col_prev is not None:
            with col_prev:
                st.markdown("")  # 세로 여백
                st.markdown("")
                st.markdown("")
                if idx > 0:
                    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
                    if st.button("◀", key="prev"):
                        st.session_state.current_idx -= 1
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── 결과 카드 ─────────────────────────────────────────────────────
        with col_card:
            with st.container(border=True):
                # 문장(좌) + 번호(우상단) — 단일 HTML 블록으로 레이아웃 고정
                st.markdown(
                    f"""
                    <div style="position:relative; margin-bottom:4px;">
                      <div style="position:absolute; top:0; right:0;
                                  font-size:0.85rem; color:#9ca3af;">
                        {idx + 1} / {total}
                      </div>
                      <div class="result-sentence">{cur["sentence_html"]}</div>
                      {"" if not cur["romanize"] else
                       f'<div class="result-roman">{cur["romanize"]}</div>'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # TTS 버튼 — JS 자급자족 컴포넌트
                st.components.v1.html(
                    renderer.render_tts_button(cur["sentence"]),
                    height=44,
                )

                st.markdown("---")

                # 형태소/구절 시각화 — HTML 내부 outer-scroll이 가로 스크롤 담당
                st.components.v1.html(cur["html"], height=cur.get("html_h", 300), scrolling=False)

                # 번역
                st.markdown(
                    renderer.render_translation(cur["translation"]),
                    unsafe_allow_html=True,
                )

        # ── ▶ 다음 버튼 ───────────────────────────────────────────────────
        if col_next is not None:
            with col_next:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                if idx < total - 1:
                    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
                    if st.button("▶", key="next"):
                        st.session_state.current_idx += 1
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── 처음으로 버튼 (카드 우하단 바깥) ────────────────────────────
        _, reset_col = st.columns([8, 2])
        with reset_col:
            st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
            if st.button("처음으로", key="reset", use_container_width=True):
                st.session_state.results     = None
                st.session_state.input_text  = ""
                st.session_state.current_idx = 0
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
