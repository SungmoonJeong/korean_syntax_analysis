# local_ner_client.py
"""로컬(오프라인) 고유명사(NNP) 판별 — 외부 API를 쓸 수 없는 환경(예: 울산대 배포)용
Gemini 대체제. services.gemini_client.GeminiHandler와 동일한 detect_nnp() 인터페이스를
제공해 호출부는 어떤 핸들러가 꽂혀있는지 몰라도 되도록 한다.

klue-dev/train-morph.conllu(2000+10000문장) 검증 결과, soddokayo/klue-roberta-large-klue-ner
(KLUE-NER PS/LC/OG 파인튜닝)를 안전장치 없이 그대로 쓰면 태깅 정확도가 오히려
베이스라인(훅 없음)보다 떨어졌다(-0.08~0.20%p). 회귀 사례를 직접 까본 결과 대부분이
스팬 자체 품질 문제였고, 아래 필터들로 그 문제들을 겨냥한다:
- 신뢰도 임계값(평균 0.80 미만은 버림) — Gemini의 CONFIDENCE_THRESHOLD와 동일 원칙
- 호칭/익명 접미사 트림(씨/군/양/님/옹/모) — "박모"/"김씨"처럼 이름이 아닌 요소가
  스팬 끝에 섞여 들어가는 것 방지 (모: "김모씨"류 언론 관행상 익명 표기)
- 공백을 가로지르는 스팬 제외 — "경기도 여주"처럼 gold가 이미 별개 단어로 나눠둔
  것을 하나로 뭉치면 회귀 유발 (Kiwi 기본분석이 각 단어를 이미 올바르게 태깅하므로
  강제 병합이 불필요함)
- 트림/필터 후 1글자만 남는 스팬 제외 — Kiwi가 단일 음절 성씨는 이미 NNP로 잘
  분석하므로 강제할 필요가 없고, 근거가 약한 단발성 오탐을 줄인다
"""
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "klue_roberta_large_ner")

CONFIDENCE_THRESHOLD = 0.80
KEEP_TYPES = {"PS", "LC", "OG"}
BOUND_HONORIFIC_SUFFIXES = {"씨", "군", "양", "님", "옹", "모"}
MIN_SPAN_LEN = 2


class LocalNERHandler:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device).eval()
            self.id2label = self.model.config.id2label
        except Exception:
            self.tokenizer = None
            self.model = None

    @torch.no_grad()
    def detect_nnp(self, sentence: str) -> list[tuple[int, int, str]]:
        """(start, end, "NNP") 리스트 반환. 실패 시 빈 리스트."""
        if not self.model:
            return []
        try:
            enc = self.tokenizer(
                sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[0]
        except Exception:
            return []

        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(-1).tolist()
        pred_probs = probs.max(-1).values.tolist()
        labels = [self.id2label[i] for i in pred_ids]

        raw_spans = []
        cur_type = cur_s = cur_e = None
        cur_probs = []

        def flush():
            if cur_type is not None and cur_probs:
                mean_p = sum(cur_probs) / len(cur_probs)
                if mean_p >= CONFIDENCE_THRESHOLD:
                    raw_spans.append((cur_s, cur_e))

        for (s, e), lab, p in zip(offsets, labels, pred_probs):
            if s == e:
                continue
            typ = lab[2:] if lab != "O" else None
            if lab.startswith("B-") and typ in KEEP_TYPES:
                flush()
                cur_type, cur_s, cur_e, cur_probs = typ, s, e, [p]
            elif lab.startswith("I-") and typ == cur_type and cur_type is not None:
                cur_e = e
                cur_probs.append(p)
            else:
                flush()
                cur_type, cur_probs = None, []
        flush()

        pre_tokens = []
        for s, e in raw_spans:
            if " " in sentence[s:e]:
                continue
            if e - s > 1 and sentence[e - 1] in BOUND_HONORIFIC_SUFFIXES:
                e -= 1
            if e - s < MIN_SPAN_LEN:
                continue
            pre_tokens.append((s, e, "NNP"))
        return pre_tokens
