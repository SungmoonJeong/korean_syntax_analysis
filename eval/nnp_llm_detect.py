"""Claude(Sonnet 5)로 고유명사(NNP) 판별 — kiwi_upgrade/add_gemini의 Claude API 버전.

목적: 베이스라인 진단(eval_kiwi_klue_dev.py, diag_segmentation_*.py)에서 확인한
NNP/NNG 오류(태그 전체 오류의 12.8%, 분절 오류에서도 최준희/류승우/수월리 등
고유명사 오분해가 반복)를 규칙으로 못 끝내는 열린집합 문제로 보고, 문장에서
고유명사 스팬만 좁게 뽑아 kiwi.analyze(..., pretokenized=...)로 위치를 강제
고정하는 add_gemini_index.py와 같은 접근을 Claude API로 재현한다.

기존 add_gemini와 다른 점:
  - Gemini의 response_mime_type(형식 강제 없음, json.loads만) 대신
    output_config.format(JSON 스키마 강제)을 사용 — 형식이 어긋난 응답 자체가
    안 나오므로 후처리 파싱 실패 케이스가 줄어든다.
  - 모델은 Sonnet 5(claude-sonnet-5) 사용 — 문장당 반복 호출되는 단순 분류
    작업이라 비용 대비 이 선택.

아직 파이프라인에 연결하지 않은 진단 스크립트 — kiwi 대비 정확도 확인용.
"""
import os
import sys
import json

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from dotenv import load_dotenv
from kiwipiepy import Kiwi
import anthropic

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "당신은 한국어 고유명사(NNP) 추출 전문가입니다. "
    "입력된 문장에서 고유명사(사람 이름, 지명, 기관/단체/브랜드명, 제품/작품명, "
    "행사/사건명 등)만 찾아 JSON으로 반환하세요.\n"
    "- 같은 형태라도 고유명사로 쓰인 것만 추출 (일반명사는 제외)\n"
    "- 인덱스(start, end)는 파이썬 문자열 슬라이싱 기준(0-based, end exclusive)으로 정확히 계산\n"
    "- 문장에 없는 단어를 만들어내거나 임의로 변형하지 말 것"
)

NNP_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "confidence": {"type": "number"},
                },
                "required": ["word", "start", "end", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

CONFIDENCE_THRESHOLD = 0.80


def detect_nnp(client, sentence: str):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": sentence}],
        output_config={"format": {"type": "json_schema", "schema": NNP_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    entities = []
    for e in data["entities"]:
        if e["confidence"] >= CONFIDENCE_THRESHOLD and sentence[e["start"]:e["end"]] == e["word"]:
            entities.append(e)
    return entities


def kiwi_current(kiwi, sentence: str):
    return [(t.form, t.tag) for t in kiwi.tokenize(sentence)]


# 지난번 진단(diag_segmentation_mismatch.py, klue_dev_confusion_breakdown.json)에서
# 확인한 실제 NNP 관련 오류/오분해 예문
TEST_SENTENCES = [
    "오는 2일 오후 11시 10분에 방송되는 이번 방송분은 '예체능' 테니스단이 경기도 여주의 숲 속 실내 테니스장에서 혹한기 지옥훈련에 돌입하는 모습을 그렸다.",
    "카투사 합격자는 6일 오후 5시에 발표됐지만, 스윙스의 합격 여부는 아직 알려지지 않았습니다.",
    "고 최진실 씨의 딸 최준희(11)양이 실시간 인터넷 방송서비스에 개인 방송을 시작한 이후 악성 댓글이 달려 누리꾼들이 우려의 목소리를 높이고 있다.",
    "이에 페인을 비롯한 호주 선수들이 류승우 선수에게 크게 항의했고 국내 축구팬들 역시 류승수 선수의 플레이를 지적했다.",
    "통영해양경비안전서는 지난 14일 통영시 도산면 수월리 앞바다에서 발견된 승용차와 그 주변에서 수습한 유골의 신원이 통영시에 살던 김모(56)씨와 문모(57·여)씨 부부로 추정된다고 19일 밝혔다.",
    "박 검사직무대리는 이번 사건을 마치는 대로 원래의 자리인 광주고검 제주지부로 돌아가게 된다.",
]


def main():
    if not ANTHROPIC_API_KEY:
        print("[경고] ANTHROPIC_API_KEY가 설정되지 않았습니다. .env에 추가한 뒤 다시 실행하세요.")
        return
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kiwi = Kiwi()

    for sent in TEST_SENTENCES:
        print("=" * 90)
        print("원문:", sent)
        entities = detect_nnp(client, sent)
        print("Claude NNP:", [(e["word"], f"{e['start']}:{e['end']}", e["confidence"]) for e in entities])
        kiwi_nnp = [f"{f}/{t}" for f, t in kiwi_current(kiwi, sent) if t in ("NNP", "NNG")]
        print("Kiwi NNP/NNG 토큰:", kiwi_nnp)


if __name__ == "__main__":
    main()
