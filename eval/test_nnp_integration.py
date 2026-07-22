"""main.py에 반영한 Gemini NNP 훅(services/gemini_client.py + core/analyzer.kiwi_morphs
의 pretokenized 인자)이 실제로 동작하는지 확인하는 스모크 테스트.

kiwi_upgrade/add_gemini/test_problem_sentences.py와 같은 문제 문장으로,
프로덕션 모듈(services.gemini_client, core.analyzer)을 그대로 가져와 검증한다.
SuPar/CE 모델은 무거워서 로드하지 않고, kiwi_morphs까지만 확인한다.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from dotenv import load_dotenv
from kiwipiepy import Kiwi

from core.analyzer import kiwi_morphs
from services.gemini_client import GeminiHandler

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TEST_SENTENCES = [
    "고 최진실 씨의 딸 최준희(11)양이 실시간 인터넷 방송서비스에 개인 방송을 시작한 이후 악성 댓글이 달려 누리꾼들이 우려의 목소리를 높이고 있다.",
    "이에 페인을 비롯한 호주 선수들이 류승우 선수에게 크게 항의했고 국내 축구팬들 역시 류승수 선수의 플레이를 지적했다.",
    "박 검사직무대리는 이번 사건을 마치는 대로 원래의 자리인 광주고검 제주지부로 돌아가게 된다.",
]


def main():
    if not GEMINI_API_KEY:
        print("[경고] GEMINI_API_KEY가 없습니다. .env를 확인하세요.")
        return

    handler = GeminiHandler(GEMINI_API_KEY)
    kiwi = Kiwi(integrate_allomorph=False)

    for sent in TEST_SENTENCES:
        print("=" * 90)
        print("원문:", sent)

        tokens0, xpos0, _, _ = kiwi_morphs(kiwi, sent)
        base = [(f, t) for f, t in zip(tokens0, xpos0) if t in ("NNP", "NNG")]
        print("훅 없이 kiwi_morphs:", base)

        nnp_spans = handler.detect_nnp(sent)
        print("Gemini 스팬:", nnp_spans)

        tokens1, xpos1, _, _ = kiwi_morphs(kiwi, sent, pretokenized=nnp_spans)
        fixed = [(f, t) for f, t in zip(tokens1, xpos1) if t in ("NNP", "NNG")]
        print("훅 적용 후 kiwi_morphs:", fixed)


if __name__ == "__main__":
    main()
