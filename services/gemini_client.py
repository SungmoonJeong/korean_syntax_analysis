# gemini_client.py
"""Gemini로 고유명사(NNP) 스팬만 추출 — kiwi_upgrade/add_gemini의 프로덕션 버전.

역할: kiwi_morphs() 이전에 원문 문장을 Gemini에 보내 고유명사 스팬을 뽑고,
core.analyzer.kiwi_morphs()의 pretokenized 인자로 넘겨 Kiwi 분석 시점에
그 위치를 NNP로 고정한다. 실패(키 없음/API 오류/형식 불일치)해도 빈 리스트를
반환해 원본 Kiwi 분석 그대로 진행되도록 한다(교정 실패가 파이프라인을
막지 않음).
"""
import json

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-flash-latest"
CONFIDENCE_THRESHOLD = 0.80

# 이름 뒤에 붙는 호칭성 의존명사(NNB) — Kiwi는 이들을 원래 정확히 분리해서
# 태깅하는데(예: 박씨 -> 박/NNP + 씨/NNB), Gemini가 스팬 경계를 이 글자까지
# 포함해서 반환하면 pretokenized가 그 잘못된 경계를 강제해버려 오히려
# Kiwi의 기본 판단을 깨뜨린다. 닫힌 소집합이라 사전 조회보다 안전하게
# 스팬 끝에서 이 글자만 잘라낸다.
BOUND_HONORIFIC_SUFFIXES = {"씨", "군", "양", "님", "옹"}

SYSTEM_INSTRUCTION = (
    """당신은 한국어 고유명사(NNP) 추출 전문가입니다.
    입력된 문장에서 고유명사를 찾아 반드시 아래 형식의 JSON 배열로만 응답하세요.
    [{"word": "단어", "start": 시작인덱스, "end": 끝인덱스, "tag": "NNP", "confidence": 0.9}]

    고유명사 기준:
    - '사람 이름', '지명', '기관/단체/브랜드명', '제품/상표명', '저작물/문화 콘텐츠', '사건/행사명/기념일', '천체', '신조어'
    - 같은 형태라도 고유명사로 쓰인 것만 추출 (일반명사는 제외)
    - 외래어로 표현하는 일반명사는 제외 (이벤트, 티켓 등)

    주의사항:
    - 인덱스(start, end)는 반드시 0부터 시작하는 파이썬 문자열 슬라이싱 기준으로 정확하게 계산할 것
    - 모든 판단은 무조건 문맥을 고려할 것
    - 문맥상 이름처럼 쓰였다면 일반 명사라도 추출할 것
    - 주어진 문장에 없는 단어를 만들어내거나 임의로 변환시키지 말 것
    - 명사의 끝과 명사 끝에 붙는 조사의 범위를 확실하게 구별할 것
    - 동일한 단어가 반복되더라도 무조건 매 경우를 검토할 것"""
)


class GeminiHandler:
    def __init__(self, api_key: str):
        # 재시도 1회(=재시도 없음)로 고정 — 429(할당량 초과) 등으로 실패할 때 SDK가
        # 서버 권고 대기시간만큼 기다렸다가 재시도하면서 매 요청 앞단에 수 초가
        # 그대로 얹힌다. 실패는 어차피 빈 리스트로 폴백되므로, 느리게 실패하느니
        # 빠르게 실패하는 쪽이 낫다.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ) if api_key else None

    def detect_nnp(self, sentence: str) -> list[tuple[int, int, str]]:
        """(start, end, "NNP") 리스트 반환. 실패 시 빈 리스트."""
        if not self.client:
            return []
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=sentence,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            items = json.loads(response.text)
        except Exception:
            return []

        pre_tokens = []
        for item in items:
            try:
                start, end = item["start"], item["end"]
                word = item["word"]
                confidence = item["confidence"]
            except (KeyError, TypeError):
                continue
            if confidence >= CONFIDENCE_THRESHOLD and sentence[start:end] == word:
                if end - start > 1 and sentence[end - 1] in BOUND_HONORIFIC_SUFFIXES:
                    end -= 1
                pre_tokens.append((start, end, "NNP"))
        return pre_tokens
