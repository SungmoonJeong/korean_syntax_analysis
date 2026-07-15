"""정답 형태소 시퀀스를 원문 문자(char) 오프셋에 정렬.

Kiwi와 정답의 형태소 분절이 달라(자모 표기/하 분리/빈칸 등) 형태소 인덱스로
직접 비교가 불가능하므로, 양측 스팬을 '원문 문자 오프셋' 공간으로 투영해서 비교한다.

핵심: 정답 형태소 각각이 '공백 제거한 원문'의 어느 char 구간을 덮는지 구한다.
한글 자모 분해로 매칭한다 (예: '아니'+'ㄴ' → '아닌').
"""

# 초성/중성/종성 (호환 자모로 표기)
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

# 겹받침 분해 (종성 복합 자모 → 단일 자모들)
JONG_SPLIT = {
    "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ",
    "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ",
}
# conjoining 자모(ᆫ 등) → 호환 자모(ㄴ) 정규화
_CONJOIN = {
    "ᆨ":"ㄱ","ᆩ":"ㄲ","ᆪ":"ㄳ","ᆫ":"ㄴ","ᆬ":"ㄵ","ᆭ":"ㄶ","ᆮ":"ㄷ","ᆯ":"ㄹ",
    "ᆰ":"ㄺ","ᆱ":"ㄻ","ᆲ":"ㄼ","ᆳ":"ㄽ","ᆴ":"ㄾ","ᆵ":"ㄿ","ᆶ":"ㅀ","ᆷ":"ㅁ",
    "ᆸ":"ㅂ","ᆹ":"ㅄ","ᆺ":"ㅅ","ᆻ":"ㅆ","ᆼ":"ㅇ","ᆽ":"ㅈ","ᆾ":"ㅊ","ᆿ":"ㅋ",
    "ᇀ":"ㅌ","ᇁ":"ㅍ","ᇂ":"ㅎ",
    "ᄀ":"ㄱ","ᄂ":"ㄴ","ᄃ":"ㄷ","ᄅ":"ㄹ","ᄆ":"ㅁ","ᄇ":"ㅂ","ᄉ":"ㅅ","ᄋ":"ㅇ",
    "ᄌ":"ㅈ","ᄎ":"ㅊ","ᄏ":"ㅋ","ᄐ":"ㅌ","ᄑ":"ㅍ","ᄒ":"ㅎ",
}


def _jamo_units(ch):
    """한 글자를 호환자모 리스트로 분해."""
    if ch in _CONJOIN:
        ch = _CONJOIN[ch]
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:  # 완성형 음절
        s = o - 0xAC00
        cho, jung, jong = s // 588, (s % 588) // 28, s % 28
        out = [CHO[cho], JUNG[jung]]
        if jong:
            jc = JONG[jong]
            out.extend(JONG_SPLIT.get(jc, jc))
        return out
    # 이미 낱자모(호환)거나 기타 문자
    if ch in JONG_SPLIT:
        return list(JONG_SPLIT[ch])
    return [ch]


def decompose_seq(s):
    """문자열 s -> [(jamo, src_char_index), ...]"""
    out = []
    for i, ch in enumerate(s):
        for j in _jamo_units(ch):
            out.append((j, i))
    return out


def align_gold_to_chars(text, morphs):
    """정답 형태소 -> 각 형태소의 (char_start, char_end) (공백 제거 원문 기준, inclusive).

    빈 형태소('')는 (None,None). 매칭 실패 형태소도 (None,None).
    반환: (spans_per_morph, nospace_text)
    """
    nospace = "".join(text.split())
    surf = decompose_seq(nospace)  # [(jamo, char_idx)]
    p = 0  # surf 포인터
    result = []
    for m in morphs:
        if m is None or m.strip() == "":
            result.append((None, None))
            continue
        mj = [j for j, _ in decompose_seq(m)]
        if not mj:
            result.append((None, None))
            continue
        # surf[p:] 에서 mj 를 순차 매칭
        start_char = None
        end_char = None
        q = 0
        pp = p
        while q < len(mj) and pp < len(surf):
            if surf[pp][0] == mj[q]:
                if start_char is None:
                    start_char = surf[pp][1]
                end_char = surf[pp][1]
                q += 1
                pp += 1
            else:
                # 불일치: surf 한 칸 건너뛰기 (희귀 - 안전장치)
                pp += 1
                if start_char is not None:
                    # 이미 매칭 시작했는데 끊기면 포기
                    break
        if q == len(mj):
            result.append((start_char, end_char))
            p = pp
        else:
            # 매칭 실패 - 포인터 유지
            result.append((None, None))
    return result, nospace


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gold_parser import load_gold
    gold = load_gold("/data/edutem/seongmoon/seongmoon_Syntactic_analysis/docs/문장분석_정답_데이터_0410_수정.xlsx")
    for d in gold[:3]:
        print("="*70)
        print("#", d["num"], d["text"])
        spans, nospace = align_gold_to_chars(d["text"], d["morphs"])
        for m, (s, e) in zip(d["morphs"], spans):
            seg = nospace[s:e+1] if s is not None else "∅"
            print(f"   {m!r:12s} -> [{s},{e}] {seg!r}")
        # 스팬을 char로
        print("  --- 스팬(char) ---")
        for lab, gs, ge in d["spans"]:
            cs = spans[gs][0]
            ce = spans[ge][1]
            # 시작이 None이면 뒤로, 끝이 None이면 앞으로 보정
            i = gs
            while cs is None and i <= ge:
                cs = spans[i][0]; i += 1
            i = ge
            while ce is None and i >= gs:
                ce = spans[i][1]; i -= 1
            seg = nospace[cs:ce+1] if cs is not None else "?"
            print(f"     [{cs:3},{ce:3}] {lab:32s} {seg!r}")
