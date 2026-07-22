"""Gemini NNP 훅을 켰을 때/껐을 때를 같은 문장 집합에 대해 비교.

klue-dev-morph.conllu 기준으로 eval_kiwi_klue_dev.py와 같은 정렬·정확도 계산
방식을 재사용하되, 매 문장마다 (a) Gemini 없이 kiwi_morphs, (b) Gemini
detect_nnp로 얻은 스팬을 pretokenized로 넘긴 kiwi_morphs를 둘 다 돌려서
같은 표본으로 직접 비교한다. 서로 다른 표본을 비교하면 "표본이 달라서
생긴 차이"와 "Gemini 효과"가 섞여버리므로, 반드시 같은 문장으로 비교.
"""
import os
import sys
import time
import difflib
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.analyzer import kiwi_morphs
from eval_harness import load_parser_kiwi, norm
from eval_kiwi_klue_dev import load_conllu, clean_tag, CONLLU_PATH
from services.gemini_client import GeminiHandler

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def score_one(gforms, gxpos, tokens, xpos):
    """정렬된 토큰 쌍의 (aligned, correct) 카운트를 반환."""
    gn = [norm(m) for m in gforms]
    kn = [norm(m) for m in tokens]
    sm = difflib.SequenceMatcher(a=gn, b=kn, autojunk=False)
    aligned = correct = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                aligned += 1
                if gxpos[i1 + off] == clean_tag(xpos[j1 + off]):
                    correct += 1
    return aligned, correct


def main(limit=20):
    if not GEMINI_API_KEY:
        print("[경고] GEMINI_API_KEY가 없습니다.")
        return

    print("리소스 로딩 중 (Kiwi + SuPar)...", file=sys.stderr)
    _, kiwi = load_parser_kiwi()
    gemini = GeminiHandler(GEMINI_API_KEY)

    sents = load_conllu(CONLLU_PATH)[:limit]
    print(f"대상 문장: {len(sents)}개", file=sys.stderr)

    base_aligned = base_correct = 0
    gem_aligned = gem_correct = 0
    changed_sentences = []
    t0 = time.time()

    for si, s in enumerate(sents):
        text = s["text"]
        gforms = [r["form"] for r in s["rows"]]
        gxpos = [r["xpos"] for r in s["rows"]]

        tokens0, xpos0, _, _ = kiwi_morphs(kiwi, text)
        a0, c0 = score_one(gforms, gxpos, tokens0, xpos0)
        base_aligned += a0
        base_correct += c0

        nnp_spans = gemini.detect_nnp(text)
        tokens1, xpos1, _, _ = kiwi_morphs(kiwi, text, pretokenized=nnp_spans)
        a1, c1 = score_one(gforms, gxpos, tokens1, xpos1)
        gem_aligned += a1
        gem_correct += c1

        if tokens0 != tokens1 or xpos0 != xpos1:
            changed_sentences.append({
                "text": text,
                "nnp_spans": nnp_spans,
                "before": list(zip(tokens0, [clean_tag(t) for t in xpos0])),
                "after": list(zip(tokens1, [clean_tag(t) for t in xpos1])),
                "acc_before": (a0, c0),
                "acc_after": (a1, c1),
            })

        elapsed = time.time() - t0
        print(f"  ...{si+1}/{len(sents)} (경과 {elapsed:.0f}초)", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"표본: {len(sents)}문장")
    print(f"Gemini 없음: {base_correct}/{base_aligned} = {base_correct/base_aligned*100:.2f}%")
    print(f"Gemini 적용: {gem_correct}/{gem_aligned} = {gem_correct/gem_aligned*100:.2f}%")
    print(f"바뀐 문장 수: {len(changed_sentences)}개")
    print("=" * 70)

    for cs in changed_sentences:
        print("\n원문:", cs["text"])
        print("  Gemini 스팬:", cs["nnp_spans"])
        print("  변경 전 acc:", cs["acc_before"], " 변경 후 acc:", cs["acc_after"])
        # 실제 달라진 토큰만 표시
        for (bf, bt), (af, at) in zip(cs["before"], cs["after"]) if len(cs["before"]) == len(cs["after"]) else []:
            if (bf, bt) != (af, at):
                print(f"    {bf}/{bt} -> {af}/{at}")
        if len(cs["before"]) != len(cs["after"]):
            print(f"    (토큰 개수 변경: {len(cs['before'])} -> {len(cs['after'])})")
            print("    변경 전:", cs["before"])
            print("    변경 후:", cs["after"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(limit=n)
