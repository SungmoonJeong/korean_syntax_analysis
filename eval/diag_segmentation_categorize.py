"""분절 불일치(replace opcode)를 패턴별로 자동 분류해서 비중을 집계.

diag_segmentation_mismatch.py의 표본 관찰에서 나온 3대 패턴을 전체 dev set
1,915건의 replace opcode에 규칙으로 매칭해 실제 비중을 잰다:
  A. 하다-형용사: gold 1토큰(VA) vs kiwi 2토큰([XR, XSA])
  B. 동사 압축: gold 쪽에 VV/EC/VX/NNG 등 여러 토큰, kiwi 쪽은 VV 1토큰(또는 반대)
  C. 복합명사 경계: 양쪽 다 명사류(NNG/NNP/XSN/XPN/NNB) 토큰들의 개수만 다름
  D. 그 외(고유명사 오분석 포함)
"""
import os
import sys
import difflib
from collections import Counter

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.analyzer import kiwi_morphs
from eval_harness import load_parser_kiwi, norm
from eval_kiwi_klue_dev import load_conllu, clean_tag, CONLLU_PATH

NOMINAL = {"NNG", "NNP", "XSN", "XPN", "NNB", "MM", "MMN", "MMA", "MMD", "NR", "SL", "SN", "SH"}
VERBISH = {"VV", "VA", "VX", "EC", "EP", "ETM", "ETN"}


def classify(g_tags, k_tags):
    if g_tags == ["VA"] and k_tags == ["XR", "XSA"]:
        return "A_하다형용사"
    if k_tags == ["VA"] and g_tags == ["XR", "XSA"]:
        return "A_하다형용사"

    g_has_verb = any(t in ("VV", "VA") for t in g_tags)
    k_has_verb = any(t in ("VV", "VA") for t in k_tags)
    if len(g_tags) == 1 and g_tags[0] in ("VV", "VA") and len(k_tags) > 1 and k_has_verb:
        return "B_동사압축"
    if len(k_tags) == 1 and k_tags[0] in ("VV", "VA") and len(g_tags) > 1 and g_has_verb:
        return "B_동사압축"

    if all(t in NOMINAL for t in g_tags) and all(t in NOMINAL for t in k_tags):
        return "C_복합명사경계"

    return "D_기타"


def main():
    parser, kiwi = load_parser_kiwi()
    sents = load_conllu(CONLLU_PATH)

    cat_counter = Counter()
    cat_examples = {}

    for s in sents:
        text = s["text"]
        gold_rows = s["rows"]
        gforms = [r["form"] for r in gold_rows]
        gxpos = [r["xpos"] for r in gold_rows]

        tokens, xpos, spans, lemmas = kiwi_morphs(kiwi, text)

        gn = [norm(m) for m in gforms]
        kn = [norm(m) for m in tokens]
        sm = difflib.SequenceMatcher(a=gn, b=kn, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace":
                continue
            g_tags = gxpos[i1:i2]
            k_tags = [clean_tag(t) for t in xpos[j1:j2]]
            cat = classify(g_tags, k_tags)
            cat_counter[cat] += 1
            if len(cat_examples.setdefault(cat, [])) < 6:
                g_seg = " ".join(f"{gforms[i]}/{gxpos[i]}" for i in range(i1, i2))
                k_seg = " ".join(f"{tokens[j]}/{clean_tag(xpos[j])}" for j in range(j1, j2))
                cat_examples[cat].append((g_seg, k_seg))

    total = sum(cat_counter.values())
    print(f"전체 replace opcode: {total}건\n")
    for cat, cnt in cat_counter.most_common():
        pct = cnt / total * 100
        print(f"{cat:20s} {cnt:5d}건  ({pct:5.1f}%)")
        for g_seg, k_seg in cat_examples[cat]:
            print(f"    gold: {g_seg:35s} | kiwi: {k_seg}")
        print()


if __name__ == "__main__":
    main()
