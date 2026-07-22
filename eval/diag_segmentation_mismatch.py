"""klue-dev-morph.conllu 대비 Kiwi 분절 불일치(60.8%) 원인 진단.

eval_kiwi_klue_dev.py에서 "완전 정렬 안 되는 문장" 비율(60.8%)만 확인했고 왜
그런지는 안 봤다. 이 스크립트는 그 문장들을 뽑아서 gold 토큰 시퀀스와 Kiwi
토큰 시퀀스를 나란히 놓고 difflib opcode 단위로 어디가 다른지 보여준다.

목적: 분절 불일치가 (a) 이 gold 데이터가 만들어질 때 쓰인 분절 관례와 지금
Kiwi 버전의 표기 관례 차이(진짜 문제 아님)인지, (b) 진짜 Kiwi 분절 오류인지
구분하기 위한 표본 조사. 자동 판정은 안 하고 사람이 읽을 수 있는 diff만 출력.
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


def diff_view(gforms, gxpos, tokens, xpos):
    gn = [norm(m) for m in gforms]
    kn = [norm(m) for m in tokens]
    sm = difflib.SequenceMatcher(a=gn, b=kn, autojunk=False)
    lines = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        g_seg = " ".join(f"{gforms[i]}/{gxpos[i]}" for i in range(i1, i2))
        k_seg = " ".join(f"{tokens[j]}/{clean_tag(xpos[j])}" for j in range(j1, j2))
        lines.append((tag, g_seg or "∅", k_seg or "∅"))
    return lines


def main(n_samples=40, max_diff_ops=3):
    parser, kiwi = load_parser_kiwi()
    sents = load_conllu(CONLLU_PATH)

    picked = []
    op_type_counter = Counter()

    for s in sents:
        text = s["text"]
        gold_rows = s["rows"]
        gforms = [r["form"] for r in gold_rows]
        gxpos = [r["xpos"] for r in gold_rows]

        tokens, xpos, spans, lemmas = kiwi_morphs(kiwi, text)

        gn = [norm(m) for m in gforms]
        kn = [norm(m) for m in tokens]
        sm = difflib.SequenceMatcher(a=gn, b=kn, autojunk=False)
        opcodes = [oc for oc in sm.get_opcodes() if oc[0] != "equal"]
        if not opcodes:
            continue  # 완전 정렬 문장은 진단 대상 아님

        for oc in opcodes:
            op_type_counter[oc[0]] += 1

        if len(opcodes) <= max_diff_ops and len(picked) < n_samples:
            picked.append((text, gforms, gxpos, tokens, xpos))

    print(f"=== opcode 유형 집계 (분절 불일치 있는 모든 문장 기준) ===")
    for k, v in op_type_counter.most_common():
        print(f"  {k:10s} {v}")
    print()

    print(f"=== 표본 {len(picked)}개 (diff가 {max_diff_ops}개 이하로 단순한 문장 우선) ===\n")
    for text, gforms, gxpos, tokens, xpos in picked:
        print("=" * 90)
        print("원문:", text)
        for tag, g_seg, k_seg in diff_view(gforms, gxpos, tokens, xpos):
            print(f"  [{tag:7s}] gold: {g_seg:40s} | kiwi: {k_seg}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n_samples=n)
