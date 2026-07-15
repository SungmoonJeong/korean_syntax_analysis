"""정답 대비 추출기 평가 하네스.

전략:
 1) 실제 파이프라인(Kiwi -> SuPar DP -> 추출기)을 정답 문장에 실행
 2) 정답 형태소 <-> Kiwi 형태소를 difflib으로 정렬 (자모 정규화 후)
 3) 정답 스팬을 Kiwi char 오프셋 공간으로 투영 (스팬은 Kiwi char 구간이 됨)
 4) 추출기 스팬도 Kiwi char 오프셋으로 투영
 5) 라벨별 char-구간 정확매칭으로 P/R/F1 및 상세 비교 출력
"""
import os, sys, difflib
os.chdir("/data/edutem/seongmoon/seongmoon_Syntactic_analysis/korean_analysis_phase3")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiwipiepy import Kiwi
from supar import Parser
import torch, torch.serialization as ts
from config import KIWI_USER_WORDS, SPAN_DISPLAY
from core.analyzer import kiwi_morphs, parse_dep
from span_extraction.morpheme_span_extractor_klue_phase3 import extract_spans_klue
from gold_parser import load_gold

# 표시라벨(정답) -> SpanLabels 코드 (추출기 출력) 역매핑 (대소문자/공백 무시)
DISPLAY_TO_CODE = {v.strip().lower(): k for k, v in SPAN_DISPLAY.items()}
DISPLAY_TO_CODE["indirect quotation clause"] = "QuoteC_Ind"
DISPLAY_TO_CODE["direct quotation clause"] = "QuoteC_Dir"
def to_code(lab):
    return DISPLAY_TO_CODE.get(lab.strip().lower(), lab.strip())

_CONJOIN = {chr(c): d for c, d in [
    (0x11AB,'ㄴ'),(0x11AF,'ㄹ'),(0x11B7,'ㅁ'),(0x11B8,'ㅂ'),(0x11BA,'ㅅ'),(0x11BC,'ㅇ'),
    (0x11A8,'ㄱ'),(0x11BD,'ㅈ'),(0x11BE,'ㅊ'),(0x11C0,'ㅌ'),(0x11C1,'ㅍ'),(0x11C2,'ㅎ'),
    (0x11AE,'ㄷ'),
]}
def norm(m):
    if m is None: return ""
    return "".join(_CONJOIN.get(c, c) for c in m).strip()

def load_parser_kiwi():
    orig_load, orig_ts = torch.load, ts.load
    torch.load = lambda *a, **k: orig_load(*a, **{**k, "weights_only": False})
    ts.load = lambda *a, **k: orig_ts(*a, **{**k, "weights_only": False})
    try:
        parser = Parser.load("supar_morph_dp/model.pth", device="cpu")
    finally:
        torch.load, ts.load = orig_load, orig_ts
    kiwi = Kiwi(integrate_allomorph=False)
    for w, t, s in KIWI_USER_WORDS:
        kiwi.add_user_word(w, t, s)
    return parser, kiwi

def gold_morph_to_kiwi_charspans(gmorphs, ktokens, kspans):
    """정답 형태소 index -> (char_start,char_end). difflib 정렬로 kiwi char 빌림."""
    gn = [norm(m) for m in gmorphs]
    kn = [norm(m) for m in ktokens]
    sm = difflib.SequenceMatcher(a=gn, b=kn, autojunk=False)
    gmap = [None] * len(gmorphs)  # gold idx -> (cs,ce)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                cs, ce = kspans[j1 + off]
                gmap[i1 + off] = (cs, ce)
        elif tag == "replace":
            # gold[i1:i2] 전체를 kiwi[j1:j2] 전체 char 범위에 매핑
            if j2 > j1:
                cs = kspans[j1][0]; ce = kspans[j2 - 1][1]
            else:
                cs = ce = None
            for gi in range(i1, i2):
                gmap[gi] = (cs, ce)
        elif tag == "delete":
            # gold에만 있는 형태소(예: 빈칸, 하 분리 잔여) -> 인접 kiwi 위치
            cs = kspans[j1][0] if j1 < len(kspans) else (kspans[-1][1] if kspans else 0)
            for gi in range(i1, i2):
                gmap[gi] = (cs, cs)
        # insert(kiwi에만): 무시
    return gmap

def gold_span_to_char(gs, ge, gmap):
    cs = None; ce = None
    i = gs
    while cs is None and i <= ge:
        if gmap[i]: cs = gmap[i][0]
        i += 1
    i = ge
    while ce is None and i >= gs:
        if gmap[i]: ce = gmap[i][1]
        i -= 1
    return cs, ce

def main():
    parser, kiwi = load_parser_kiwi()
    gold = load_gold("../docs/문장분석_정답_데이터_0410_수정.xlsx")

    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    detail = []

    for d in gold:
        text = d["text"]
        tokens, xpos, spans, lemmas = kiwi_morphs(kiwi, text)
        _, _, arcs, rels = parse_dep(parser, tokens)
        pred = extract_spans_klue(tokens, xpos, arcs, rels)

        gmap = gold_morph_to_kiwi_charspans(d["morphs"], tokens, spans)
        # 정답 스팬 -> (code, cs, ce)
        gold_ch = []
        for lab, gsi, gei in d["spans"]:
            code = to_code(lab)
            if code == "Sentence":
                continue
            cs, ce = gold_span_to_char(gsi, gei, gmap)
            if cs is not None:
                gold_ch.append((code, cs, ce))
        # 추출 스팬 -> (code, cs, ce)
        pred_ch = []
        for lab, psi, pei in pred:
            if lab == "Sentence":
                continue
            cs = spans[psi][0]; ce = spans[pei][1]
            pred_ch.append((lab, cs, ce))

        gset = set(gold_ch); pset = set(pred_ch)
        matched = gset & pset
        for code, cs, ce in matched:
            tp[code] += 1
        only_g = gset - pset
        only_p = pset - gset
        for code, cs, ce in only_g:
            fn[code] += 1
        for code, cs, ce in only_p:
            fp[code] += 1
        detail.append((d, text, tokens, gold_ch, pred_ch, matched, only_g, only_p))

    # 요약
    print("="*78)
    print("라벨별 정확매칭(char-구간) P/R/F1")
    print("="*78)
    labels = sorted(set(list(tp)+list(fp)+list(fn)))
    tot_tp=tot_fp=tot_fn=0
    print(f"{'label':12s} {'TP':>4} {'FP':>4} {'FN':>4}  {'P':>5} {'R':>5} {'F1':>5}")
    for L in labels:
        t,f,n = tp[L],fp[L],fn[L]
        tot_tp+=t; tot_fp+=f; tot_fn+=n
        P = t/(t+f) if t+f else 0
        R = t/(t+n) if t+n else 0
        F = 2*P*R/(P+R) if P+R else 0
        print(f"{L:12s} {t:4d} {f:4d} {n:4d}  {P:5.2f} {R:5.2f} {F:5.2f}")
    P=tot_tp/(tot_tp+tot_fp) if tot_tp+tot_fp else 0
    R=tot_tp/(tot_tp+tot_fn) if tot_tp+tot_fn else 0
    F=2*P*R/(P+R) if P+R else 0
    print("-"*44)
    print(f"{'TOTAL':12s} {tot_tp:4d} {tot_fp:4d} {tot_fn:4d}  {P:5.2f} {R:5.2f} {F:5.2f}")

    # 상세 (문장별)
    if "--detail" in sys.argv:
        for d, text, tokens, gold_ch, pred_ch, matched, only_g, only_p in detail:
            print("\n" + "="*78)
            print(f"#{d['num']} {text}")
            def show(items):
                for code, cs, ce in sorted(items, key=lambda x:(x[1],x[2])):
                    print(f"      [{cs:3},{ce:3}] {code:12s} {text[cs:ce+1]!r}")
            print("   [일치]");           show(matched)
            print("   [정답에만-누락 FN]"); show(only_g)
            print("   [추출에만-오탐 FP]"); show(only_p)

if __name__ == "__main__":
    main()
