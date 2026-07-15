"""KoELECTRA gold.jsonl(문자 오프셋 정답) 기준으로 룰베이스 추출기 평가.

gold.jsonl: 각 줄 {num, text(자모분해 무공백), morphs, spans[{label,start,end(exclusive)}]}
  - spans의 start/end는 text(=''.join(morphs))의 '문자 오프셋'(end-exclusive)
  - 라벨은 추출기 출력과 동일한 SpanLabels 코드(CP/AdjP/EmC_Adj/...)

원본 추출기(yumin 사본=수정 전) vs 현재(수정본)를 같은 gold로 비교한다.
자연문(공백 포함)은 Kiwi 실행에 필요하므로 Excel 정답에서 num으로 가져온다.
비교는 정답 형태소 ↔ Kiwi 형태소 difflib 정렬 후 Kiwi 문자 오프셋 공간에서 수행.
"""
import os, sys, json, importlib.util
os.chdir("/data/edutem/seongmoon/seongmoon_Syntactic_analysis/korean_analysis_phase3")
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict
from core.analyzer import kiwi_morphs, parse_dep
from eval_harness import load_parser_kiwi, gold_morph_to_kiwi_charspans, gold_span_to_char
from gold_parser import load_gold

GOLD_JSONL = "../KoELECTRA/eval/gold.jsonl"
ORIG_PATH = "/data/edutem/yumin/korean_analysis_phase3/span_extraction/morpheme_span_extractor_klue_phase3.py"
CUR_PATH = "span_extraction/morpheme_span_extractor_klue_phase3.py"


def load_extractor(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_spans_klue


def load_gold_jsonl(path=GOLD_JSONL):
    """char-오프셋 스팬을 형태소 index 스팬으로 변환."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        morphs = d["morphs"]
        # 형태소별 [char_start, char_end)
        off = []; c = 0
        for m in morphs:
            off.append((c, c + len(m))); c += len(m)

        def char_to_morph(ch, end=False):
            # end=True면 exclusive 끝 문자(ce-1)가 속한 형태소
            target = ch - 1 if end else ch
            for i, (a, b) in enumerate(off):
                if a <= target < b:
                    return i
            return len(morphs) - 1 if end else 0

        spans = []
        for sp in d["spans"]:
            if sp["label"] == "Sentence":
                continue
            gs = char_to_morph(sp["start"])
            ge = char_to_morph(sp["end"], end=True)
            spans.append((sp["label"], gs, ge))
        out.append({"num": d["num"], "morphs": morphs, "spans": spans})
    return out


def evaluate(extract_fn, gold_jsonl, texts_by_num, parser, kiwi):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    per_sent = []
    for g in gold_jsonl:
        text = texts_by_num[g["num"]]
        tokens, xpos, spans, lemmas = kiwi_morphs(kiwi, text)
        _, _, arcs, rels = parse_dep(parser, tokens)
        pred = extract_fn(tokens, xpos, arcs, rels)

        gmap = gold_morph_to_kiwi_charspans(g["morphs"], tokens, spans)
        gold_ch = []
        for lab, gsi, gei in g["spans"]:
            cs, ce = gold_span_to_char(gsi, gei, gmap)
            if cs is not None:
                gold_ch.append((lab, cs, ce))
        pred_ch = []
        for lab, psi, pei in pred:
            if lab == "Sentence":
                continue
            pred_ch.append((lab, spans[psi][0], spans[pei][1]))

        gset, pset = set(gold_ch), set(pred_ch)
        for lab, cs, ce in (gset & pset):
            tp[lab] += 1
        for lab, cs, ce in (gset - pset):
            fn[lab] += 1
        for lab, cs, ce in (pset - gset):
            fp[lab] += 1
        per_sent.append((g["num"], gset, pset))
    return tp, fp, fn, per_sent


def totals(tp, fp, fn):
    T = sum(tp.values()); F = sum(fp.values()); N = sum(fn.values())
    P = T / (T + F) if T + F else 0
    R = T / (T + N) if T + N else 0
    Fs = 2 * P * R / (P + R) if P + R else 0
    return T, F, N, P, R, Fs


def main():
    parser, kiwi = load_parser_kiwi()
    gold_jsonl = load_gold_jsonl()
    # 자연문 텍스트는 Excel 정답에서 num으로
    excel = load_gold("../docs/문장분석_정답_데이터_0410_수정.xlsx")
    texts_by_num = {d["num"]: d["text"] for d in excel}

    orig_fn = load_extractor(ORIG_PATH, "extractor_orig")
    cur_fn = load_extractor(CUR_PATH, "extractor_cur")

    print("gold.jsonl 기준 평가 (원본 vs 수정본)\n")
    results = {}
    for name, fn in [("원본(수정전)", orig_fn), ("수정본(현재)", cur_fn)]:
        tp, fp, fnn, _ = evaluate(fn, gold_jsonl, texts_by_num, parser, kiwi)
        results[name] = (tp, fp, fnn)

    labels = sorted(set().union(*[set(r[0]) | set(r[1]) | set(r[2]) for r in results.values()]))
    print(f"{'label':11s} | {'원본 TP/FP/FN  F1':>20s} | {'수정본 TP/FP/FN  F1':>20s}")
    print("-" * 60)
    for L in labels:
        row = f"{L:11s} |"
        for name in ("원본(수정전)", "수정본(현재)"):
            tp, fp, fnn = results[name]
            t, f, n = tp[L], fp[L], fnn[L]
            P = t/(t+f) if t+f else 0; R = t/(t+n) if t+n else 0
            F1 = 2*P*R/(P+R) if P+R else 0
            row += f" {t:2d}/{f:2d}/{n:2d}  {F1:.2f} |"
        print(row)
    print("-" * 60)
    for name in ("원본(수정전)", "수정본(현재)"):
        T, F, N, P, R, Fs = totals(*results[name])
        print(f"{name}:  TP={T} FP={F} FN={N}  P={P:.3f} R={R:.3f} F1={Fs:.3f}")


if __name__ == "__main__":
    main()
