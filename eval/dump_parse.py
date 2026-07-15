"""특정 정답 문장의 형태소/품사/arc/rel 및 추출 결과를 덤프."""
import os, sys
os.chdir("/data/edutem/seongmoon/seongmoon_Syntactic_analysis/korean_analysis_phase3")
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.analyzer import kiwi_morphs, parse_dep
from span_extraction.morpheme_span_extractor_klue_phase3 import extract_spans_klue
from gold_parser import load_gold
from eval_harness import load_parser_kiwi

parser, kiwi = load_parser_kiwi()
gold = load_gold("../docs/문장분석_정답_데이터_0410_수정.xlsx")
nums = [int(x) for x in sys.argv[1:] if x.isdigit()] or [1]
for d in gold:
    if d["num"] not in nums: continue
    text = d["text"]
    tokens, xpos, spans, lemmas = kiwi_morphs(kiwi, text)
    _, _, arcs, rels = parse_dep(parser, tokens)
    print("="*80); print(f"#{d['num']} {text}\n")
    print(f"{'i':>2} {'form':10s} {'xpos':7s} {'head':>4} {'rel':10s} {'child_of_head_form'}")
    for i,(t,x,h,r) in enumerate(zip(tokens,xpos,arcs,rels)):
        hf = tokens[h-1] if h>0 else "ROOT"
        print(f"{i:2d} {t:10s} {x:7s} {h:4d} {r:10s} -> {hf}")
    pred = extract_spans_klue(tokens, xpos, arcs, rels)
    print("\n--- 추출 스팬 (형태소 index) ---")
    for lab,s,e in sorted(pred, key=lambda z:(z[1],z[2])):
        print(f"   [{s:2d}-{e:2d}] {lab:12s} {''.join(tokens[s:e+1])}")
    print("\n--- 정답 스팬 (정답 형태소 index / 표면) ---")
    for lab,s,e in d["spans"]:
        print(f"   [{s:2d}-{e:2d}] {lab:32s} {' '.join(d['morphs'][s:e+1])}")
