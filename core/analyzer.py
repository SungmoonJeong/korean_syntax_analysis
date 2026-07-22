# analyzer.py
"""원문 -> 형태소 분석 -> 의존구문 분석 -> 어절 그룹핑"""

import re

import torch
from supar import Parser


# ============================================================================
# 1. 형태소 분석 및 의존구문 분석
# ============================================================================
def kiwi_morphs(kiwi, text: str, pretokenized=None):
    """Kiwi morph tokenizer

    pretokenized: [(start, end, tag), ...] — 주어지면 그 구간을 지정된 품사로
    고정한다(예: services.gemini_client.GeminiHandler.detect_nnp가 찾은 고유명사
    스팬). 없으면 Kiwi 기본 분석 그대로.
    """
    kwargs = {"top_n": 1}
    if pretokenized:
        kwargs["pretokenized"] = pretokenized
    res = kiwi.analyze(text, **kwargs)[0]
    morphs = res[0] if isinstance(res, tuple) else res

    tokens, xpos, spans, lemmas = [], [], [], []
    for m in morphs:
        form = getattr(m, "form", str(m))
        lemma = getattr(m, "lemma", form)
        tag = getattr(m, "tag", "")
        # VV-I/VV-R 처럼 불규칙(-I) · 규칙(-R) 접미어 제거 — 글로싱에서는 구분 불필요
        tag = tag.split("-")[0] if isinstance(tag, str) else tag
        start = getattr(m, "start", 0)
        L = getattr(m, "length", None) or getattr(m, "len", None) or len(form)
        end = start + L

        # 과거 'EF+JX(요)' 분리 규칙은 제거됨.
        # Kiwi(0.23.2)는 'EF+JX' 복합 태그를 만들지 않는다 — 체언/부사 뒤 '요'는
        # 이미 '요/JX'로 분리해 주고, 용언 종결형 뒤 '요'는 '어요/네요/세요'처럼
        # 하나의 EF로 통합한다. 통합 EF는 glossing_rules의 RULE_FORM_POS에 통째로
        # 등록되어 있어(예: 어요→"Polite Casual Ending", 세요→"Polite Command"),
        # 인위적으로 '어/EF'+'요/JX'로 쪼개면 사전 매칭이 깨져 글로스가 열화된다.
        # 따라서 Kiwi 출력을 그대로 사용한다.
        tokens.append(form)
        xpos.append(tag)
        spans.append((start, end))
        lemmas.append(lemma)
    return tokens, xpos, spans, lemmas

def parse_dep(parser: Parser, tokens):
    """
    Parse with SuPar (형태소 단위)
    parser.predict는 CPU 기반이기 때문에 GPU tensor가 들어오면 PyTorch가 copy를 떠서 CPU에 얹고 그게 루프 돌때마다 GPU에 남는 캐시로 쌓이는것
    """
    if torch.is_tensor(tokens):
        tokens = tokens.detach().cpu().tolist()
    elif any(torch.is_tensor(t) for t in tokens):
        tokens = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in tokens]
    dataset = parser.predict([tokens], verbose=False)
    s = dataset.sentences[0]
    return (
        tokens,
        getattr(s, "tags", [""] * len(tokens)),
        s.arcs,
        s.rels,
    )  # words, upos, heads(1-based), rels
    
    
# ============================================================================
# 2. 어절 그룹핑
# ============================================================================
def split_eojeols(text: str):
    """원문을 어절 단위로 split"""
    parts = []
    for m in re.finditer(r"\S+", text):
        s, e, frag = m.span()[0], m.span()[1], m.group(0)
        if frag and frag[-1] in ".,?!;:'\"’”)]}":
            if e - 1 > s:
                parts.append((s, e - 1, frag[:-1]))
            parts.append((e - 1, e, frag[-1]))
        else:
            parts.append((s, e, frag))
    return parts

def group_by_eojeol(text, tokens, spans):
    """형태소 분석 결과를 어절 단위로 그룹핑"""
    eoj, groups, i = split_eojeols(text), [], 0
    for es, ee, surf in eoj:
        start_morph_idx = i
        while i < len(spans) and spans[i][0] < ee:
            i += 1
        groups.append({"surface": surf, "morph_start": start_morph_idx, "morph_end": i})
    return groups

def render_eojeol_row(groups, N):
    items = []
    for g in groups:
        s = g["morph_start"]
        e = g["morph_end"] - 1
        items.append(
            f"<div class='ejitem' style='grid-column:{s+1}/{e+2}'>"
            f"  <div class='ejchip'>{g['surface']}</div>"
            f"</div>"
        )
    # 래퍼 제거: 순수 셀만 리턴
    return "".join(items)


def pack_levels(items):
    """겹치지 않는 span들을 표시용 레벨(행)로 그리디 배치"""
    levels = []
    for label, s, e in items:
        for level in levels:
            if not any(max(s, S) <= min(e, E) for _, S, E in level):
                level.append((label, s, e))
                break
        else:
            levels.append([(label, s, e)])
    return levels