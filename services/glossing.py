# glossing.py

import re
from typing import Dict, List, Optional, Tuple

import config
import numpy as np
import torch
from config import POS_ENG_MAP, POS_MAP, device_ce
from services.openai_client import (OpenAIHandler, openai_generate_gloss,
                                    openai_pick_from_candidates)



# ============================================================================
# 2. 우선순위 glossig strategy
# ============================================================================
# Gemini NNP 병합 토큰 내부 재분해 기준 — 예: "제주지부"를 통째로 사전/LLM에
# 물으면 "branch"만 남고 "제주(Jeju)" 의미가 사라진다. Kiwi 기본분석으로
# 재분해가 정확히 2조각(제주+지부)이고 뒷조각이 사전에 독립 명사로 있을 때만
# 각각 글로스를 구해 합성한다 — 토큰/품사 자체는 그대로 두므로 파싱 결과에는
# 영향이 없다(klue-dev 검증에서 3조각 이상 분해나 뒷조각 미존재 시 원래
# 통짜 처리와 회귀가 생기는 걸 확인했기 때문에 정확히 2조각인 경우로 제한).
NNP_MIN_LEN_FOR_DECOMPOSE = 3
NNP_TAIL_MIN_LEN = 2


def decompose_compound_nnp(form: str, pos: str, kiwi) -> Optional[Tuple[str, str, str]]:
    """병합된 고유명사(NNP) 토큰을 Kiwi 기본분석으로 재분해.

    (head_form, head_pos, tail_form) 반환, 조건 미충족 시 None
    (뒷조각이 사전에 독립 명사로 있는지는 호출부에서 확인한다).
    """
    if pos != "NNP" or len(form) < NNP_MIN_LEN_FOR_DECOMPOSE or kiwi is None:
        return None
    try:
        default = kiwi.analyze(form, top_n=1)[0][0]
    except Exception:
        return None
    pieces = [(m.form, m.tag.split("-")[0]) for m in default]
    if len(pieces) != 2:
        return None
    (head_form, head_pos), (tail_form, _tail_pos) = pieces
    if len(tail_form) < NNP_TAIL_MIN_LEN or not head_form:
        return None
    return head_form, head_pos, tail_form


def compound_rule_with_context(tokens, pos_tags, i):
    """(1) 복합 규칙 — (gloss, pos_eng_override, n, meta) 반환; 미매칭 시 None"""
    # 1-1. 접사 결합 규칙 (2-gram만 지원: 어근 + 접사)
    if i + 1 < len(tokens):
        key = (tokens[i], pos_tags[i], tokens[i + 1], pos_tags[i + 1])
        if key in config.AFFIX_COMPOUND:
            gloss, pos_eng_override = config.AFFIX_COMPOUND[key]
            return gloss, pos_eng_override, 2, "affix_compound"
        # 와일드카드 matching: (ANY, POS1, form2, POS2)
        key_wildcard = (config.ANY, pos_tags[i], tokens[i + 1], pos_tags[i + 1])
        if key_wildcard in config.AFFIX_COMPOUND:
            gloss, pos_eng_override = config.AFFIX_COMPOUND[key_wildcard]
            return gloss, pos_eng_override, 2, "affix_compound"

    # 1-2. 조사/어미 결합 규칙 (기존)
    for key in sorted(config.COMPOUND_JKB_TO.keys(), key=lambda k: len(k), reverse=True):
        n = len(key) // 2
        if i + n > len(tokens):
            continue
        ok = True
        for k in range(n):
            f_exp, p_exp = key[2 * k], key[2 * k + 1]
            f_act, p_act = tokens[i + k], pos_tags[i + k]
            if f_exp != config.ANY and f_exp != f_act:
                ok = False
                break
            if p_exp != config.ANY and p_exp != p_act:
                ok = False
                break
        if ok:
            val = config.COMPOUND_JKB_TO[key]
            if isinstance(val, tuple):
                gloss, pos_eng_override = val
            else:
                gloss, pos_eng_override = val, None
            return gloss, pos_eng_override, n, "compound_ngram"
    return None

def rule_gloss(form: str, pos: str) -> Optional[Tuple[str, str]]:
    """(2) 단일 규칙"""
    if pos.split("+")[0] in config.PUNC_TAGS or re.fullmatch(r"[^\w\s]+", form):
        return form, "punct_self"

    g = config.RULE_FORM_POS.get((form, pos))
    if g is not None:
        return (g if isinstance(g, str) else g[0]), "rule_form_pos"
    rules = config.RULE_REGEX_POS.get(pos)
    if rules:
        for pat, gloss in rules:
            if pat.fullmatch(form) or pat.search(form):
                return gloss, "rule_regex_pos"
    return None


def select_best_gloss_ce(
    sentence: str,
    tokens: List[str],
    idx: int,
    lemma: str,
    pos_kor: str,
    candidates: List[str],
    tau: float,
    margin: float,
    ce_tok=None,
    ce_model=None,
):
    """(4) Cross-Encoder 선별 (복수 후보)"""
    if not candidates:
        return None, {"rule": "no_candidates"}
    scores = score_candidates_ce(
        sentence, tokens, idx, lemma, pos_kor, candidates, ce_tok, ce_model
    )
    soft = torch.softmax(torch.tensor(scores), dim=0).numpy().tolist()
    top1_idx = int(np.argmax(soft))
    top1_p = float(soft[top1_idx])
    meta_common = {
        "candidates": candidates,
        "probs": [float(p) for p in soft],
        "top1_idx": top1_idx,
        "tau": tau,
        "margin": margin,
    }
    if tau > 0.0 and top1_p < tau:
        m = {"rule": "reject_tau", "top1": top1_p}
        m.update(meta_common)
        return None, m
    if margin > 0.0 and len(candidates) > 1:
        srt = sorted(soft, reverse=True)
        if (srt[0] - srt[1]) < margin:
            m = {"rule": "reject_margin", "top1": float(srt[0]), "top2": float(srt[1])}
            m.update(meta_common)
            return None, m
    m = {"rule": "ce", "top1": top1_p}
    m.update(meta_common)
    return candidates[top1_idx], m

def build_pair_for_gloss(
    sentence: str,
    tokens: List[str],
    idx: int,
    lemma: str,
    pos_kor: str,
    candidate: str,
    win=40,
):
    """(5) 표제어(lemma) 검색"""
    l = max(0, idx - win)
    r = min(len(tokens), idx + win + 1)
    marked = tokens[l:r].copy()
    marked[idx - l] = f"[TGT]{tokens[idx]}[/TGT]"
    text_a = " ".join(marked)
    text_b = f"lemma={lemma}; pos={pos_kor}; gloss={candidate}"
    return text_a, text_b

@torch.no_grad()
def score_candidates_ce(
    sentence: str,
    tokens: List[str],
    idx: int,
    lemma: str,
    pos_kor: str,
    candidates: List[str],
    ce_tok,
    ce_model,
) -> List[float]:
    """(6) LLM Fallback"""
    if ce_model is None or not candidates:
        return [1.0 / len(candidates)] * len(candidates) if candidates else []
    pairs = [
        build_pair_for_gloss(sentence, tokens, idx, lemma, pos_kor, c)
        for c in candidates
    ]
    enc = ce_tok(
        [a for a, _ in pairs],
        [b for _, b in pairs],
        padding=True,
        truncation=True,
        max_length=320,
        return_tensors="pt",
    ).to(device_ce)
    logits = ce_model(**enc).logits.squeeze(-1)
    return logits.detach().cpu().tolist()


# ============================================================================
# 3. Glossing
# ============================================================================
def gloss_sequence_from_tokens(
    sentence: str,
    tokens: List[str],
    lemmas: List[str],
    xpos: List[str],
    gloss_dict: Dict[Tuple[str, str], List[str]],
    tau: float,
    margin: float,
    ce_tok,
    ce_model,
    use_llm: bool,
    ai_handler: OpenAIHandler,
    kiwi=None,
) -> List[dict]:
    # Kiwi xpos 직접 사용 — BERT POS 태거 제거 (정확도 비교 결과 Kiwi 88.7% > BERT 87.3%)
    pos_tags = xpos

    outputs = []
    i = 0
    while i < len(tokens):
        form, lemma, pos = tokens[i], lemmas[i], pos_tags[i]
        pos_kor = POS_MAP.get(pos, "NA")
        pos_eng = config.FORM_POS_ENG_OVERRIDE.get((form, pos)) or POS_ENG_MAP.get(pos, "NA")

        # compound
        hit = compound_rule_with_context(tokens, pos_tags, i)
        if hit is not None:
            gloss, pos_eng_override, span, meta_rule = hit
            head_pos_eng = pos_eng_override if pos_eng_override is not None else pos_eng

            # 접사 결합: 글로스가 None이면 각 토큰의 글로스를 구해서 합침
            if meta_rule == "affix_compound" and gloss is None:
                # 어근 토큰의 글로스 구하기
                root_form_pos_key = (tokens[i], pos_kor)
                root_gloss = None

                # 규칙 확인
                root_rule = rule_gloss(tokens[i], pos_tags[i])
                if root_rule:
                    root_gloss = root_rule[0]

                # 사전 확인
                if not root_gloss:
                    root_cand = gloss_dict.get(root_form_pos_key, [])
                    if len(root_cand) == 1:
                        root_gloss = root_cand[0]
                    elif len(root_cand) > 1:
                        root_gloss, _ = select_best_gloss_ce(
                            sentence, tokens, i, lemmas[i], pos_kor, root_cand, tau, margin, ce_tok, ce_model
                        )

                # fallback
                if not root_gloss:
                    root_gloss = tokens[i]

                gloss = root_gloss  # 어근의 글로스가 합쳐진 글로스

            outputs.append(
                {
                    "token": form,
                    "lemma": lemma,
                    "pos": pos,
                    "pos_kor": pos_kor,
                    "pos_eng": head_pos_eng,
                    "gloss": gloss if gloss else form,
                    "meta": {"rule": meta_rule, "span": span},
                }
            )
            for k in range(1, span):
                j = i + k
                outputs.append(
                    {
                        "token": tokens[j],
                        "lemma": lemmas[j],
                        "pos": pos_tags[j],
                        "pos_kor": POS_MAP.get(pos_tags[j], "NA"),
                        "pos_eng": head_pos_eng,
                        "gloss": gloss if gloss else form,
                        "meta": {"rule": "compound_tail", "head_index": i},
                    }
                )
            i += span
            continue

        # NNP 내부 재분해 (병합된 고유명사의 의미 손실 방지 — 위 설명 참고)
        decomposed = decompose_compound_nnp(form, pos, kiwi)
        if decomposed is not None:
            head_form, head_pos, tail_form = decomposed
            head_pos_kor = POS_MAP.get(head_pos, "NA")
            tail_cand = gloss_dict.get((tail_form, "명사"), [])
            if tail_cand:  # 뒷조각이 독립 명사로 사전에 있을 때만 재분해를 신뢰
                if len(tail_cand) == 1:
                    tail_gloss = tail_cand[0]
                else:
                    tail_gloss, _ = select_best_gloss_ce(
                        sentence, tokens, i, tail_form, "명사", tail_cand, tau, margin, ce_tok, ce_model
                    )
                    tail_gloss = tail_gloss or tail_cand[0]

                head_gloss = None
                head_rule = rule_gloss(head_form, head_pos)
                if head_rule:
                    head_gloss = head_rule[0]
                if not head_gloss:
                    head_cand = gloss_dict.get((head_form, head_pos_kor), [])
                    if len(head_cand) == 1:
                        head_gloss = head_cand[0]
                    elif len(head_cand) > 1:
                        head_gloss, _ = select_best_gloss_ce(
                            sentence, tokens, i, head_form, head_pos_kor, head_cand, tau, margin, ce_tok, ce_model
                        )
                if not head_gloss and use_llm and ai_handler:
                    head_gloss = openai_generate_gloss(head_form, head_pos_kor, sentence, _llm_fn=ai_handler.openai_call)
                if not head_gloss:
                    head_gloss = head_form

                outputs.append(
                    {
                        "token": form,
                        "lemma": lemma,
                        "pos": pos,
                        "pos_kor": pos_kor,
                        "pos_eng": pos_eng,
                        "gloss": f"{head_gloss} {tail_gloss}",
                        "meta": {"rule": "nnp_compound_decompose", "head": head_form, "tail": tail_form},
                    }
                )
                i += 1
                continue

        # rule
        rule = rule_gloss(form, pos)
        if rule is not None:
            gloss, meta_rule = rule
            outputs.append(
                {
                    "token": form,
                    "lemma": lemma,
                    "pos": pos,
                    "pos_kor": pos_kor,
                    "pos_eng": pos_eng,
                    "gloss": gloss,
                    "meta": {"rule": meta_rule},
                }
            )
            i += 1
            continue

        # lexicon
        gloss, meta = None, {"rule": "none"}
        cand = gloss_dict.get((form, pos_kor), [])
        if len(cand) == 1:
            gloss, meta = cand[0], {"rule": "lexicon_form"}
        elif len(cand) > 1:
            gloss, meta = select_best_gloss_ce(
                sentence, tokens, i, lemma, pos_kor, cand, tau, margin, ce_tok, ce_model
            )

        if gloss is None:
            cand2 = gloss_dict.get((lemma, pos_kor), [])
            if len(cand2) == 1:
                gloss, meta = cand2[0], {"rule": "lexicon_lemma"}
            elif len(cand2) > 1:
                gloss, meta = select_best_gloss_ce(
                    sentence,
                    tokens,
                    i,
                    lemma,
                    pos_kor,
                    cand2,
                    tau,
                    margin,
                    ce_tok,
                    ce_model,
                )

        # LLM fallback
        if gloss is None and use_llm and ai_handler:
            pool = cand if cand else (cand2 if "cand2" in locals() else [])
            if pool:
                chosen = openai_pick_from_candidates(
                    form, lemma, pos_kor, pos_eng, pool, sentence, _llm_fn=ai_handler.openai_call
                )
                gloss = chosen
                try:
                    idx = pool.index(chosen)
                except ValueError:
                    idx = None
                meta = {"rule": "llm_select", "candidates": pool, "top1_idx": idx}
            else:
                gloss = openai_generate_gloss(form, pos_kor, sentence, _llm_fn=ai_handler.openai_call)
                meta = {"rule": "llm_generate"}

        outputs.append(
            {
                "token": form,
                "lemma": lemma,
                "pos": pos,
                "pos_kor": pos_kor,
                "pos_eng": pos_eng,
                "gloss": gloss,
                "meta": meta,
            }
        )
        i += 1
    return outputs
