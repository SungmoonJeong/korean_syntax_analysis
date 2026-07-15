# openai_client.py
import json
import re


def _safe_json_extract(s: str):
    if not s:
        return None
    try:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(s)
    except Exception:
        m_idx = re.search(r"\"index\"\s*:\s*(\d+)", s)
        if m_idx:
            return {"index": int(m_idx.group(1))}
        m_gloss = re.search(r"\"gloss\"\s*:\s*\"([A-Za-z0-9\-\s]+)\"", s)
        if m_gloss:
            return {"gloss": m_gloss.group(1).strip()}
    return None

class OpenAIHandler:
    def __init__(self, client):
        """생성될 때 클라이언트를 받아서 내부에 저장"""
        self.client = client
        
    def openai_call(self, prompt: str, max_tokens: int = 64) -> str:
        """기존의 llm_fn 역할"""
        if not self.client:
            return ""
        resp = self.client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=0.0,
            max_output_tokens=max_tokens,
        )
        return (resp.output_text or "").strip()

    def translate_to_english(self, sentence: str) -> str:
        """한국어 문장을 영어로 번역"""
        if not self.client:
            return ""
        prompt = (
            "Translate the following Korean sentence into natural English. "
            "Output ONLY the English translation, nothing else.\n\n"
            f"Korean: {sentence}"
        )
        try:
            return self.openai_call(prompt, max_tokens=256)
        except Exception:
            return ""


def openai_pick_from_candidates(
    form, lemma, pos_kor, pos_eng, candidates, sentence, _llm_fn
) -> str:
    gloss_list = "\n".join(f"{i+1}. {g}" for i, g in enumerate(candidates))
    sent_marked = sentence.replace(form, f"[TGT]{form}[/TGT]", 1)
    prompt = (
        "You are a bilingual lexicographer conforming to strict output rules.\n"
        "\n"
        "[TASK]\n"
        "Pick the MOST LITERAL English gloss (dictionary lemma sense) for the TARGET token.\n"
        "Avoid idioms, paraphrases, inflected forms, or multiword expressions unless the base sense itself is multiword.\n"
        "\n"
        "[INPUT]\n"
        f"Sentence: {sent_marked}\n"
        f"Surface: {form}\n"
        f"Lemma: {lemma}\n"
        f"POS: {pos_eng} ({pos_kor})\n"
        "Candidates:\n"
        f"{gloss_list}\n"
        "\n"
        "[DECISION RULES]\n"
        "1) Prioritize the dictionary base meaning of the lemma.\n"
        "2) Select the candidate that best fits the SENTENCE context literally (not figuratively).\n"
        "3) If multiple are equally literal, choose the one with:\n"
        "   - (a) part-of-speech compatibility, then\n"
        "   - (b) shortest and most canonical lemma form, then\n"
        "   - (c) earliest index.\n"
        "4) Do NOT invent a new gloss. Choose only from the list.\n"
        "\n"
        "[OUTPUT FORMAT]\n"
        "Return STRICT JSON only, no prose, matching this schema and example.\n"
        'Schema: {"index": <integer 1..N>}\n'
        'Example: {"index": 2}\n'
    )

    out = _llm_fn(prompt)
    obj = _safe_json_extract(out)
    if isinstance(obj, dict) and "index" in obj:
        idx = int(obj["index"]) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def openai_generate_gloss(form, pos_kor, sentence, _llm_fn) -> str:
    pos_hint = (
        "Verb → base form (e.g., 'go'); "
        "Noun → singular base form (e.g., 'book'); "
        "Adjective → positive base form (e.g., 'good')."
    )

    prompt = (
        "You are a bilingual lexicographer. Generate a SINGLE-WORD English gloss.\n"
        "\n"
        "[TASK]\n"
        "Given the sentence and the target word with POS (Korean), output the most suitable, literal English gloss.\n"
        "Do NOT output explanations, examples, or multiple words unless the canonical lemma is hyphenated (e.g., 'so-so').\n"
        "\n"
        "[INPUT]\n"
        f'Sentence: "{sentence}"\n'
        f'Word: "{form}" (POS: {pos_kor})\n'
        "\n"
        "[FORM CONSTRAINTS]\n"
        f"{pos_hint}\n"
        "Lowercase preferred. No punctuation except hyphen when part of lemma.\n"
        "Avoid idioms, phrasal verbs, or paraphrases.\n"
        "\n"
        "[OUTPUT FORMAT]\n"
        "Return STRICT JSON only, no prose.\n"
        'Schema: {"gloss": "<single word>"}\n'
        "Examples:\n"
        '{"gloss": "go"}\n'
        '{ "gloss": "book" }\n'
        '{ "gloss": "good" }\n'
    )

    out = _llm_fn(prompt)
    obj = _safe_json_extract(out)
    if isinstance(obj, dict) and isinstance(obj.get("gloss"), str):
        gloss = re.sub(r"[^A-Za-z0-9\- ]", "", obj["gloss"]).strip().lower()
        return (gloss.split()[0] if " " in gloss else gloss) or "unknown"
    fallback = re.sub(r"[^A-Za-z0-9\- ]", "", (out or "")).strip().lower()
    return fallback.split()[0] if fallback else "unknown"