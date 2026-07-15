class SpanLabels:
    SENT      = "Sentence"
    MAINC     = "MainC"
    SUBC      = "SubC"
    QUOTEC_DIR = "QuoteC_Dir"
    QUOTEC_IND = "QuoteC_Ind"
    EMC_ADJ   = "EmC_Adj"
    EMC_N     = "EmC_N"
    EMC_ADV   = "EmC_Adv"
    VP   = "VP"
    ADJP = "AdjP"
    ADVP = "AdvP"
    NP   = "NP"
    NPS  = "NPS"
    SP   = "SP"
    OP   = "OP"
    CP   = "CP"
    TP   = "TP"


PAGE_TITLE = "한국어 문장 구조 분석 엔진"
PAGE_LAYOUT = "wide"

SPAN_DISPLAY: dict = {
    "Sentence":   "Sentence",
    "MainC":      "Main Clause",
    "SubC":       "Subordinate Clause",
    "QuoteC_Dir": "Direct Quotation",
    "QuoteC_Ind": "Indirect Quotation",
    "EmC_Adj":    "Embedded Clause (Adjectival)",
    "EmC_N":      "Embedded Clause (Noun)",
    "EmC_Adv":    "Embedded Clause (Adverbial)",
    "VP":   "Verb Phrase",
    "AdjP": "Adjectival Phrase",
    "AdvP": "Adverbial Phrase",
    "NP":   "Noun Phrase",
    "NPS":  "Noun Phrase Sequence",
    "SP":   "Subject Phrase",
    "OP":   "Object Phrase",
    "CP":   "Complement Phrase",
    "TP":   "Topic Phrase",
}

PUNC_TAGS = {"SF", "SP", "SS", "SE", "SO", "SW", "SY"}

PHRASES = {
    SpanLabels.TP,
    SpanLabels.SP,
    SpanLabels.OP,
    SpanLabels.VP,
    SpanLabels.ADVP,
    SpanLabels.ADJP,
    SpanLabels.NP,
    SpanLabels.NPS,
    SpanLabels.CP,
}

CLAUSES = {
    SpanLabels.MAINC,
    SpanLabels.SUBC,
    SpanLabels.QUOTEC_DIR,
    SpanLabels.QUOTEC_IND,
    SpanLabels.EMC_ADJ,
    SpanLabels.EMC_N,
    SpanLabels.EMC_ADV,
}