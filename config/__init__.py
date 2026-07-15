import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

from .display import (
    CLAUSES,
    PAGE_LAYOUT,
    PAGE_TITLE,
    PHRASES,
    PUNC_TAGS,
    SPAN_DISPLAY,
    SpanLabels,
)
from .glossing_rules import (
    AFFIX_COMPOUND,
    ANY,
    COMPOUND_JKB_TO,
    RULE_FORM_POS,
    RULE_REGEX_POS,
)
from .pos_mapping import (
    FORM_POS_ENG_OVERRIDE,
    KIWI_USER_WORDS,
    POS_ENG_MAP,
    POS_MAP,
)

__all__ = [
    # Device & Seed
    "set_seed",
    "device_ce",
    # App Config (display.py)
    "PAGE_TITLE",
    "PAGE_LAYOUT",
    "SPAN_DISPLAY",
    "PUNC_TAGS",
    "PHRASES",
    "CLAUSES",
    "SpanLabels",
    # POS Mapping (pos_mapping.py)
    "KIWI_USER_WORDS",
    "POS_MAP",
    "POS_ENG_MAP",
    "FORM_POS_ENG_OVERRIDE",
    # Glossing Rules (glossing_rules.py)
    "RULE_FORM_POS",
    "RULE_REGEX_POS",
    "ANY",
    "COMPOUND_JKB_TO",
    "AFFIX_COMPOUND",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device_ce = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
