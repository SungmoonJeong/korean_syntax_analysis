# loader.py

import os
import pickle
import warnings


warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
warnings.filterwarnings("ignore", message="apply_permutation is deprecated")

from contextlib import contextmanager

import streamlit as st


# ============================================================================
# Torch 2.6 safe-load patch
# ============================================================================
import torch
import torch.serialization as ts
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from openai import OpenAI
from supar import Parser
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import KIWI_USER_WORDS
from services.gemini_client import GeminiHandler
from services.local_ner_client import LocalNERHandler

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# NNP(고유명사) 판별 백엔드 — "gemini"(기본) 또는 "local"(외부 API 불가 환경, 예: 울산대)
NNP_BACKEND = os.getenv("NNP_BACKEND", "gemini")


@contextmanager
def allow_pickle_load():
    # PyTorch 2.6의 기본 safe-load 제약 완화 (supar pickle 호환)
    orig_load, orig_ts_load = torch.load, ts.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    def patched_ts_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_ts_load(*args, **kwargs)

    torch.load, ts.load = patched_load, patched_ts_load
    try:
        yield
    finally:
        torch.load, ts.load = orig_load, orig_ts_load


# ============================================================================
# Model loader (Jupyter cache)
# ============================================================================
@st.cache_resource(show_spinner=True)
def load_resources(supar_model_path: str = "supar_morph_dp/model.pth"):
    # KLUE-DP 파인튜닝 SuPar 모델 로드
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with allow_pickle_load():
        parser = Parser.load(supar_model_path, device=device)
        print("SuPar device:", next(parser.model.parameters()).device)
    kiwi = Kiwi(integrate_allomorph=False)
    for word, tag, score in KIWI_USER_WORDS:
        kiwi.add_user_word(word, tag, score)
    return parser, kiwi

def load_ce_model(path_model, path_tok):
    tok = AutoTokenizer.from_pretrained(path_tok)
    mdl = AutoModelForSequenceClassification.from_pretrained(path_model)
    return tok, mdl


# ============================================================================
# openai loader
# ============================================================================
@st.cache_resource(show_spinner=True)
def load_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


# ============================================================================
# NNP(고유명사) 판별 핸들러 로더 — NNP_BACKEND에 따라 Gemini/로컬 중 선택
# ============================================================================
@st.cache_resource(show_spinner=True)
def load_nnp_handler():
    if NNP_BACKEND == "local":
        return LocalNERHandler()
    return GeminiHandler(GEMINI_API_KEY)


# ============================================================================
# gloss loader
# ============================================================================
@st.cache_resource(show_spinner=True)
def load_gloss_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=True)
def load_ce_model_cached(ce_model_path, ce_tok_path, ce_device: str):
    ce_tok, ce_model = load_ce_model(ce_model_path, ce_tok_path)
    ce_model.to(torch.device(ce_device)).eval()
    return ce_tok, ce_model