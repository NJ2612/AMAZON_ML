#!/usr/bin/env python3
"""Central config for the kaggle_2 neural pipeline.

Every path / knob is env-overridable so the SAME code runs unchanged locally
(repo layout, CPU smoke tests) and on Kaggle GPU (read-only /kaggle/input for
weights + shipped assets, writable /kaggle/working for outputs). Defaults
reproduce the repo layout.
"""
from __future__ import annotations

import os
import sys

# --- make `import common` work from src/neural/ (both local & Kaggle) ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import common as c  # noqa: E402  (paths, IO, folds, scorer — reused verbatim)

# --- device -------------------------------------------------------------------
def get_device() -> str:
    """cuda if a GPU is visible else cpu (lets modules import on a CPU box)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --- model weights (shipped in the private dataset; internet OFF) -------------
# Local default: models/neural/<name>.  Kaggle: set BER_NEURAL_MODELS to the
# /kaggle/input/<dataset>/models/neural mount.
NEURAL_MODELS = (os.environ.get("BER_NEURAL_MODELS")
                 or os.path.join(c.MODELS, "neural"))

# Bi-encoder retriever (Apache-2.0, 471M, 768-d, 109-language) — embeds the RAW
# native-script business_name (NOT the romanized name_core).
BIENCODER_NAME = os.environ.get("BER_BIENCODER", "LaBSE")
BIENCODER_DIR = os.path.join(NEURAL_MODELS, BIENCODER_NAME)

# Cross-encoder reranker (Apache-2.0, ~306M) — reads S1 & candidate jointly.
RERANKER_NAME = os.environ.get("BER_RERANKER", "gte-multilingual-reranker-base")
RERANKER_DIR = os.path.join(NEURAL_MODELS, RERANKER_NAME)

# --- retrieval / rerank knobs -------------------------------------------------
EMB_BATCH = int(os.environ.get("BER_EMB_BATCH", "256"))     # bi-encoder encode batch
EMB_DIM = 768
EMB_MAX_TOK = int(os.environ.get("BER_EMB_MAXTOK", "64"))   # names are short
ANN_TOPM = int(os.environ.get("BER_ANN_TOPM", "50"))        # dense candidates / S1
# Lexical candidates per S1. MUST be identical in the CV devset build
# (make_devset_neural) and at inference (infer_neural) — else the reported
# macro-F0.5 describes a candidate pool the submission never builds. Kept small
# so the cross-encoder rerank cost fits inside the 12h GPU cap.
LEX_TOPK = int(os.environ.get("BER_LEX_TOPK", "25"))
FAISS_NLIST = int(os.environ.get("BER_FAISS_NLIST", "4096"))  # IVF cells (large corpus)
FAISS_NPROBE = int(os.environ.get("BER_FAISS_NPROBE", "64"))

RERANK_MAXTOK = int(os.environ.get("BER_RERANK_MAXTOK", "96"))
RERANK_TRAIN_BATCH = int(os.environ.get("BER_RR_TRAIN_BATCH", "64"))
RERANK_EVAL_BATCH = int(os.environ.get("BER_RR_EVAL_BATCH", "256"))
RERANK_EPOCHS = float(os.environ.get("BER_RR_EPOCHS", "2"))
RERANK_LR = float(os.environ.get("BER_RR_LR", "2e-5"))
RERANK_NEG_RATIO = float(os.environ.get("BER_RR_NEG", "5"))  # train-side downsample
RERANK_WARMUP = float(os.environ.get("BER_RR_WARMUP", "0.06"))

# --- CV / leakage (must match the GBDT side exactly) --------------------------
K_FOLDS = 5
SEED = 42

# --- reused assets (shipped in the dataset) -----------------------------------
DEVSET_PARQUET = os.path.join(c.WORK, "devset_india_n20000.parquet")
OOF_PARQUET = os.path.join(c.WORK, "devset_india_n20000_oof.parquet")
INDIA_INDEX = os.path.join(c.WORK, "idx_india_cap3000.pkl")

# --- outputs ------------------------------------------------------------------
# kaggle_2 writes here (distinct from the GBDT output/ tree).
NEURAL_OUT = os.environ.get("BER_NEURAL_OUT") or os.path.join(
    c.ROOT, "output", "kaggle_results_2")
NEURAL_WORK = os.environ.get("BER_NEURAL_WORK") or os.path.join(c.WORK, "neural")
for _d in (NEURAL_OUT, NEURAL_WORK):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass

# text used by the bi-encoder (raw name; native script preserved) and the
# cross-encoder (name + address, joined). Kept here so every stage agrees.
def name_text(name: str) -> str:
    """Bi-encoder input: raw business_name, whitespace-collapsed."""
    return " ".join((name or "").split())


def pair_text(name: str, addr: str) -> str:
    """Cross-encoder per-record text: 'name [SEP-ish] address' (raw)."""
    name = " ".join((name or "").split())
    addr = " ".join((addr or "").split())
    return f"{name} | {addr}" if addr else name
