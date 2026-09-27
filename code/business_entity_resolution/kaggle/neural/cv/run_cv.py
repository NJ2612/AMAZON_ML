#!/usr/bin/env python3
"""Kaggle CV/train kernel bootstrap (GPU, internet OFF).

Runs on parasrawal117/ber-neural-cv. The private dataset ber-neural-assets is
mounted read-only at /kaggle/input/ber-neural-assets and mirrors the repo layout
(src/, dataset/clean, dataset/train, dataset/work, models/neural, wheels/), so
common.py resolves ROOT=<mount> and every read path is correct with NO override
except the two writable dirs (NEURAL_WORK / NEURAL_OUT -> /kaggle/working).

Steps: (1) offline-install jellyfish+rapidfuzz from shipped wheels; (2) build the
neural dev set (LaBSE dense ∪ lexical, recall-ceiling audit); (3) leakage-safe
5-fold reranker CV + freeze deploy_neural.json + save reranker_final/ (both land
in /kaggle/working/neural for the inference kernel to pick up via kernel_sources).
"""
import glob
import os
import subprocess
import sys

# --- durable, line-buffered logging -------------------------------------------
# The prior run ERRORED with a 0-byte log: a hard kill (OOM / out-of-disk) skips
# nbconvert, so Kaggle's own log is empty. Mirror stdout+stderr to a real file
# under /kaggle/working (captured as a kernel OUTPUT file regardless of a kill),
# line-buffered so everything up to the kill point survives. faulthandler dumps a
# C-level traceback on fatal signals where possible.
os.makedirs("/kaggle/working/neural", exist_ok=True)
_LOGF = open("/kaggle/working/neural/run.log", "w", encoding="utf-8", buffering=1)
import faulthandler
faulthandler.enable(file=_LOGF)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        # delegate everything else (fileno, encoding, buffer, ...) to the real stream
        return getattr(self.streams[0], name)


sys.stdout = _Tee(sys.__stdout__, _LOGF)
sys.stderr = _Tee(sys.__stderr__, _LOGF)
print("=== [cv] durable logging active -> /kaggle/working/neural/run.log ===",
      flush=True)

# --- discover the real mount layout ------------------------------------------
# Kaggle does not always mount a dataset at /kaggle/input/<slug>; here it lands
# under /kaggle/input/datasets/<owner>/<slug>. Rather than hard-code that, locate
# our modules by name and derive every path from what we actually find.
def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


_mdn = _find("make_devset_neural.py")
if not _mdn:
    raise SystemExit("!!! make_devset_neural.py not found under /kaggle/input")
NEURAL_SRC = os.path.dirname(_mdn)                            # <mount>/.../src/neural
SRC = os.path.dirname(NEURAL_SRC)                            # <mount>/.../src
BASE = os.path.dirname(os.path.dirname(os.path.dirname(SRC)))  # <mount>
_whl = glob.glob("/kaggle/input/**/*.whl", recursive=True)
WHEELS = os.path.dirname(_whl[0]) if _whl else None
_ts1 = _find("test_source1.tsv")
CLEAN = os.path.dirname(os.path.dirname(_ts1)) if _ts1 else f"{BASE}/dataset/clean"
print(f"BASE={BASE}\nSRC={SRC}\nWHEELS={WHEELS}\nCLEAN={CLEAN}", flush=True)

# --- offline deps (torch+transformers are preinstalled on the Kaggle GPU image);
#     jellyfish/rapidfuzz feed blocking + features. No-op if already satisfied. ---
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", WHEELS, "jellyfish", "rapidfuzz"], check=False)
# internet is ON in this kernel (HF weight fetch), so GUARANTEE the deps from PyPI
# regardless of what the shipped wheels dir holds — the offline attempt silently
# missed rapidfuzz once, which only surfaced at the cv_neural (features) import.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jellyfish", "rapidfuzz", "sentencepiece"], check=False)

# --- fetch pretrained weights ON KAGGLE (internet ON) ---------------------------
# The local Windows box could not complete the HF download (repeated WinError
# 10054 connection resets), so instead of shipping weights in the private dataset
# we pull them here from the Hub, where Kaggle's network to HF is reliable. Both
# models are permissively licensed (LaBSE Apache-2.0, bge-reranker-v2-m3
# Apache-2.0) and this is NOT a competition submission, so an internet-ON build
# step is fine; the only network use is fetching these pretrained weights. Retries
# ride out any transient blip. Weights land in a writable dir, BER_NEURAL_MODELS points there.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# scratch (NOT /kaggle/working): base weights must not bloat the kernel OUTPUT —
# only the fine-tuned reranker_final + deploy json need to persist for the infer
# kernel. Keeping output small also makes `kaggle kernels output` pulls fast.
WT = "/kaggle/temp/neural_models"


def _fetch(repo_id, sub, ignore):
    from huggingface_hub import snapshot_download
    dst = os.path.join(WT, sub)
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=repo_id, local_dir=dst,
                              ignore_patterns=ignore, max_workers=4)
            print(f"  [fetch] {repo_id} -> {dst} (try {attempt})", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  [fetch] {repo_id} try {attempt} failed: {e}", flush=True)
    raise RuntimeError(f"could not fetch {repo_id} after 5 tries")


os.makedirs(WT, exist_ok=True)
_IMG = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.h5", "*.ot", "*.msgpack",
        "*.onnx", "onnx/*", "*.bin"]
# Reranker = BAAI/bge-reranker-v2-m3 — a STANDARD-architecture multilingual
# cross-encoder (XLM-RoBERTa, Apache-2.0). It REPLACES Alibaba gte-multilingual-
# reranker-base, whose trust_remote_code "new-impl" forward crashed the prior run
# with a CUDA device-side assert on fold 0's first training step. bge uses the
# plain HF XLMRobertaForSequenceClassification path (no unpadding / RoPE / auto-
# fetched modeling.py), so none of the custom fast-path settings that triggered
# the assert exist here. Keep the *.bin ignore ONLY for LaBSE (ships safetensors);
# for the reranker download whatever weight format the repo has so the load can
# never fail on a missing shard.
_RR_IGNORE = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.h5", "*.ot", "*.msgpack",
              "*.onnx", "onnx/*"]
_fetch("sentence-transformers/LaBSE", "LaBSE", _IMG)
_fetch("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3", _RR_IGNORE)

# --- writable dirs only; everything else resolves off the read-only mount -------
os.environ["BER_NEURAL_WORK"] = "/kaggle/working/neural"
os.environ["BER_NEURAL_OUT"] = "/kaggle/working"
os.environ["BER_CLEAN"] = CLEAN
os.environ["BER_NEURAL_MODELS"] = WT

# --- reranker = bge-reranker-v2-m3 (see fetch note). These env knobs are read by
#     nconfig at import (below) so RERANKER_DIR resolves to the fetched dir and the
#     run is tuned to the bigger 560M XLM-R model so it fits the 12h GPU cap and
#     never OOMs on a 16 GB card:
#       * batch 32 (vs 64) — 560M params + fp32 AdamW states leave less headroom;
#       * eval batch 128 (vs 256);
#       * 1 epoch (vs 2) — a reranker converges in one pass over ~287k pairs/fold,
#         and 5 folds + the final fit must all complete inside 12h.
os.environ["BER_RERANKER"] = "bge-reranker-v2-m3"
os.environ["BER_RR_TRAIN_BATCH"] = "32"
os.environ["BER_RR_EVAL_BATCH"] = "128"
os.environ["BER_RR_EPOCHS"] = "1"

# SRC/NEURAL_SRC were discovered above; common.py's ROOT=abspath(HERE/../../..)
# resolves to <mount> so every read path (CLEAN/WORK/GROUND_TRUTH/DEVSET/OOF) is
# correct. Import the modules normally (NOT runpy __main__) and call main() — this
# also avoids the multiprocessing-pickling trap runpy's __main__ reparenting sets.
sys.path.insert(0, NEURAL_SRC)
sys.path.insert(0, SRC)

# --- src patch override (3-fold) ------------------------------------------------
# Rather than re-upload the 4.8 GB ber-neural-assets dataset just to ship two
# edited files, a TINY second dataset (ber-neural-srcpatch) carries the 3-fold
# cv_neural.py + cv_eval.py. Front its dir on sys.path so `import cv_neural` /
# `import cv_eval` resolve to the patched versions while common/nconfig/
# train_reranker still load from the big assets mount. FATAL if missing — running
# the stale 5-fold code would silently blow the 12h GPU cap again.
_marker = _find("SRCPATCH_3FOLD.txt")
if not _marker:
    raise SystemExit("!!! 3-fold src patch (SRCPATCH_3FOLD.txt) not found under /kaggle/input")
PATCH_DIR = os.path.dirname(_marker)
sys.path.insert(0, PATCH_DIR)
print(f"[srcpatch] 3-fold cv_neural/cv_eval override -> {PATCH_DIR}", flush=True)

print("=== [cv] build neural devset (dense u lexical) ===", flush=True)
# --- checkpoint rolling: reuse a prior committed run's output if mounted --------
# When this kernel is re-pushed with kernel_sources=["parasrawal117/ber-neural-cv"]
# (a resume run after a crash), the previous run's committed /kaggle/working lands
# under /kaggle/input/**. We then (1) reuse its devset_neural_india.parquet to SKIP
# the ~63-min LaBSE embed, and (2) restore any per-fold pce_fold*.npy so build_oof
# reloads completed folds instead of retraining them. On a FRESH push (no
# kernel_sources) nothing is found and the run proceeds normally — but it still
# WRITES checkpoints, so any future resume can roll forward from it.
import shutil  # noqa: E402
_CKPT = "/kaggle/working/neural/ckpt"
os.makedirs(_CKPT, exist_ok=True)
_resume_dev = None
for _h in glob.glob("/kaggle/input/**/devset_neural_india.parquet", recursive=True):
    _resume_dev = _h
    break
_resume_ckpts = glob.glob("/kaggle/input/**/pce_fold*.npy", recursive=True)
for _c in _resume_ckpts:
    try:
        shutil.copy(_c, os.path.join(_CKPT, os.path.basename(_c)))
    except Exception as e:  # noqa: BLE001
        print(f"  [resume] could not stage {_c}: {e}", flush=True)
if _resume_ckpts:
    print(f"[resume] restored fold checkpoint(s): "
          f"{sorted(os.path.basename(c) for c in _resume_ckpts)}", flush=True)

if _resume_dev:
    shutil.copy(_resume_dev, "/kaggle/working/neural/devset_neural_india.parquet")
    print(f"[resume] reusing prior devset {_resume_dev} -> SKIP LaBSE embed",
          flush=True)
else:
    sys.argv = ["make_devset_neural.py"]
    import make_devset_neural  # noqa: E402
    make_devset_neural.main()

print("=== [cv] leakage-safe 3-fold reranker CV + save final ===", flush=True)
# Enable gradient checkpointing on the reranker so the 560M XLM-R fits a 16 GB
# GPU at batch 32 (trades ~25% compute for a large activation-memory cut — the
# guard against an OOM hard-kill). Patch the shared loader so BOTH the per-fold
# CV models and the final saved model get it; use_reentrant=False when the
# installed transformers supports it, else the legacy call.
import train_reranker as _tr  # noqa: E402
_orig_load_rr = _tr.load_reranker


def _load_rr_ckpt(model_dir=None, device=None):
    model, tok = _orig_load_rr(model_dir=model_dir, device=device)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        print("  [rr] gradient checkpointing ON", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [rr] grad-checkpointing unavailable: {e}", flush=True)
    return model, tok


_tr.load_reranker = _load_rr_ckpt

# 3-fold CV (user-chosen 2026-09-27): 5 folds + the all-data final fit cannot
# finish inside the 12h GPU cap at the observed ~1.2 it/s (each fold ~2.08h), so
# run 3 leakage-safe folds — each model STILL trains on 80% of dev — plus the
# final all-data fit. Honest 3-fold macro-F0.5 in one ~9h window. Checkpoint
# rolling still writes pce_fold{0,1,2}.npy, so a crash resumes forward.
sys.argv = ["cv_neural.py", "--save-final", "--folds", "0,1,2"]
import cv_neural  # noqa: E402
cv_neural.main()

print("=== [cv] DONE — deploy_neural.json + reranker_final in /kaggle/working/neural ===",
      flush=True)
