#!/usr/bin/env python3
"""Kaggle inference kernel bootstrap (GPU, internet OFF).

Runs on parasrawal117/ber-neural-infer. Mounts BOTH the assets dataset and the
CV kernel's output (kernel_sources), which carries the frozen deploy_neural.json
+ the fine-tuned reranker_final/. Streams the two submission TSVs to
/kaggle/working (matching_results.tsv, candidate_pairs.tsv).

Paths are DISCOVERED by name under /kaggle/input (Kaggle may nest a mount at
/kaggle/input/datasets/<owner>/<slug>), so this is robust to any mount layout.
"""
import glob
import os
import subprocess
import sys

# --- durable, line-buffered logging (survives an OOM/disk kill that skips the
#     nbconvert log) -> captured as a kernel OUTPUT file at neural/run.log -------
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
print("=== [infer] durable logging active -> /kaggle/working/neural/run.log ===",
      flush=True)


def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


_inf = _find("infer_neural.py")
if not _inf:
    raise SystemExit("!!! infer_neural.py not found under /kaggle/input")
NEURAL_SRC = os.path.dirname(_inf)                            # <mount>/.../src/neural
SRC = os.path.dirname(NEURAL_SRC)                            # <mount>/.../src
BASE = os.path.dirname(os.path.dirname(os.path.dirname(SRC)))  # <mount>
_whl = glob.glob("/kaggle/input/**/*.whl", recursive=True)
WHEELS = os.path.dirname(_whl[0]) if _whl else None
_ts1 = _find("test_source1.tsv")
CLEAN = os.path.dirname(os.path.dirname(_ts1)) if _ts1 else f"{BASE}/dataset/clean"
# CV kernel output (frozen operating point + fine-tuned reranker)
_dj = _find("deploy_neural.json")
if not _dj:
    raise SystemExit("!!! deploy_neural.json (CV output) not found under /kaggle/input")
CVN = os.path.dirname(_dj)                                    # <cvout>/neural
RERANKER_FINAL = os.path.join(CVN, "reranker_final")
print(f"BASE={BASE}\nSRC={SRC}\nWHEELS={WHEELS}\nCLEAN={CLEAN}\n"
      f"DEPLOY_JSON={_dj}\nRERANKER_FINAL={RERANKER_FINAL}", flush=True)

if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", WHEELS, "jellyfish", "rapidfuzz"], check=False)
# internet is ON here (HF weight fetch): guarantee deps from PyPI too (the offline
# wheels dir was missing rapidfuzz, which broke the features import at CV time).
# sentencepiece is REQUIRED: reranker_final/ is bge-reranker-v2-m3 (XLM-RoBERTa),
# whose tokenizer can fall back to the SentencePiece model on load.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jellyfish", "rapidfuzz", "sentencepiece"], check=False)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# scratch (NOT /kaggle/working): keep the kernel OUTPUT to just the two TSVs.
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
_fetch("sentence-transformers/LaBSE", "LaBSE",
       ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.h5", "*.ot", "*.msgpack",
        "*.onnx", "onnx/*", "*.bin"])

os.environ["BER_NEURAL_WORK"] = "/kaggle/working/neural"
os.environ["BER_NEURAL_OUT"] = "/kaggle/working"
os.environ["BER_CLEAN"] = CLEAN
os.environ["BER_NEURAL_MODELS"] = WT
os.environ["BER_RERANKER_FINAL"] = RERANKER_FINAL
os.environ["BER_DEPLOY_JSON"] = _dj
# reranker_final/ is the fine-tuned bge-reranker-v2-m3 (560M XLM-R). Cap the eval
# batch to 128 (matches the CV kernel) so full-test scoring never OOMs the 16 GB
# GPU; and name the reranker so nc.RERANKER_NAME is accurate (load is from the
# explicit RERANKER_FINAL dir, so this is for metadata correctness only).
os.environ["BER_RR_EVAL_BATCH"] = "128"
os.environ["BER_RERANKER"] = "bge-reranker-v2-m3"

sys.path.insert(0, NEURAL_SRC)
sys.path.insert(0, SRC)

print("=== [infer] full-test rerank -> submission TSVs ===", flush=True)
sys.argv = ["infer_neural.py"]
import infer_neural  # noqa: E402
infer_neural.main()

print("=== [infer] DONE — matching_results.tsv + candidate_pairs.tsv in /kaggle/working ===",
      flush=True)
