#!/usr/bin/env python3
"""Fine-tune + score the cross-encoder reranker (one fold at a time).

AutoModelForSequenceClassification with num_labels=1 (a relevance logit), BCE
loss, AdamW + linear warmup, fp16 autocast. Works for gte-multilingual-reranker
-base (trust_remote_code), bge-reranker-v2-m3, or mmarco-mMiniLM — one code path.
Negatives are downsampled on the TRAIN side only; the val fold is always scored
in full (leakage-safe, matches the GBDT CV protocol).
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd

import nconfig as nc
import rerank_dataset as rd


def load_reranker(model_dir: str | None = None, device: str | None = None):
    """Load (model, tokenizer) with a single-logit classification head."""
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = model_dir or nc.RERANKER_DIR
    device = device or nc.get_device()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=1, trust_remote_code=True)
    model.to(device)
    return model, tok


def downsample_train(df: pd.DataFrame, neg_ratio: float, seed: int) -> np.ndarray:
    """Row indices: all positives + neg_ratio x that many negatives."""
    rng = np.random.default_rng(seed)
    y = df["label"].to_numpy()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k = min(len(neg), int(len(pos) * neg_ratio))
    neg = rng.choice(neg, size=k, replace=False)
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


def _logits(model, enc, device):
    import torch
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    lg = out.logits
    return lg.squeeze(-1) if lg.dim() > 1 else lg


def train_model(model, tok, train_df: pd.DataFrame, device: str | None = None,
                epochs: float | None = None, lr: float | None = None,
                batch: int | None = None, neg_ratio: float | None = None,
                seed: int = 0, log_every: int = 200):
    """Fine-tune `model` in place on `train_df` (downsampled). Returns model."""
    import torch
    from torch.optim import AdamW
    device = device or nc.get_device()
    epochs = nc.RERANK_EPOCHS if epochs is None else epochs
    lr = lr or nc.RERANK_LR
    batch = batch or nc.RERANK_TRAIN_BATCH
    neg_ratio = nc.RERANK_NEG_RATIO if neg_ratio is None else neg_ratio

    idx = downsample_train(train_df, neg_ratio, seed)
    ds = rd.PairDataset(train_df["s1_name"].to_numpy()[idx],
                        train_df["s1_addr"].to_numpy()[idx],
                        train_df["c_name"].to_numpy()[idx],
                        train_df["c_addr"].to_numpy()[idx],
                        train_df["label"].to_numpy()[idx])
    n = len(ds)
    steps_per_epoch = (n + batch - 1) // batch
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup = int(total_steps * nc.RERANK_WARMUP)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    collate = rd.make_collate(tok)

    print(f"    [train] {n:,} rows ({int(train_df['label'].to_numpy()[idx].sum()):,} pos) "
          f"x{epochs}ep = {total_steps:,} steps, bs={batch}", flush=True)
    model.train()
    step = 0
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()
    done = False
    for ep in range(int(np.ceil(epochs))):
        order = rng.permutation(n)
        for mb in rd.iter_minibatches(ds, order, batch):
            enc, y = collate([(q, p, yy) for q, p, yy in mb])
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logit = _logits(model, enc, device)
                loss = lossf(logit.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % log_every == 0:
                print(f"      step {step}/{total_steps} loss={loss.item():.4f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)
            if step >= total_steps:
                done = True
                break
        if done:
            break
    return model


def score_df(model, tok, df: pd.DataFrame, device: str | None = None,
             batch: int | None = None) -> np.ndarray:
    """Return sigmoid(logit) match prob for every row of df (in order)."""
    import torch
    device = device or nc.get_device()
    batch = batch or nc.RERANK_EVAL_BATCH
    ds = rd.PairDataset(df["s1_name"].to_numpy(), df["s1_addr"].to_numpy(),
                        df["c_name"].to_numpy(), df["c_addr"].to_numpy())
    collate = rd.make_collate(tok)
    model.eval()
    out = np.empty(len(ds), dtype=np.float32)
    order = np.arange(len(ds))
    pos = 0
    use_amp = device == "cuda"
    with torch.no_grad():
        for mb in rd.iter_minibatches(ds, order, batch):
            enc, _ = collate([(q, p, None) for q, p, _ in mb])
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logit = _logits(model, enc, device)
            p = torch.sigmoid(logit.float()).cpu().numpy()
            out[pos:pos + len(p)] = p
            pos += len(p)
    return out

