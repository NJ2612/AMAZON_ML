#!/usr/bin/env python3
"""Build kaggle/matching.ipynb — the self-contained, offline, no-submit inference
notebook that runs on the private ber-matching-inputs dataset. Regenerate with:
    python build_notebook.py
"""
import json
import os

CODE = r'''# BER full-test inference on Kaggle — OFFLINE, NO competition submit.
# Uses the private ber-matching-inputs dataset (code + cleaned test + idf + wheels
# + dev matrix/OOF). Retrains the deploy LGB+XGB on Kaggle (fast, deterministic —
# sidesteps any cross-version pickle issue), then scores every test S1 and writes
# matching_results.tsv + candidate_pairs.tsv to /kaggle/working (the kernel output).
import os, sys, glob, shutil, subprocess, time

hits = glob.glob('/kaggle/input/**/src/infer.py', recursive=True)
assert hits, 'attach the ber-matching-inputs dataset (src/infer.py not found)'
DS = os.path.dirname(os.path.dirname(hits[0]))          # /kaggle/input/<slug>
print('input dataset dir:', DS, flush=True)

# 1) offline deps (internet is OFF): install the shipped linux wheels
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                '--find-links', DS + '/wheels', 'jellyfish', 'rapidfuzz'], check=True)

# 2) writable dirs; the idf caches must live under BER_WORK
os.makedirs('/kaggle/working/work', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)
for f in ('tokdf_india.pkl', 'numidf_india.pkl'):
    shutil.copy(DS + '/work/' + f, '/kaggle/working/work/' + f)

env = dict(os.environ)
env['BER_CLEAN']  = DS + '/clean'         # -> {DS}/clean/test/test_source*.tsv
env['BER_WORK']   = '/kaggle/working/work'
env['BER_MODELS'] = '/kaggle/working/models'
env['BER_OUTPUT'] = '/kaggle/working'     # the two TSVs land here (downloadable)
SRC = DS + '/src'

# 3) retrain deploy models on Kaggle from the uploaded dev matrix + OOF
#    (~2 min, same seed => same models; t* read from the same leakage-safe OOF)
subprocess.run([sys.executable, SRC + '/train.py',
                DS + '/work/devset_india_n20000_feats.parquet',
                '--oof', DS + '/work/devset_india_n20000_oof.parquet',
                '--neg', '3', '--prob', 'blend'], cwd=SRC, env=env, check=True)

# 4) full-test inference (subprocess => multiprocessing spawn is clean)
ncpu = os.cpu_count() or 4
cmd = [sys.executable, SRC + '/infer.py', '--topk', '200',
       '--batch-s1', '5000', '--procs', str(ncpu)]
print('running:', ' '.join(cmd), '(ncpu=%d)' % ncpu, flush=True)
t0 = time.time()
subprocess.run(cmd, cwd=SRC, env=env, check=True)
print('infer done in %.1f min' % ((time.time() - t0) / 60), flush=True)

# 5) validate output format (no ground truth needed)
subprocess.run([sys.executable, DS + '/utils/validate_submission.py',
                '--matching', '/kaggle/working/matching_results.tsv',
                '--candidate', '/kaggle/working/candidate_pairs.tsv',
                '--test-dir', DS + '/clean/test', '--check-ids'], check=False)
print('TSV outputs:', [x for x in os.listdir('/kaggle/working') if x.endswith('.tsv')], flush=True)
'''

MD = ("# Business Entity Resolution — full-test inference\n\n"
      "Offline (internet off), **not** a competition submission — compute + private "
      "storage only. Retrains the deploy LGB+XGB blend on the uploaded India dev "
      "matrix, scores every test S1 (India/US/France) at t\\*=0.98, and writes "
      "`matching_results.tsv` + `candidate_pairs.tsv` to the kernel output.")

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": CODE.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matching.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
