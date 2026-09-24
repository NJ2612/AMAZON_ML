# Business Entity Resolution — Pipeline

Self-contained, runnable pipeline for the ML Challenge 2026 Business Entity
Resolution task. It reads the 3-source business records, generates candidate
pairs (blocking), scores them with a matching model, and writes the two
submission files.

> This is the reproducible-code package described in the challenge README's
> *Final Submission Package* section. Fill in the sections marked _TODO_ as the
> pipeline is built.

## Layout

```
code/business_entity_resolution/
├── src/                 # all source code
├── README.md            # this file
└── requirements.txt     # pinned dependencies
```

Data and outputs live at the repository root, not inside this folder:

```
../../dataset/train/     # train_source1/2/3.tsv, train_ground_truth.tsv
../../dataset/test/      # test_source1/2/3.tsv
../../output/            # matching_results.tsv, candidate_pairs.tsv (generated)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce end-to-end

_TODO: wire these to the actual entry points as `src/` is built._

```bash
# From the repository root.

# 1. Candidate generation (blocking) -> output/candidate_pairs.tsv
python code/business_entity_resolution/src/blocking.py \
    --test-dir dataset/test \
    --out output/candidate_pairs.tsv

# 2. Matching model inference -> output/matching_results.tsv
python code/business_entity_resolution/src/match.py \
    --candidates output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --out output/matching_results.tsv

# 3. Validate both files before submitting
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Notes

- All files are **tab-separated** — always read/write with `sep="\t"`.
- `country` is an open set of labels (train has `US`/`India`; test adds
  `France`). Do not hard-code the country set.
- The scoring metric is macro-averaged **F_0.5** (precision-weighted); correctly
  predicting singletons (empty match list) earns full credit.
- **No external data lookup** is allowed — only the provided training data.
