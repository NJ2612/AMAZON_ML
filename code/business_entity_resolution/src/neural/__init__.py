"""kaggle_2 — GPU neural retrieve-and-rerank entity-resolution pipeline.

A genuinely different architecture from the GBDT blend (src/*.py): a dense
multilingual bi-encoder (LaBSE) retrieves candidates by native-script name
similarity, unioned with the existing lexical blocking, then a cross-encoder
reranker (gte-multilingual-reranker-base) scores each (S1, candidate) pair
jointly. Leakage-safe 5-fold CV reuses common.assign_folds so the reranker OOF
aligns row-for-row with the GBDT OOF for an optional logistic ensemble.

All models are MIT/Apache-2.0 and <=8B params. No external data lookup; only
pretrained weights (shipped in the private Kaggle dataset, internet OFF).
"""
