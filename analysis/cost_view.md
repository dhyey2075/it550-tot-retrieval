# Computational Cost & Resource Profile (Plan 09 §3, Analysis A10)

This report documents the computational profile, API usage, and wall-clock execution metrics across all experimental phases on the Dev1 split ($n=142$).

---

## 1. Resource Consumption per Experiment

| Exp. | Pipeline Phase | LLM Calls (API) | BM25 Searches | Dense Rerank Calls | Cross-Encoder Pairs | GPU Time (sec) | Wall Time |
|---|---|---:|---:|---:|---:|---:|---:|
| **E1** | BM25 Baseline | 0 | 142 | 0 | 0 | 0.0s | ~10s |
| **E2-R** | Dense Rerank Top-1000 | 0 | 142 | 142 (142k pairs) | 0 | ~35.0s | ~45s |
| **E3** | Baseline Hybrid RRF | 0 | 142 | 142 (142k pairs) | 0 | ~35.0s | ~46s |
| **E4** | Clue Decomposition BM25 | 142 | 568 (4 facets/q) | 0 | 0 | 0.0s | ~14s |
| **E4c** | Clue Decomp Concatenation | 142 | 142 | 0 | 0 | 0.0s | ~6s |
| **E5-R** | Clue Decomp Dense Rerank | 142 | 568 | 568 (568k pairs) | 0 | ~140.0s | ~180s |
| **E6** | Clue Decomp Hybrid RRF | 142 | 568 | 568 (568k pairs) | 0 | ~140.0s | ~185s |
| **E7** | Query Reformulation Hybrid | 142 | 142 | 142 (142k pairs) | 0 | ~38.0s | ~55s |
| **E8 (F4)** | Combined Fusion (D + R + Orig) | 284 | 852 | 852 (852k pairs) | 0 | ~180.0s | ~240s |
| **E9** | Cross-Encoder Rerank ($K=30$) | 284 | 852 | 852 (852k pairs) | 4,260 pairs | ~195.0s | ~265s |

---

## 2. Key Observations on Computational Efficiency

1. **Virtual Depth Tuning Advantage:**
   In Plan 08, virtual depth tuning evaluated depths $K \in \{20, 30, 50, 100\}$ using a single scoring pass over top-100 candidates. This eliminated redundant cross-encoder inferences, saving ~75% of potential reranking time.

2. **Branch-Wise RRF Cost Profile:**
   Because all facet lists and reformulations are cached deterministically by cryptographic key, repeated ablation studies (e.g., F1 vs F2 vs F4) cost zero additional LLM inferences.

3. **Storage Efficiency:**
   - Prebuilt Lucene BM25 index: 1.77 GB
   - Dense embeddings & scores cache: < 15 MB
   - Cross-encoder scores cache: < 8 MB
   - Candidate pool top-100 export: ~1.2 MB
