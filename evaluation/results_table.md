# Results table (generated from results.csv)

| Exp. | Method (run_name) | corpus_tier | n | nDCG@10 | MRR | R@10 | R@100 | R@1000 |
|---|---|---|--:|---:|---:|---:|---:|---:|
| E1 | E1_dev1_tierB_orig_20261112 | dense_rerank_bm25 | 142 | 0.030516 | 0.030096 | 0.042254 | 0.112676 | 0.239437 |
| E2-R | E2-R_dev1_tierB_orig_20261112 | dense_rerank_bm25 | 142 | 0.021127 | 0.022234 | 0.021127 | 0.042254 | 0.239437 |
| E3 | E3_dev1_tierB_orig_20261112 | dense_rerank_bm25 | 142 | 0.027193 | 0.027347 | 0.035211 | 0.119718 | 0.239437 |
| E4 | E4_dev1_tierB_extractive_20261112 | dense_rerank_bm25 | 142 | 0.006587 | 0.004074 | 0.021127 | 0.056338 | 0.197183 |
| E5-R | E5-R_dev1_tierB_extractive_20261112 | dense_rerank_bm25 | 142 | 0.006665 | 0.005713 | 0.014085 | 0.056338 | 0.077465 |
| E6 | E6_dev1_tierB_extractive_20261112 | dense_rerank_bm25 | 142 | 0.009515 | 0.008405 | 0.021127 | 0.077465 | 0.176056 |
| E7 | E7_dev1_tierB_keyword_20261112 | dense_rerank_bm25 | 142 | 0.013722 | 0.011311 | 0.028169 | 0.056338 | 0.204225 |
| E8 | E8_dev1_tierB_f2_twolevel_20261112 | dense_rerank_bm25 | 142 | 0.010099 | 0.005943 | 0.028169 | 0.056338 | 0.140845 |
| E9 | E9_dev1_tierB_k30_orig_20261112 | dense_rerank_bm25 | 142 | 0.038603 | 0.034832 | 0.056338 | 0.098592 | 0.232394 |
| E4c | E4c_dev1_tierB_concat_20261112 | dense_rerank_bm25 | 142 | 0.026905 | 0.023367 | 0.042254 | 0.070423 | 0.246479 |
| E7-bm25 | E7-bm25_dev1_tierB_keyword_20261112 | dense_rerank_bm25 | 142 | 0.002347 | 0.002428 | 0.007042 | 0.049296 | 0.204225 |
| E7-dense | E7-dense_dev1_tierB_keyword_20261112 | dense_rerank_bm25 | 142 | 0.029091 | 0.027845 | 0.035211 | 0.063380 | 0.204225 |
| E8-F1 | E8_dev1_tierB_f1_flat_20261112 | dense_rerank_bm25 | 142 | 0.010099 | 0.005943 | 0.028169 | 0.056338 | 0.140845 |
| E8-F4 | E8_dev1_tierB_f4_plusorig_20261112 | dense_rerank_bm25 | 142 | 0.031237 | 0.028053 | 0.049296 | 0.098592 | 0.232394 |
| E9-reform | E9_dev1_tierB_k30_reform_20261112 | dense_rerank_bm25 | 142 | 0.030039 | 0.024101 | 0.056338 | 0.098592 | 0.232394 |
| E9-concat | E9_dev1_tierB_k30_concat_20261112 | dense_rerank_bm25 | 142 | 0.038603 | 0.034832 | 0.056338 | 0.098592 | 0.232394 |

_Caption note: with one relevant document per query, Success@10 = Recall@10; only Recall@10 is shown. Unrun experiments stay absent (do not invent TBD rows here — add them in the report template only)._
