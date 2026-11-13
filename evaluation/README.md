# Evaluation (plan_v2/03)

One evaluator for every experiment.

## Metrics (single relevant doc)

| Metric | Definition |
|---|---|
| MRR | `1/r` with list truncated at 1000; 0 if absent |
| nDCG@10 | `1/log2(r+1)` if `r≤10` else 0 |
| Recall@K | 1 if `r≤K` else 0 (= Success@K here) |

## Commands

```bash
# From repo root
python -m pytest evaluation/tests tot/tests -q

python -m evaluation.evaluate_run runs/E1_dev3_....run \
  --exp E1 --qrels data/dev3-qrels.txt \
  --split dev3 --corpus-tier dense_rerank_bm25

python -m evaluation.compare_runs \
  --a evaluation/per_query/A.csv --b evaluation/per_query/B.csv

python -m evaluation.make_results_table
```

`TEST_` run names are refused for `results.csv` unless `--no-append` or `--allow-test-prefix`.
