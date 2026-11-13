# 03 — Evaluation Framework

| | |
|---|---|
| **Scope reference** | §13 Evaluation Metrics, §14 Experimental Fairness, §20 Results Table |
| **Depends on** | 01, 02 (qrels loaded and verified) |
| **Produces** | `evaluation/metrics.py`, unit tests, `results.csv` schema, significance-testing script |
| **Suggested effort** | 1–2 work sessions |
| **Course link** | Module 3: Cranfield paradigm, Precision/Recall, MRR, nDCG |
| **Owner** | `[fill in]` |

Build and test this **before** any retrieval experiment. A wrong metric implementation invalidates everything downstream, and it is much cheaper to find it now.

---

## 1. Goal & Exit Criteria

One script evaluates every experiment the same way and writes comparable, auditable output.

- [ ] Metrics: nDCG@10, MRR, Recall@10, Recall@100, Recall@1000, Success@10.
- [ ] Per-query output saved for every run (needed for significance tests and error analysis).
- [ ] Unit tests pass on hand-built toy cases.
- [ ] Cross-check against a second implementation agrees.
- [ ] A paired significance test between any two runs works.

---

## 2. Metric Definitions (write these in the report too)

With exactly one relevant document per query (check in plan 02), the metrics simplify. Let `r` be the rank of the target in the ranked list (`r = ∞` if absent).

| Metric | Definition for one query | Note |
|---|---|---|
| **MRR** | `1/r` | Decide and state the cutoff: full list depth (e.g. 1000) vs. unbounded. Use the same everywhere |
| **nDCG@10** | `1/log2(r+1)` if `r ≤ 10`, else `0` | The ideal DCG is 1 with a single relevant doc, so no extra normalization effect |
| **Recall@K** | `1` if `r ≤ K` else `0` | Averaged over queries it is the fraction of queries with the target in the top K |
| **Success@K** | `1` if any relevant doc in top K | **Identical to Recall@K** when each query has one relevant doc |

Because Success@10 equals Recall@10 here, report one value and add a one-line note in the results table caption. Do not present them as two independent findings.

Also record, as supporting diagnostics:
- **Rank of target** per query (the raw material for everything else).
- **Median rank** and **mean reciprocal rank** over *found* queries only (clearly labeled, not mixed with the main MRR).
- **Not-found rate** within depth 1000.

---

## 3. Implementation Plan

### 3.1 `evaluation/metrics.py`

Two layers, deliberately separate:

1. **Own simple implementation** operating on `{qid: [docid, ...]}` and `{qid: target_docid}`. It is about 30 lines and doubles as a teaching artifact for the viva.
2. **Library implementation** with `ranx` (and `trec_eval` / `pytrec_eval` as a third check) for the headline numbers and significance testing.

The two must agree to numerical precision on every run. A disagreement is a bug to investigate, not something to average away.

### 3.2 Input contract

- Run file: `qid Q0 docid rank score run_name` (TREC format). Re-sort by score yourself; do not trust the rank column blindly.
- Handling rules, all applied identically to every run:
  - Query missing from the run → counts as a miss (score 0), and the number of missing queries is **printed**.
  - Duplicate docids within a query's list → keep the first occurrence and log a warning.
  - Lists longer than 1000 → truncated at 1000 for every metric.
  - Evaluation set = all 536 Dev3 qrels queries. Never silently evaluate on the subset a run happened to produce.

### 3.3 Output schema

`evaluation/per_query/<run_name>.csv`

| qid | target_docid | rank_of_target | rr | ndcg10 | r10 | r100 | r1000 |
|---|---|--:|--:|--:|--:|--:|--:|

`evaluation/results.csv` (one row per run)

| exp_id | run_name | split | corpus_tier | n_queries | ndcg10 | mrr | r10 | r100 | r1000 | ci_low_ndcg10 | ci_high_ndcg10 | config_hash | date |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|

`corpus_tier` is carried into every row so that full-corpus and reranking-only results can never be confused (scope §15.2).

---

## 4. Test Plan for the Evaluator

Write these as real `pytest` tests in `evaluation/tests/` and run them in CI or at notebook start.

| # | Test | Expected |
|---|---|---|
| T1 | Perfect run (target at rank 1 for all queries) | nDCG@10 = MRR = R@10 = R@100 = R@1000 = 1.0 |
| T2 | Target at rank 2 for all queries | MRR = 0.5; nDCG@10 = 1/log2(3) ≈ 0.6309; R@10 = 1 |
| T3 | Target at rank 11 for all queries | nDCG@10 = 0; R@10 = 0; R@100 = 1; MRR = 1/11 |
| T4 | Target absent from the list | all metrics 0 |
| T5 | Query missing from run | counted as 0; missing count reported |
| T6 | Mixed toy set of 5 queries with known ranks | hand-computed means match |
| T7 | Random run over real Dev3 qrels | all metrics very close to 0 (sanity ceiling for chance) |
| T8 | Own implementation vs `ranx` vs `trec_eval` | agree within ~1e-6 |
| T9 | Duplicated docids in a list | handled as specified in §3.2 |

Add one end-to-end **plumbing test** on real data: an "oracle" run built **only for testing** from the qrels (target at rank 1) must score 1.0 through the real pipeline code path. Never let oracle runs into `results.csv`; give them a `TEST_` prefix and delete them.

---

## 5. Statistical Comparison

With 536 queries, small differences can still be noise. Plan for it up front.

- **Paired tests on per-query scores.** Compare run A vs. run B on the same queries. Options: paired randomization (Fisher) test or paired t-test (both available in `ranx.compare`), and a paired **bootstrap** confidence interval for the metric difference.
- **Multiple comparisons.** You will compare many pairs (up to 36 for nine experiments). Pre-register the **planned comparisons** instead of testing everything:

| Planned comparison | Question (scope §19) |
|---|---|
| E1 vs E2 | Q2: does dense help vs. BM25? |
| E1/E2 vs E3 | Q3: does hybrid improve robustness? |
| E1 vs E4, E2 vs E5, E3 vs E6 | Q4: does decomposition help each retriever? |
| E3 vs E7 | Q5: does reformulation help? |
| E6 vs E8, E7 vs E8 | Q6: are they complementary? |
| E8 vs E9 | Q7: does reranking help? |

  Apply a correction (Bonferroni or Holm) across this family of about 9–10 tests and say which one you used.
- **Report effect sizes with confidence intervals**, not only p-values. A table of "p < 0.05" markers without magnitudes is weak evidence.
- Metric choice for the headline significance test: nDCG@10 and MRR (per scope §13). Recall@K differences are binary per query, so use McNemar's test or the paired bootstrap on those.

---

## 6. Breakdown Analyses (feed plan 09)

- **By domain** (movie / landmark / celebrity). If the query files do not carry a domain label, derive one **without the LLM and without target leakage into the pipeline**: for analysis only, look up the *target document's* category or infobox type from the corpus after the experiments are finished. Document the method. This is an analysis-time label, not an input to any system.
- **By query length** (e.g. quartiles).
- **By rank-of-target bucket:** rank 1, 2–3, 4–10, 11–100, 101–1000, not found.
- **Pairwise win/tie/loss table:** for each planned comparison, count queries where B has the better rank, equal, or worse rank than A.
- **Rank-shift scatter:** log-rank of target in A vs. B.

---

## 7. Fairness Checklist (scope §14) — verified by code, not memory

Add an `assert_fair(run_list)` helper that checks and prints:

- [ ] Same query set (536 Dev3 IDs) for every run
- [ ] Same qrels file hash
- [ ] Same corpus tier label across runs in a comparison (or an explicit warning)
- [ ] Same retrieval depth (1000) and same `fuse_out` depth
- [ ] Prompt version recorded for LLM-based runs, and equal to the frozen version
- [ ] Config hash recorded

---

## 8. Reproducing Official Numbers (optional but valuable)

If official baseline runs for Dev3 are available (see plan 02, Step A), evaluate them with **your** script. Two uses:

1. Validates your evaluator against the track's reporting conventions.
2. Gives a reference BM25/dense point for the report (clearly labeled "official baseline run, evaluated with our script", not as your own E1/E2).

---

## 9. Deliverables

1. `metrics.py` + `tests/` (all green).
2. `evaluate_run.py` CLI: `python evaluate_run.py runs/E1_dev3_....run --exp E1`.
3. `results.csv` and per-query CSV generation.
4. `compare_runs.py` for planned paired comparisons with correction.
5. `make_results_table.py` that renders the scope §20 table directly from `results.csv` (so the report can never contain a hand-typed number).

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Different MRR cutoffs across tables | Fix one definition in this file; enforce in code |
| Evaluating only on queries a run returned | Missing queries count as zero; print their number |
| Reporting Success@10 and Recall@10 as different | Caption note; single column |
| Overclaiming from tiny differences | Paired tests + CIs + multiple-comparison correction |
| Hand-copied numbers drift from files | Generate every table from `results.csv` |

## 11. Definition of Done

- [ ] All T1–T9 pass.
- [ ] The oracle plumbing test passes and its output is deleted.
- [ ] A dummy run of random rankings produces a believable near-zero score.
- [ ] You can explain nDCG@10 and MRR on a one-relevant-document task on a whiteboard.
