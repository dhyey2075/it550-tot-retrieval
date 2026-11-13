# 08 — Cross-Encoder Reranking (E9)

| | |
|---|---|
| **Scope reference** | §10, §11 (E9), §17 Phase 5, question Q7 |
| **Depends on** | 07 (E8 candidates), 03 (evaluator) |
| **Produces** | frozen reranker configuration (D-09), cached scores, run E9 |
| **Suggested effort** | 2–4 work sessions |
| **Course link** | Module 7 (bi-encoder vs cross-encoder, retrieval re-ranking) |
| **Owner** | `[fill in]` |

---

## 1. Goal & Exit Criteria

Test Q7: **can a cross-encoder improve the ordering of the candidates E8 already retrieved?**

- [ ] Reranker model, input format, depth `K`, and query side chosen on **Dev1** and frozen (D-09).
- [ ] Cross-encoder scores cached per `(qid, docid, model, input-format)` so reruns are free.
- [ ] E9 run on Dev3 and evaluated; E8 vs E9 planned comparison done.
- [ ] Compute cost (time, GPU memory) recorded.

---

## 2. What a Cross-Encoder Does Here

A bi-encoder (plan 04) embeds query and document **separately**. A cross-encoder reads `[query, document]` **together** and outputs one relevance score, which is usually more precise and much more expensive. That is why scope §10 limits it to the top 50–100 candidates.

```text
E8 fused list ──► top-K (K = 50…100) ──► cross-encoder scores ──► re-sorted top-K ──► E9
                  └─ ranks > K keep their E8 order below the reranked block ─┘
```

### Consequences you can predict before running (use them as sanity checks)

- Documents ranked below `K` in E8 keep their E8 order. So **Recall@K' for any cutoff `K' ≥ K` is identical for E8 and E9**. A change there means a bug.
- Only metrics sensitive to order inside the top `K` can change meaningfully: nDCG@10, MRR, Recall@10.
- The ceiling of E9 is E8's Recall@K. If E8 does not contain the target in its top `K`, no reranker can fix that query. Report E8's Recall@K as the "reachable fraction".

---

## 3. Design Decisions to Freeze on Dev1 (D-09)

### 3.1 Model choice

Shortlist 2–3 cross-encoders that fit a Colab T4 in fp16. Typical families:

| Family | Notes |
|---|---|
| MS MARCO MiniLM-class cross-encoders | small and fast; trained on web-passage relevance |
| BGE-reranker-class models | stronger multilingual/general rerankers; larger |
| monoT5-class models | used in published ToT pipelines; heavier |

Pick on Dev1 by nDCG@10 and MRR, prefer the smaller model when results are within bootstrap noise, and record the model card, maximum input length, and license.

### 3.2 Input format
- Query side: the **original query text** is the primary choice, because it holds the user's full phrasing and the cross-encoder can read nuance. Ablation: the **reformulated** query, and `original + reformulated` concatenated. Choose on Dev1 and record the choice.
- Document side: `title + lead text` truncated to the model's maximum length minus the query length. State the truncation clearly, since long Wikipedia pages may carry the matching clue outside the truncated window.
- Keep the format identical for every query.

### 3.3 Depth `K`
Evaluate `K ∈ {20, 50, 100}` on Dev1. Trade-off: larger `K` raises the ceiling and the cost. Published work on ToT pipelines reports that the best depth for a cross-encoder stage was not simply the largest, so test rather than assume. Choose the smallest `K` within noise of the best.

### 3.4 Score usage
- **Primary:** sort the top `K` by cross-encoder score only (matches scope).
- **Ablation (Dev1, optional):** interpolate cross-encoder score with the E8 RRF score or rank. If it helps, report it as an *extra* variant, not as E9.

---

## 4. Implementation Plan

1. Load `candidates_E8_dev3.jsonl` from plan 07 (top-100 per query, with provenance).
2. Build `(qid, docid, query_text, doc_text)` pairs for the top `K`.
3. Score in **batches** with fp16 and `torch.no_grad()`; sort pairs by length to reduce padding.
4. **Write scores to the cache after each batch** (resumable on disconnect).
5. Re-sort per query; append the rest of the E8 list unchanged below the reranked block; write the TREC run file.
6. Evaluate with plan 03; run the planned E8-vs-E9 paired tests.

### Cost estimate (do the arithmetic, then measure)
Pairs to score = `536 queries × K`. For `K = 100` that is 53,600 pairs. Time ten batches on your GPU and extrapolate. Expect minutes to well under an hour for small models, longer for large ones; use your own measurement in the report, not a guess.

### Tier-B note
If plan 02 chose Tier B (dense is only a reranker), E9 becomes the *third* stage on top of a candidate list produced by BM25 plus dense reranking. Describe the pipeline stage by stage in the report so readers do not confuse the stages.

---

## 5. Analysis Tasks (feed plan 09)

- [ ] E8 vs E9 on nDCG@10, MRR, Recall@10 with paired tests and CIs.
- [ ] **Reachable fraction:** share of queries whose target is within E8's top `K`. Report E9's performance *conditioned on* reachable queries as well, to separate "reranker weakness" from "candidate-set weakness".
- [ ] Rank movement: histogram of `rank_E9 − rank_E8` for the target; count promoted, demoted, unchanged.
- [ ] Queries where E9 **hurt**: look at the pairs. Are they cases where the target page's lead section lacks the clue, or where a popular, semantically similar page wins (scope "semantic false positive" / "entity ambiguity")?
- [ ] Per-domain effect (movie / landmark / celebrity).
- [ ] Query-side ablation (original vs reformulated vs concatenated) as a Dev1 table in the report appendix.
- [ ] Cost-benefit: seconds per query for E9 vs. the gain.

---

## 6. Stretch Options (only if everything else is done)

These are *outside* the nine-experiment scope; label them as extras.

- **Chunked scoring:** split long pages into passages, score each, and take the max, to reduce the truncation problem.
- **Cross-encoder as a reranker for E7 or E6 instead of E8:** shows whether reranking gains depend on first-stage fusion quality.
- **LLM listwise reranking:** the scope's boundary list excludes fine-tuning and a production-scale build, but a small zero-shot listwise reranker over top-10 might be feasible. Keep it clearly separate from E9 and make sure the same no-target-leakage rules hold.

---

## 7. Deliverables

1. `reranking/cross_encoder.py` with batching, caching, and tests (e.g. cached scores are reused; ordering is deterministic).
2. `cache/ce_scores/` keyed by model and input format.
3. E9 run file, config snapshot, and timing log.
4. Decision row D-09 with Dev1 comparison tables (model, `K`, query side).
5. `analysis/reranking_analysis.md` with the reachable-fraction analysis and rank-movement plots.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GPU out-of-memory with long inputs | Smaller batches; truncate; fp16; sort by length |
| Reranker lowers performance vs E8 | Valid result; analyze with the reachable-fraction and failure inspection |
| Truncation hides the clue | Document the limit; optional chunked scoring |
| Accidentally tuning `K` on Dev3 | Choose `K` on Dev1; freeze before running Dev3 |
| Domain mismatch (web-search training vs. Wikipedia ToT) | State as a limitation; compare two models on Dev1 |
| Recall@K' differs between E8 and E9 for `K' ≥ K` | Treat as a bug; add an automated assertion |

## 9. Definition of Done

- [ ] E9 row in `results.csv` with the tier label and `K`.
- [ ] Automated check confirms Recall@K' equality for `K' ≥ K`.
- [ ] You can explain in two sentences why a cross-encoder is accurate but cannot be run over the full corpus.
