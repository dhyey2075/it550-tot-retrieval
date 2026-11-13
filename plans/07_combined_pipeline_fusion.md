# 07 — Combined Decomposition + Reformulation (E8)

| | |
|---|---|
| **Scope reference** | §9, §11 (E8), §17 Phase 4, central question Q6 |
| **Depends on** | 04, 05 (E6 and its frozen fusion protocol), 06 (E7) |
| **Produces** | frozen E8 fusion protocol (D-08), run E8, complementarity analysis, candidate pool for E9 |
| **Suggested effort** | 2–3 work sessions |
| **Course link** | Module 8 (hybrid retrieval), data fusion ideas from Module 2/7 |
| **Owner** | `[fill in]` |

Scope calls this "one of the central questions of the project". Treat the analysis as the main product, and the run as the input to it.

---

## 1. Goal & Exit Criteria

Decide, with evidence from Dev1, how to fuse the decomposition and reformulation signals, then test scope Q6: **are the two LLM transformations complementary?**

- [ ] E8 fusion protocol chosen on Dev1, written down, frozen (D-08), and tagged **before** the Dev3 E8 run.
- [ ] E8 evaluated on Dev3.
- [ ] Complementarity analysis completed (§5).
- [ ] The top-K candidate pool from E8 saved for plan 08.

---

## 2. Inputs Available (all already cached or run)

For each query:

| List | Source |
|---|---|
| `BM25(ci)` for each clue | plan 05 |
| `Dense(ci)` for each clue | plan 05 |
| `BM25(reformulated)` | plan 06 |
| `Dense(reformulated)` | plan 06 |
| *(optional)* `BM25(original)`, `Dense(original)` | plan 04 |

No new LLM calls and no new retrieval are required for the main E8 run. Fusion is cheap, so you can compare several fusion variants on Dev1 within minutes.

---

## 3. Fusion Protocols to Compare on Dev1

Scope §9 lists four sources: the original/decomposed clue queries, the reformulated query, BM25, and dense. These variants cover the sensible ways to combine them.

| ID | Protocol | Description |
|---|---|---|
| **F1 (primary candidate)** | **Flat RRF** over all lists | One RRF over `2n` clue lists plus the 2 reformulation lists. Simple, no tuning |
| **F2** | **Two-level** (branch-wise) | Branch D = E6 fusion (frozen from plan 05). Branch R = E7 fusion. Then RRF(branch D, branch R). Gives each branch equal weight, regardless of clue count |
| **F3** | F2 with **weights** | e.g. weight the reformulation branch higher; a small grid on Dev1 only |
| **F4** | F1/F2 **plus the original query's** lists | Tests whether the raw query still adds anything |

Selection rules (write before looking at numbers):

1. Compare on Dev1 using nDCG@10 and MRR, with R@100 as a tiebreaker.
2. Prefer the **simplest** protocol unless a more complex one wins by a margin clearly outside bootstrap noise.
3. The chosen protocol's treatment of clues must match the one used in E6 (plan 05), so that E6 → E8 isolates the addition of reformulation.
4. Record the full comparison table in the decision log (D-08), then freeze.

Important: the scope says the **exact fusion protocol will be fixed before evaluation and reported clearly**. A tagged config and a decision-log row are your evidence.

---

## 4. Running E8

```text
Dev1: compare F1–F4 → choose → freeze (git tag e8-protocol-v1)
Dev3: run E8 once with the frozen protocol → evaluate (plan 03)
```

- Output depth 1000, same as other experiments.
- Save the **per-source ranks** for every returned doc (which lists contained it and at which rank). This provenance table makes the complementarity analysis and the error analysis much easier.
- Save the top-100 candidates per query, with doc text snippets, to `reranking/candidates_E8_dev3.jsonl` for plan 08.

---

## 5. Complementarity Analysis (the real answer to Q6)

### 5.1 Define "complementary" before computing
Suggested operational definition, to be written in the report **before** you see Dev3 results:

> Decomposition and reformulation are complementary if (a) E8 improves significantly over both E6 and E7 on nDCG@10 and MRR under the planned paired tests, **and** (b) a meaningful share of Dev3 queries have their target retrieved by only one branch within the top-100.

If only (b) holds but (a) does not, report that the signals are *different* but the fusion fails to exploit them. That is an honest and interesting finding.

### 5.2 Analyses

| Analysis | How |
|---|---|
| **Overlap of found targets** | For each query, is the target in top-100 for E6, E7, E8? Build a three-set Venn/UpSet count |
| **Unique contributions** | Queries found only by decomposition, only by reformulation, by both, by neither |
| **Oracle upper bound** | Per query, best rank among E6/E7 → gives the ceiling of any fusion of the two. Compare E8 with the oracle to see how much headroom remains. (Analysis only; the oracle uses qrels, so never use it as a system) |
| **Paired tests** | E6 vs E8 and E7 vs E8 (planned comparisons with correction) |
| **Rank-shift view** | Scatter of log-rank(E6) vs log-rank(E7), colored by whether E8 improved |
| **Fusion sensitivity** | Table of F1–F4 on Dev1 (already collected) to show the protocol was not cherry-picked |
| **Dilution check** | Cases where E8 is worse than the better of E6/E7: what happened? Too many noisy lists? A bad clue outranking a good reformulation? |
| **Cost view** | LLM calls and retrieval calls needed per query for E6, E7, E8 |

### 5.3 Controls worth adding (cheap, high value)
- **Equal-list-count control:** does E8 beat E6 merely because it has more lists to vote with? Run a control where the extra lists come from the *original* query (BM25 + dense of the original) instead of the reformulation. If the control gains as much as E8, reformulation is not adding unique value.
- **Different-random-seed control** is not needed (everything is deterministic given the caches); mention that in the report.

---

## 6. Deliverables

1. `retrieval/hybrid/e8_fusion.py` implementing the frozen protocol and the F1–F4 variants.
2. Decision row D-08 with the Dev1 comparison table.
3. E8 run file, config snapshot, and provenance table.
4. `reranking/candidates_E8_dev3.jsonl` (top-100 per query) for plan 08.
5. `analysis/complementarity.ipynb` or `.md` containing the overlap, oracle and control analyses.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Choosing the protocol after seeing Dev3 | Decide on Dev1; tag; record in decision log |
| Flat RRF overweights the decomposition branch | Compare against two-level fusion (F2) |
| Gains come from "more lists," not from reformulation | Equal-list-count control |
| Overcomplicated weighting overfits Dev1 | Prefer F1/F2 unless a clear margin appears |
| Negative result (E8 not better) | Report it; the oracle and overlap analysis explain why |

## 8. Definition of Done

- [ ] Fusion protocol frozen and tagged before Dev3.
- [ ] E8 row in `results.csv`.
- [ ] Complementarity conclusion written with its criteria, evidence, and the control result.
- [ ] Candidate pool for plan 08 saved and checked (every query has up to 100 candidates; no duplicate docids).
