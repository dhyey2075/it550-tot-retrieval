# 10 — Report, Presentation, Submission & Risk Register

| | |
|---|---|
| **Scope reference** | §21 Major Outcomes, §22 Scope Boundaries, §25 Final Research Question, §27 One-Sentence Summary |
| **Depends on** | 01–09 |
| **Produces** | final report, slides, clean repository, viva preparation notes, risk register, scope-cut ladder |
| **Suggested effort** | 4–6 work sessions (start the report skeleton in week 2, not at the end) |
| **Course context** | IT550 weights the course project at **35%** of the grade; lab assignments 20%, so keep a few hours for those separately |
| **Owner** | `[fill in]` |

---

## 1. Goal & Exit Criteria

- [ ] A report that answers the final research question (scope §25) using only real, traceable numbers.
- [ ] A repository a stranger can run from the README, in order.
- [ ] A presentation/viva kit: slides, a one-minute pitch, and rehearsed answers to likely questions.
- [ ] Risk register kept alive through the whole project, with a pre-agreed scope-cut ladder.

---

## 2. Report Outline (mapped to the scope)

Suggested length: 12–18 pages plus appendices. Adjust to whatever the course requires.

| § | Section | Content | Source |
|--:|---|---|---|
| 1 | Abstract | 150–200 words: problem, method, setting, main finding | write last |
| 2 | Introduction | ToT as known-item retrieval; vocabulary mismatch; the research question (scope §25); contributions stated modestly (scope §1: integration and evaluation study, not a new architecture) | scope §1–2 |
| 3 | Related work | ToT task and TREC-ToT 2025; **Lin et al. 2023**: field-based decomposition, extractive vs. predictive prompting, specialized experts, weighted combination with a base retriever; **Mukhopadhyay et al. 2026**: keyword-focused guarded rewriting, BM25 → dense → monoT5 → listwise-LLM cascade, PRF failure; BM25, dense retrieval, RRF, cross-encoders. State clearly what you adapted (plans 05, 06) | scope §4–5 + course Modules 2, 7, 8 |
| 4 | Dataset and setting | Dev3 (536 queries), split roles, corpus tier (D-01), document representation, why Test is untouched | plan 02 |
| 5 | Methods | E1–E9 pipelines with the pipeline diagram (scope §26), prompts (appendix), RRF formula, fusion protocol, cross-encoder | plans 04–08 |
| 6 | Experimental setup | Environment, libraries and versions, hyperparameters and how they were chosen (Dev1), caching, fairness measures | plans 01, 03 |
| 7 | Results | Main table with CIs; planned comparisons; recall curves; per-domain; cost | plan 09 |
| 8 | Analysis | Q1–Q7 answers; complementarity; reranking reachable-fraction | plan 09 |
| 9 | Error analysis | Failure taxonomy counts; worked examples | plan 09 |
| 10 | Limitations and threats to validity | list from plan 09 §6 | plan 09 |
| 11 | Conclusion and future work | what the evidence supports; what it does not; natural next steps (full-corpus dense, more domains, agentic/iterative retrieval) | — |
| A | Appendices | prompts (all frozen versions), decision log, hyperparameter grids, annotation guideline, extra ablations | repo |

### Writing rules (non-negotiable)

1. **Every number comes from a generated file.** No typed-in results. Unrun cells stay `TBD` (scope §20) until they are actually run.
2. **Claims match evidence.** "Improves" requires a paired test; otherwise write "numerically higher, not statistically significant".
3. **Label each setting** (full-corpus vs. dense reranking of BM25 candidates vs. reduced corpus) in captions.
4. **Say what you adapted, not reproduced** from the two papers (scope §4.1, §5.1).
5. **Do not compare absolute scores** with the published numbers unless the setting is identical.
6. **Report negative and null results** with the same care as positive ones.
7. **Disclose LLM use in the pipeline** (model id, temperature, caching) and state that no target names or qrels reach the LLM.
8. **Post-hoc changes** are labeled as post-hoc.

### Figures to prepare

1. Pipeline diagram (scope §26), redrawn cleanly.
2. Experiment progression E1→E9 (scope §12).
3. Rank-of-target stacked bars (plan 09, A3).
4. Recall@K curves (A7).
5. Complementarity overlap diagram (A8).
6. Failure-label distribution (plan 09 §4.5).
7. One worked example, shown end-to-end (query → clues → rewrite → ranks).

---

## 3. Mapping to IT550 Course Outcomes

Use this in the report's introduction or a "course relevance" paragraph if your instructor wants it.

| Course outcome | Where this project demonstrates it |
|---|---|
| **CO1:** indexing, query processing, similarity, ranking | Inverted-index BM25, RRF, vector similarity (plans 02, 04) |
| **CO2:** evaluate ranking models with Precision/Recall/MAP/MRR/NDCG | Plan 03 and 09; note you use the metrics suited to one relevant item (nDCG@10, MRR, Recall@K) |
| **CO3:** relevance feedback, query expansion, classification, clustering | LLM-based query expansion/reformulation as a modern relative of Rocchio/RM3; optional RM3 run (plan 06) |
| **CO4:** neural IR: dense retrieval, transformers, vector search | Bi-encoder retrieval, FAISS, cross-encoder reranking (plans 04, 08) |
| **CO5:** LLM-based IR systems | Decomposition and reformulation with an LLM, hybrid retrieval (plans 05–07) |

Course topics your project touches: BM25 (Module 2), evaluation (Module 3), query expansion (Module 4), dense retrieval, Bi/Cross-encoder, FAISS (Module 7), hybrid retrieval and limitations of classical/neural retrieval (Module 8).

---

## 4. Repository Cleanup Checklist

- [ ] `README.md`: purpose, one-paragraph results summary (copied from generated files), then **run order** (plan files 01→09) with expected runtimes.
- [ ] `requirements.txt` pinned; Colab setup cell included.
- [ ] `data/MANIFEST.md`, `configs/decisions.md`, `prompts/CHANGELOG.md` present and complete.
- [ ] All caches (`cache/*.jsonl`) included (they are small) so graders can reproduce without paying for API calls.
- [ ] Large artifacts (indices, embeddings) excluded from git; a script or link explains how to rebuild.
- [ ] No API keys, tokens or personal Drive paths in the notebook or history. If a key was ever committed, rotate it.
- [ ] Notebook runs top-to-bottom in a fresh Colab runtime (test this once at the end).
- [ ] Every table/figure in the report has the generating script named in the caption or the appendix.
- [ ] Git tags for frozen states: `baselines-v1`, `prompts-final`, `e8-protocol-v1`, `ce-config-v1`, `final-submission`.
- [ ] Confirm **no Test split** loading anywhere in the repo (`grep` for the split name).

---

## 5. Presentation & Viva Kit

### 5.1 Slide plan (about 10–12 slides)

1. Title and team
2. The ToT problem with one example query
3. Why it is hard: vocabulary mismatch, noisy memory
4. Research question and the two papers
5. Dataset and setting (Dev3, split discipline, corpus tier)
6. Pipeline diagram
7. The nine experiments as a progression
8. Main results table
9. What helped, what did not (A2/A4)
10. Error analysis and one worked example
11. Limitations and honest caveats
12. Conclusion and future work

### 5.2 One-minute pitch

> Users often remember what something is like but not its name. We test whether using an LLM to break a vague description into clues, or to rewrite it as a clearer query, helps BM25, dense retrieval and their fusion find the right Wikipedia page, and whether a cross-encoder improves the final ranking. We compare nine controlled configurations on 536 TREC-ToT 2025 Dev3 queries and analyze why they succeed or fail.

### 5.3 Questions you should be ready for

| Question | Where the answer lives |
|---|---|
| Why Dev3 and not the official test set? | Plan 02 §5: Test is held out; Dev3 is large and covers three domains |
| Is Dev3 smaller than the corpus? | No: Dev3 is a query split; retrieval still needs the large corpus or a labeled substitute |
| Did the LLM see the answers? | Pipeline never sees qrels/titles; audit rates reported; training-data contamination acknowledged |
| What exactly is RRF and why use it? | Plan 04 §4.1: rank-based fusion that avoids calibrating BM25 vs. cosine scores |
| Why can't a cross-encoder run on the whole corpus? | It scores query-document pairs jointly; cost scales with documents |
| Why do nDCG@10 and MRR suit ToT? | One relevant item; rank position matters (plan 03 §2) |
| Is this a new method? | No: it is an adaptation and controlled comparison of existing ideas |
| How does your decomposition differ from Lin et al.? | Plan 05 §2.1: facets instead of book fields, zero-shot BM25/dense instead of fine-tuned experts, RRF instead of weighted scores, extractive primary instead of predictive; predictive is an extension |
| The paper found predictive prompting better, so why is extractive your primary? | Scope §7 forbids unsupported facts; predictive is run as a labeled extension with a hallucination audit |
| Why does reformulation help BM25 more than dense? | Keyword-style rewrites match BM25 but dense models prefer natural text (2026 paper, Table 7; Lin et al. Table 4); tested via `E7-bm25`, `E7-dense`, `E7-split` |
| Why not match the 2026 paper's numbers? | Different LLM, tuned BM25, corpus tier, hardware (it used 500 GB RAM and two RTX 6000 GPUs); we use its numbers only as sanity checks |
| Did pseudo-relevance feedback help? | `E-PRF` (plan 06): the 2026 paper found it hurt; we report whether we replicate |
| What would you do with more compute? | Full-corpus dense index, larger rerankers, more domains |
| What failed? | Failure taxonomy and counts (plan 09) |
| How do you know differences are not noise? | Paired tests, CIs, multiple-comparison correction |

Prepare one whiteboard explanation each for: BM25, RRF, bi-encoder vs. cross-encoder, nDCG@10 on a single relevant document.

---

## 6. Risk Register (review weekly)

Likelihood (L) and impact (I): H / M / L.

| # | Risk | L | I | Early warning | Mitigation | Fallback |
|--:|---|:-:|:-:|---|---|---|
| R1 | Full-corpus indexing infeasible on Colab Free | H | H | Step B arithmetic in plan 02 | Decide the tier in week 1 | Tier B, then Tier C; label clearly |
| R2 | Colab disconnects lose work | H | M | Any run > 1 hour | Sharded, resumable jobs; cache after each unit | Smaller batches; shorter sessions |
| R3 | LLM API rate limits or cost | M | M | 429 errors in pilot | Backoff, cache, start generation early | Switch provider; reduce pilot size, never reduce Dev3 coverage |
| R4 | LLM leaks the target | M | H | Audit flags in pilot | Prompt rules; heuristic checks; regenerate fully | Report leakage rate; note limitation |
| R5 | Tuning on Dev3 by accident | M | H | Prompt edits after Dev3 look | Freeze tags; decision log | Label as post-hoc; rerun as an extra |
| R6 | Dense model/prefix misconfigured | M | M | Dense far below BM25 | Check model card; test prefix on Dev1 | Try second model |
| R7 | Metrics bug | L | H | Own vs. `ranx` mismatch | Plan 03 tests | Fix before any final run |
| R8 | Null or negative results | M | L | — | Report honestly; use error analysis | Reframe conclusion around "when does LLM help" |
| R9 | Time runs out | M | H | Gates in plan 01 slip | Scope-cut ladder (§7) | Drop from the top of the ladder |
| R10 | Team coordination gaps (if more than one person) | M | M | Duplicate or conflicting runs | Owners in each plan file; shared decision log | Weekly sync, one person owns evaluation |
| R11 | Qrels/docid mismatch | M | H | Perfect-run test fails | Normalize IDs once; assert coverage | Debug before baselines |
| R12 | Colab/library version drift | L | M | Install errors | Pin versions; record env logs | Re-pin and rerun smoke test |
| R13 | Predictive decomposition or keyword rewriting hallucinates or leaks names | M | H | Heuristic flags in pilot | Extractive primary; ban non-query proper names; audits; regenerate everything if the prompt changes | Report the leak rate; drop the extension |
| R14 | Dev1 tuning does not transfer to Dev3 (BM25 nDCG@10 0.083 vs. 0.337 in the paper) | H | M | Large Dev1 vs. Dev3 gap | Robust aggregate across splits; small grids; optional Dev2 (A1) | State as a limitation; avoid claims from Dev1 absolutes |
| R15 | Rewrite or facet outputs vary between API runs | M | M | Stability CV above 5% | Cache first generation; report mean ± sd (plans 05, 06) | Report variability explicitly |

---

## 7. Scope-Cut Ladder (decide now, use only if needed)

If time or compute runs short, cut from the top. Anything cut must be stated plainly in the report.

| Order | Cut | Cost to the study |
|--:|---|---|
| 1 | Optional ablations and extras (RM3 run, chunked scoring, weighted fusion) | none to the nine experiments |
| 2 | Worked examples beyond four; reduce annotation sample sizes evenly | smaller error analysis |
| 3 | E9 reranker bake-off: use a single cross-encoder, fixed `K = 50` | less tuning evidence |
| 4 | Dense per-clue experiments (E5) via the Tier-B reranking setting if not already | weaker dense claims; label it |
| 5 | Move from Tier A/B to the Tier C shared sub-corpus | results not comparable to official numbers |
| 6 | Run Dev3 on the full 536 queries but analyze fewer strata | thinner analysis |
| **Never cut** | frozen prompts, caches, split discipline, evaluator tests, honest labeling of settings | these protect validity |

---

## 8. Final Submission Checklist

- [ ] Report numbers regenerated from final `results.csv` and cross-checked against the PDF.
- [ ] Scope §21 achievements list ticked off, item by item, with the file or section that proves each.
- [ ] Scope §22 boundaries respected: no fine-tuning, no test-set tuning, no target names introduced, no unsupported SOTA claims.
- [ ] Repository tagged `final-submission` and archived (zip) with caches.
- [ ] Slides exported to PDF and a backup copy saved offline.
- [ ] Each team member can explain every experiment and the main failure types.
- [ ] Lab-assignment deadlines checked separately.

## 9. Definition of Done

- [ ] You can answer the final research question (scope §25) in two honest sentences backed by the results table.
- [ ] A reader who only has the repository and README can reproduce the nine rows.
- [ ] The risk register has no open "High / High" items.
