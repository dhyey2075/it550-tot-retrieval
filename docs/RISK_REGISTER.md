# Live Risk Register and Scope-Cut Ladder

**Course:** IT550 — Information Retrieval  
**Project:** Tree-of-Thought Retrieval for Tip-of-the-Tongue Queries  
**Last Updated:** Project Completion / Final Submission Review  

---

## 1. Live Risk Register (R1 – R15)

The risk register was maintained continuously across all 10 project phases to monitor technical feasibility, computational constraints, and experimental validity.

| # | Risk Description | Likelihood (L) | Impact (I) | Early Warning Indicator | Pre-Agreed Mitigation | Fallback Strategy | Final Resolution / Status |
|:---|:---|:---:|:---:|:---|:---|:---|:---|
| **R1** | Full-corpus dense indexing infeasible on Colab Free | **H** | **H** | Step B RAM arithmetic in Plan 02 exceeds 12 GB | Decide tier in week 1; adopt Tier B | Adopt Tier B (`dense_rerank_bm25`) | **CLOSED:** Tier B adopted early. All rerankers evaluate top-1000 BM25 candidates. Zero RAM crashes. |
| **R2** | Colab session disconnects cause loss of intermediate progress | **H** | **M** | Any pipeline run duration $> 45$ minutes | Sharded, resumable jobs; disk checkpointing | Resume from last completed query JSONL | **CLOSED:** Sharded batching and disk caching (`cache/*.jsonl`) preserved all progress across sessions. |
| **R3** | LLM API rate limits or excessive quota cost | **M** | **M** | HTTP 429 errors in initial pilot batch | Exponential backoff, local disk caching | Switch provider or use deterministic fallback | **CLOSED:** Deterministic extractive stub and persistent cache eliminated repeat API calls. |
| **R4** | LLM prompts inadvertently leak target title or qrels | **M** | **H** | Heuristic audit flags in pilot inspection | Strict prompt isolation; automated audit script | Regenerate polluted runs from scratch | **CLOSED:** `scripts/run_fairness_audit.py` confirmed 0 title leaks across all 536 queries (`PASS`). |
| **R5** | Accidental hyperparameter tuning on Dev3 evaluation split | **M** | **H** | Parameter adjustments following Dev3 inspection | Strict split quarantine; freeze tags | Label post-hoc and rerun on frozen params | **CLOSED:** Hyperparameters ($k_1=0.9, b=0.4, k=60, K=30$) locked on Dev1 prior to any Dev3 run. |
| **R6** | Dense encoder prefix or pooling misconfiguration | **M** | **M** | Dense baseline severely degrades ($< 50\%$ BM25) | Verify model card; validate on Dev1 | Evaluate secondary encoder | **CLOSED:** Verified `query:` and `passage:` prefixes for E5-Small. Baseline degradation was proven genuine semantic drift. |
| **R7** | Metric computation discrepancies between custom code and `ranx` | **L** | **H** | Custom vs. standard library divergence $> 10^{-4}$ | Unit test suite (`tests/test_plan03_metrics.py`) | Align with standard TREC evaluation definitions | **CLOSED:** 35/35 automated unit tests passing. Custom metrics match standard known-item definitions. |
| **R8** | Null or negative experimental results | **M** | **L** | E2-R or E5 underperforming classical BM25 | Report negative results with equal scientific rigor | Focus analysis on failure taxonomy & error bounds | **CLOSED:** Negative results (E2-R dense degradation, E4 clue RRF dilution) fully documented and analyzed. |
| **R9** | Project schedule slippage and deadline pressure | **M** | **H** | Milestone gate delays in Plans 01–04 | Execute structured scope-cut ladder | Prune optional ablations sequentially | **CLOSED:** All core milestones (Plans 01–10) delivered on schedule without cutting core experiments. |
| **R10** | Team coordination and file conflict gaps | **M** | **M** | Duplicate runs or overwritten cache files | Explicit plan ownership and decision log | Single coordinator reviews shared assets | **CLOSED:** Single unified execution log in `configs/decisions.md` and automated notebook generation. |
| **R11** | DocID or Qrel identifier normalization mismatch | **M** | **H** | Perfect-run synthetic test yields 0.0 nDCG | Normalize IDs once; assert 100% qrels coverage | Debug mapping pipeline before runs | **CLOSED:** `load_qrels` and `load_run` normalized across all scripts; verified via automated test suite. |
| **R12** | Colab / Python library dependency drift | **L** | **M** | Version conflicts during package installation | Pinned `requirements.txt`; virtualenv locking | Lock exact PyPI wheels in setup cell | **CLOSED:** Dependencies pinned in `requirements.txt` and verified in isolated `.venv312`. |
| **R13** | Predictive decomposition or keyword rewrites hallucinate | **M** | **H** | Audit script detects hallucinated proper nouns | Extractive prompting as primary; name banning | Ban non-query proper nouns; discard invalid | **CLOSED:** Extractive primary preserved. Hallucination audit verified zero synthetic entity pollution. |
| **R14** | Dev1 tuning does not generalize to Dev3 | **H** | **M** | Large discrepancy between Dev1 and Dev3 gains | Coarse grid tuning; robust parameter defaults | Document split sensitivity as limitation | **CLOSED:** Generalization confirmed: Dev1 optimal parameters ($K=30$, RRF $k=60$) held robustly on Dev3. |
| **R15** | LLM generation instability across runs | **M** | **M** | High variance in reformulation outputs ($CV > 5\%$) | Set temperature $T=0.0$; disk cache prompt outputs | Report variance confidence bounds | **CLOSED:** Cached frozen generations at $T=0.0$; deterministic reproducibility guaranteed. |

---

## 2. Scope-Cut Ladder & Execution Log

The scope-cut ladder defines the pre-agreed hierarchy of components to reduce in case of severe computational or time bottlenecks, protecting the core scientific validity of the study.

| Priority | Component / Scope Item | Potential Time/Compute Savings | Cost to the Study | Execution Decision |
|:---:|:---|:---|:---|:---|
| **Level 1** | Optional ablations (RM3 PRF run, multi-model dense bake-off, chunked scoring) | 2–3 compute hours | Minimal: Core 9 experiments remain intact | **APPLIED:** Focused on E1–E9 core progression; RM3 cited as verified failure from literature. |
| **Level 2** | Reduce worked error case studies from 10 to 4; sample 94 stratified queries | 4–6 manual annotation hours | Minor: Statistical error taxonomy remains intact | **APPLIED:** Sampled 94 stratified queries across 9 strata; documented 4 comprehensive case studies. |
| **Level 3** | Cross-encoder bake-off: use single model (MiniLM-L6) at depth $K=30$ | 2 compute hours | Low: Isolates cross-encoder effect cleanly | **APPLIED:** Standardized on MiniLM-L6 with depth ablation ($K \in [10, 30, 50]$). |
| **Level 4** | Dense clue experiments (E5) run via Tier B candidate reranking | 5 compute hours | Moderate: Evaluates dense clues on candidate pool | **APPLIED:** Tier B candidate reranking uniformly applied across all dense and hybrid runs. |
| **Level 5** | Downscale corpus from full Wikipedia (6.4M) to shared sub-corpus | 10+ compute hours | Severe: Loss of comparability with benchmark | **REJECTED (Not Needed):** Preserved full 6.4M Wikipedia corpus in Tier B. |
| **Level 6** | Downsample Dev3 from 536 queries to a 100-query subset | 6 compute hours | High: Substantial loss of statistical power | **REJECTED (Not Needed):** All 536 queries fully evaluated with 5,000 bootstrap resamples. |
| **Level 7** | **NON-NEGOTIABLE CORE:** Frozen prompts, zero-leakage, split quarantine, 95% CIs | 0 hours (Safety) | Fatal: Destroys scientific validity | **STRICTLY ENFORCED:** Zero compromises on split discipline or statistical rigor. |

---

## 3. Risk Audit & Governance Checklist

- [x] **Zero Open High-High Risks:** All initial high-likelihood / high-impact risks (R1, R4, R5, R11) were proactively mitigated and permanently closed.
- [x] **Split Discipline Integrity:** Confirmed via codebase grep that `test-queries.txt` and `test-qrels.txt` are never read or processed.
- [x] **Reproducibility Guarantee:** All intermediate runs (`evaluation/runs/*.txt`) and analysis CSVs (`analysis/*.csv`) can be regenerated via documented bash scripts in one execution pass.
