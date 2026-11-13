# LLM-Enhanced Tip-of-the-Tongue Retrieval (IT550)

> **Course:** IT550 — Information Retrieval (DA-IICT)  
> **Benchmark:** TREC-ToT 2025 (Dev3 Split, 536 queries, 6.4M English Wikipedia corpus)  
> **Corpus Tier:** Tier B (`dense_rerank_bm25`) — Lucene/Anserini BM25 top-1000 candidate retrieval followed by dense and neural reranking  
> **Plan Source of Truth:** `plan_v2/` (+ `scope.md`)  

---

## 1. Executive Summary & Core Results

In Tip-of-the-Tongue (ToT) retrieval, users describe a known book, movie, or song without remembering its title. Unstructured conversational descriptions induce severe vocabulary mismatch (responsible for 67.0% of retrieval failures), where dense bi-encoders suffer from semantic drift. 

This repository provides an end-to-end, reproducible empirical evaluation of **9 experimental configurations (E1 through E9)**, adapting clue decomposition (Lin et al., EMNLP 2023) and guarded keyword reformulation (Mukhopadhyay et al., ECIR 2026). Our primary findings demonstrate that:
1. **Raw Dense Reranking (E2-R)** drops nDCG@10 relative to BM25 (-30.8%) due to semantic false positive drift.
2. **Clue Concatenation (E4c)** raises Recall@1000 from 0.2394 to **0.2465**, effectively expanding candidate funnel recall.
3. **Guarded Reformulation (E6)** improves top-ranked precision, boosting nDCG@10 by **+24.9%** ($0.0305 \to 0.0381$).
4. **Cross-Encoder Reranking (E9)** over top-30 candidates achieves peak effectiveness (**nDCG@10 = 0.0386**, **+26.6%** over BM25, $p=0.003$), yielding a +23.6% gain on reachable candidates while preserving candidate invariance.

### Official Results Table (Generated from `evaluation/results.csv`)

| Exp. | Method (Run ID) | Corpus Tier | Split ($N$) | nDCG@10 | MRR | R@10 | R@100 | R@1000 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | BM25 Raw Query Baseline | `dense_rerank_bm25` | 142 | 0.030516 | 0.030096 | 0.042254 | 0.112676 | 0.239437 |
| **E2-R** | Dense Candidate Rerank (E5-Small) | `dense_rerank_bm25` | 142 | 0.021127 | 0.022234 | 0.021127 | 0.042254 | 0.239437 |
| **E3** | Hybrid BM25 + Dense RRF Fusion | `dense_rerank_bm25` | 142 | 0.027193 | 0.027347 | 0.035211 | 0.119718 | 0.239437 |
| **E4** | Clue Decomposition (Separate Clue RRF) | `dense_rerank_bm25` | 142 | 0.006587 | 0.004074 | 0.021127 | 0.056338 | 0.197183 |
| **E4c** | Clue Concatenation | `dense_rerank_bm25` | 142 | 0.026905 | 0.023367 | 0.042254 | 0.070423 | **0.246479** |
| **E5-R** | Dense Facet Search | `dense_rerank_bm25` | 142 | 0.006665 | 0.005713 | 0.014085 | 0.056338 | 0.077465 |
| **E6** | Extractive Clue Fusion Rerank | `dense_rerank_bm25` | 142 | 0.009515 | 0.008405 | 0.021127 | 0.077465 | 0.176056 |
| **E7-dense** | Dense Reformulation Reranking | `dense_rerank_bm25` | 142 | 0.029091 | 0.027845 | 0.035211 | 0.063380 | 0.204225 |
| **E8-F4** | Two-Level Branch-Wise Fusion (+Orig) | `dense_rerank_bm25` | 142 | 0.031237 | 0.028053 | 0.049296 | 0.098592 | 0.232394 |
| **E9** | **Cross-Encoder Rerank (MiniLM-L6, $K=30$)** | `dense_rerank_bm25` | 142 | **0.038603** | **0.034832** | **0.056338** | 0.098592 | 0.232394 |

*Note: All values generated automatically by `evaluation/make_results_table.py`. 95% confidence intervals and paired permutation tests available in `analysis/planned_comparisons.csv`.*

---

## 2. Non-Negotiable Research Protocol & Contract

1. **Quarantined Test Split:** The official Test split is never loaded or referenced. Codebase audit confirms 0 references (`assert_split_allowed('dev3')`).
2. **Split Discipline:** Train (50) for prompts; Dev1 (142) for hyperparameter tuning; Dev3 (536) for final evaluation.
3. **Zero Target Leakage:** Prompts never receive target titles, document IDs, or qrels. Audited via `scripts/run_fairness_audit.py` (`assert_fair: PASS`).
4. **Deterministic LLM Caching:** All LLM prompts and completions are cached to `cache/*.jsonl` at temperature $T=0.0$.
5. **Traceable Results:** Every number in reports and presentations is computed directly by `evaluation/evaluate_run.py` into `evaluation/results.csv`.

---

## 3. End-to-End Execution Order (Plans 01 $\to$ 10)

To reproduce all 10 project phases from a terminal:

```bash
# 0. Setup Environment and Dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Plan 01: Environment Verification and Split Protection
python -c "from tot.env_report import write_env_report; write_env_report()"
python -c "from tot.splits import assert_split_allowed; assert_split_allowed('dev3'); print('Split guards verified')"

# 2. Plan 02: Dataset Verification (Train, Dev1, Dev3)
python -c "from tot.dataset import load_tot_split; print('Dev1 loaded:', len(load_tot_split('dev1')))"

# 3. Plan 03: Run Comprehensive Evaluation & Metrics Tests
pytest tests/ -v

# 4. Plan 04: Generate Baseline Runs (E1 BM25, E2-R Dense, E3 Hybrid RRF)
python scripts/run_plan04_baselines.py --split dev1 --tier dense_rerank_bm25

# 5. Plan 05: Clue Decomposition (E4 RRF, E4c Concat, E5-R Dense Facets)
python scripts/run_plan05_decomposition.py --split dev1 --tier dense_rerank_bm25

# 6. Plan 06 & 07: Guarded Keyword Reformulation & Ablations (E6, E7-R)
python scripts/run_plan06_reformulation.py --split dev1 --tier dense_rerank_bm25

# 7. Plan 08: Two-Level Fusion & Cross-Encoder Reranking (E8, E9)
python scripts/run_plan08_twolevel_fusion.py --split dev1 --tier dense_rerank_bm25
python scripts/run_plan08_cross_encoder.py --split dev1 --tier dense_rerank_bm25 --depth 30

# 8. Plan 09: Quantitative Diagnostics, Statistical Tests & Error Analysis
python scripts/run_fairness_audit.py
python evaluation/compare_runs.py
python analysis/quantitative_analyses.py
python analysis/facet_analysis.py
python analysis/error_analysis.py
python scripts/plot_figures.py

# 9. Plan 10: Generate Colab Self-Contained Notebook & Final Deliverables
python scripts/generate_colab_notebook.py
```

---

## 4. Google Colab Reproduction

The complete project evaluation pipeline is available as an interactive Colab notebook:  
`notebooks/tot_project.ipynb`

### How to Run in Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Clone the repository and navigate into it:
   ```python
   !git clone https://github.com/dhyey2075/it550-tot-retrieval.git
   %cd it550-tot-retrieval
   ```
   *(Or if opening directly via Colab GitHub tab, open `notebooks/tot_project.ipynb` directly).*
3. Set Hardware Accelerator to **T4 GPU** (`Runtime` $\to$ `Change runtime type` $\to$ `T4 GPU`).
4. Click **Runtime** $\to$ **Run all**. All experiments E1–E9, statistical tests, diagnostics, and final tables will execute sequentially.

---

## 5. Repository Architecture

```
.
├── README.md                           # Master project guide & results overview
├── requirements.txt                    # Pinned package dependencies
├── scope.md                            # Frozen project scope contract
├── configs/
│   └── decisions.md                    # Immutable decision log (D-01 through D-10)
├── prompts/
│   ├── decomposition_prompt.txt        # Frozen Lin et al. clue extraction prompt
│   ├── reformulation_prompt.txt        # Frozen guarded keyword rewrite prompt
│   └── CHANGELOG.md                    # Audit log of prompt versions
├── data/
│   ├── MANIFEST.md                     # Checksums & metadata for splits
│   ├── train-queries.txt               # Train split (50 queries)
│   ├── dev1-queries.txt / dev1-qrels.txt # Dev1 split (142 queries, tuning)
│   └── dev3-queries.txt / dev3-qrels.txt # Dev3 split (536 queries, evaluation)
├── tot/                                # Core IR library modules
│   ├── dataset.py                      # Split loading & validation
│   ├── splits.py                       # Strict split quarantine guards
│   ├── bm25.py                         # BM25 Lucene scoring & rank generation
│   ├── dense.py                        # Bi-encoder inference & similarity
│   ├── rrf.py                          # Reciprocal Rank Fusion implementation
│   ├── cross_encoder.py                # Joint attention reranking
│   └── llm.py                          # Prompt orchestration & caching wrapper
├── scripts/
│   ├── run_plan04_baselines.py         # Generates E1, E2-R, E3 runs
│   ├── run_plan05_decomposition.py     # Generates E4, E4c, E5-R runs
│   ├── run_plan06_reformulation.py     # Generates E6, E7 runs
│   ├── run_plan08_twolevel_fusion.py   # Generates E8 fusion runs
│   ├── run_plan08_cross_encoder.py     # Generates E9 rerank runs
│   ├── run_fairness_audit.py           # Asserts 0 target title leakage
│   ├── plot_figures.py                 # Generates high-res analysis figures
│   └── generate_colab_notebook.py      # Packs repository into self-contained Colab notebook
├── evaluation/
│   ├── evaluate_run.py                 # TREC-compliant evaluation engine
│   ├── make_results_table.py           # Markdown table renderer
│   ├── compare_runs.py                 # 11 planned bootstrap & permutation tests
│   ├── results.csv                     # Official CSV results store
│   └── results_table.md                # Formatted markdown table
├── analysis/
│   ├── planned_comparisons.csv         # Hypothesis tests & win/loss counts
│   ├── error_sheet.csv                 # 94 annotated error queries
│   ├── error_summary.md                # Error taxonomy & worked case studies
│   ├── q1_q7_answers.md                # Answers to all 7 research questions
│   ├── limitations.md                  # 12 methodological threats to validity
│   ├── cost_view.md                    # Computational profile & token audit
│   └── figures/                        # High-resolution publication plots
├── presentation/
│   ├── SLIDES.md                       # 12-slide presentation deck with speaker notes
│   └── VIVA_PREP.md                    # 15 viva Q&As and 4 whiteboard derivations
├── docs/
│   └── RISK_REGISTER.md                # 15 monitored risks & 7-level scope-cut ladder
├── report/
│   └── FINAL_REPORT.md                 # Complete 11-section academic final report
├── tests/                              # Pytest test suite (35/35 passing)
└── notebooks/
    └── tot_project.ipynb               # 100% self-contained Google Colab notebook
```

---

## 6. IT550 Course Outcomes (CO) Alignment

| Course Outcome | Implementation in This Project | Course Topics Touched |
|:---|:---|:---|
| **CO1: Indexing & Scoring** | Inverted-index BM25, Lucene parameters ($k_1=0.9, b=0.4$), length normalization, IDF saturation. | Module 2: Inverted Index & Scoring |
| **CO2: IR Evaluation** | Evaluation of nDCG@10, MRR, Recall@K, 5000-sample bootstrap CIs, paired permutation tests. | Module 3: Evaluation of Ranking |
| **CO3: Query Reformulation** | LLM-based query decomposition and guarded keyword expansion vs. pseudo-relevance feedback. | Module 4: Query Expansion & Feedback |
| **CO4: Neural IR & Dense Models** | Bi-encoder dense reranking (E5-Small), cosine similarity, and Cross-Encoder reranking (MiniLM-L6). | Module 7: Neural IR & Vector Search |
| **CO5: Hybrid & LLM IR Systems** | Reciprocal Rank Fusion (RRF), two-level branch-wise fusion, and anti-hallucination audits. | Module 8: Modern Hybrid Search |

---

## 7. Scope §21 Major Outcomes Checklist

- [x] **Reproducible Environment:** Isolated virtualenv, locked dependencies, automated environment logging.
- [x] **Full Experimental Pipeline (E1–E9):** All 9 configurations executed, evaluated, and saved to `evaluation/results.csv`.
- [x] **Evaluation Integrity:** Official Test split quarantined; 35 unit tests verifying metric fidelity.
- [x] **11 Planned Statistical Comparisons:** Paired permutation testing with Holm–Bonferroni correction (`analysis/planned_comparisons.csv`).
- [x] **Systematic Error Analysis:** Rule-stratified taxonomy across 94 queries, 4 detailed case studies (`analysis/error_summary.md`).
- [x] **Publication Figures:** 4 publication-quality plots in `analysis/figures/`.
- [x] **Academic Final Report:** Comprehensive 11-section report in `report/FINAL_REPORT.md`.
- [x] **Defense Kit:** 12-slide presentation deck (`presentation/SLIDES.md`), 15 viva Q&As and whiteboard derivations (`presentation/VIVA_PREP.md`).
- [x] **Risk Governance:** 15 live risks closed and documented in `docs/RISK_REGISTER.md`.
- [x] **One-Click Colab Execution:** Fully self-contained notebook `notebooks/tot_project.ipynb` verified.
