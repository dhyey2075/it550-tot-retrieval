# 01 — Master Plan, Environment & Reproducibility

**Project:** LLM-Enhanced Tip-of-the-Tongue (ToT) Retrieval — Query Decomposition, Reformulation, Hybrid Ranking
**Course:** IT550 Information Retrieval · **Benchmark:** TREC-ToT 2025 Dev3 · **Env:** Colab Free + Python + LLM API

| | |
|---|---|
| **Scope reference** | §15 Implementation Plan, §16 LLM Usage, §23 Reproducibility |
| **Depends on** | nothing (start here) |
| **Produces** | working Colab/Drive skeleton, config system, run-naming convention, decision log |
| **Suggested effort** | 1–2 work sessions |
| **Owner** | `[fill in]` |

---

## 1. The Ten Plan Files at a Glance

| # | File | Covers | Scope § | Experiments |
|--:|---|---|---|---|
| 01 | `01_master_plan_environment_reproducibility.md` | Roadmap, environment, repo, config, caching rules | 15, 16, 23 | — |
| 02 | `02_data_and_corpus_strategy.md` | Dev3 data, corpus feasibility, candidate-pool design | 3, 15.2 | — |
| 03 | `03_evaluation_framework.md` | Metrics, run format, significance tests, sanity tests | 13, 14 | — |
| 04 | `04_baselines_bm25_dense_hybrid.md` | BM25 (tuned for recall), dense, RRF hybrid on the original query | 6, 11 | E1–E3 |
| 05 | `05_llm_query_decomposition.md` | Faceted clue prompts (extractive primary, predictive extension), per-facet retrieval, fusion variants | 7 | E4–E6 |
| 06 | `06_llm_query_reformulation.md` | Keyword vs. descriptive rewrite, stability and leakage audits, PRF baseline | 8 | E7 |
| 07 | `07_combined_pipeline_fusion.md` | Decomposition + reformulation fusion, complementarity | 9 | E8 |
| 08 | `08_cross_encoder_reranking.md` | Cross-encoder over top 50–100 | 10 | E9 |
| 09 | `09_results_and_error_analysis.md` | Results table, significance, failure taxonomy, Q1–Q7 | 18, 19, 20 | all |
| 10 | `10_report_submission_and_risk_register.md` | Report, viva prep, risks, scope-cut ladder | 21, 22, 27 | all |

### Dependency graph

```text
01 ──► 02 ──► 03 ──► 04 (E1–E3) ──┬──► 05 (E4–E6) ──┐
                                   │                 ├──► 07 (E8) ──► 08 (E9) ──► 09 ──► 10
                                   └──► 06 (E7) ─────┘
```

05 and 06 are independent of each other once 04 works, so two people (or two Colab sessions) can run them in parallel.

### Mapping to the scope's six phases

| Scope phase | Plan files |
|---|---|
| Phase 1 — Dataset and baseline | 02, 03, 04 |
| Phase 2 — Query decomposition | 05 |
| Phase 3 — Query reformulation | 06 |
| Phase 4 — Combined approach | 07 |
| Phase 5 — Reranking | 08 |
| Phase 6 — Analysis | 09, 10 |

---

## 2. Suggested Schedule Template

You did not state a deadline, so this is expressed in relative weeks. Compress or stretch it to fit; keep the order.

| Week | Focus | Gate to pass before moving on |
|---|---|---|
| 1 | Plans 01–03: environment, data, corpus decision, metrics | Corpus tier chosen and logged (D-01); metrics unit tests pass |
| 2 | Plan 04: E1–E3 | E1–E3 numbers on Dev1 look sane and reproducible |
| 3 | Plans 05–06: prompts frozen on Train/Dev1, caches generated for Dev3 | Prompts frozen (git tag), all 536 queries cached |
| 4 | Plans 05–07: E4–E8 on Dev3 | All runs saved, evaluated by one script |
| 5 | Plan 08: E9, plus ablations | E9 evaluated; K and model frozen on Dev1 first |
| 6 | Plans 09–10: analysis, report, slides, cleanup | Report numbers match `results.csv` exactly |

Keep roughly 20% slack. Colab disconnects and API rate limits will eat it.

---

## 3. The Project "Contract" (rules every plan file inherits)

These come straight from scope §3.2, §14, §16 and §22. Print them at the top of the notebook.

1. **Test split is never loaded.** Enforce it in code (see §6 below), not by promise.
2. **Split roles:** Train → prompt development · Dev1 → pilots and hyperparameter choices · Dev3 → final evaluation only · Test → untouched. **Optional amendment A1 (needs your approval):** also allow Dev2 (movie-only) for tuning. The 2026 reformulation paper tuned sequentially on Train → Dev1 → Dev2 and treated Dev3 as a pseudo-test set; a second development split makes the tuning less noisy. Dev1, Dev2 and Train are movie-only and much harder for BM25 than Dev3 (that paper reports BM25 nDCG@10 of 0.083 on Dev1 vs. 0.337 on Dev3), so Dev1 numbers are **not** a preview of Dev3 numbers.
3. **No parameter, prompt, or fusion choice is tuned on Dev3.** Anything chosen after seeing Dev3 numbers must be labeled "post-hoc" in the report.
4. **LLM never sees qrels or target titles** and is never asked to name the answer.
5. **LLM outputs are cached** and the Dev3 final run reads only from cache.
6. **One evaluation script** for every experiment (plan 03).
7. **No fabricated or "expected" numbers** anywhere. Unrun cells stay `TBD`.
8. **Full-corpus vs candidate-reranking runs are labeled differently** and never merged into one row (scope §15.2).

---

## 3a. Paper-to-Plan Traceability (checked against the full text of both papers)

| # | Finding | Source | Consequence for the plan | Plan |
|---|---|---|---|---|
| L1 | Decomposition is **field-based**: one sub-query per metadata field, "N/A" when absent | Lin et al. §2.1 | Facet schema with N/A instead of free-form "2–5 clues" | 05 |
| L2 | **Extractive** (grounded) vs. **predictive** (may hallucinate) prompting; predictive was better (R@5 32.1 vs 26.6) | Lin §2.1, Table 4 | Scope §7 forbids unsupported facts, so extractive is primary; predictive is a labeled extension | 05 |
| L3 | Sub-query results are combined **together with a base retriever on the original query** | Lin §1 | Run with and without the original list; E8 variant includes it | 05, 07 |
| L4 | Combination is a weighted score sum tuned on validation | Lin §2.4 | Scope uses RRF; weighted RRF is an ablation | 05, 07 |
| L5 | A single rewritten query was worse than the original for a dense retriever | Lin Table 4 | Evaluate E7 per retriever | 06 |
| L6 | Plot clue strongest and most common; dates and genre weak | Lin Table 6 | Per-facet effectiveness table | 05, 09 |
| L7 | Closed corpus of gold + negatives; performance drops as negatives are added | Lin §3, Fig. 2 | Corpus-size sensitivity study if a sub-corpus is used | 02, 09 |
| L8 | ToT queries are long with low lexical overlap with the target (0.19–0.25 vs. 0.4–0.58 elsewhere) | Lin Table 1 | Length and overlap diagnostics | 02, 09 |
| L9 | Failure analysis: 54% query–document relationship, 28% document representation, 18% query errors | Lin Table 7 | Map taxonomy; paper-comparable sample | 09 |
| M1 | Rewrite prompt is keyword-focused, strictly grounded; Mistral-7B best of three 7–8B LLMs | Mukhopadhyay §3.2, Table 6 | Two rewrite styles; LLM is a variable | 06 |
| M2 | RM3, Bo1, KL feedback all underperformed BM25 on original queries | M §4.1.1 | `E-PRF` baseline | 06 |
| M3 | Rewrites raised Dev3 BM25 Recall@1000 from 0.7705 to 0.8284 (Mistral) | M Table 6 | Published reference points for sanity checks only | 03 |
| M4 | Dense reranking of BM25 top-1000 cut nDCG@10 by 18% but raised Success@100 | M Table 7 | Expect recall up, precision down; Tier-B precedent | 04, 09 |
| M5 | BM25 tuned for Recall@1000, coarse grid then local grid across splits, robust score μ − 0.5σ | M §3.3–3.4 | Tuning protocol (D-10) | 01, 04 |
| M6 | E5-large-v2 was the strongest dense model; Contriever and ColBERTv2 weaker | M §5.2.2 | E5 in the bake-off | 04 |
| M7 | monoT5 gave the largest single-stage gain; best depth K = 50; union of BM25 and dense top-100 as pool | M §3.5, §4.3 | Depth sweep; pool variant | 08 |
| M8 | Rewrite reruns varied even with greedy decoding (CV 3.67%) | M §4.1.3 | Stability check | 05, 06 |
| M9 | Query-type, vocabulary and indirect-reference analyses | M §4.1.4–4.1.6 | Audits | 06, 09 |
| M10 | Their pipeline used a 500 GB RAM machine and two RTX 6000 GPUs | M §3.3, §3.5 | Colab feasibility is a real risk | 02 |

---

## 4. Colab Free Environment Plan

### 4.1 Known constraints to design around

- Sessions disconnect (idle timeout and a maximum runtime), and local disk is wiped each time. Treat every session as disposable.
- Google Drive free storage is 15 GB total. A full-corpus Lucene index or large embedding matrix may not fit on Drive permanently (see plan 02).
- A T4-class GPU is typical but not guaranteed; RAM is limited. Write code that reads data in streams or shards.
- Check current Colab and LLM-API free-tier limits when you start; they change.

### 4.2 Setup checklist

- [ ] Create a Drive folder `tot_project/` and mount it at the top of the notebook.
- [ ] Store API keys in Colab **Secrets** (never in the notebook or on GitHub).
- [ ] Create `requirements.txt` with **pinned** versions after the first working run (`pip freeze` subset, not everything).
- [ ] Decide the Java requirement early: Pyserini/Anserini and PyTerrier need a JVM. Test installing it in Colab in the first session.
- [ ] Create a private GitHub repo; push code and prompts (not data, not large indices).
- [ ] Add a first cell that prints GPU, RAM, free disk, Python and library versions, and writes them to `logs/env_<date>.txt`.

### 4.3 Candidate libraries (from scope §15.1)

| Need | Candidates | Note |
|---|---|---|
| Dataset loading | `ir_datasets` (`trec-tot/2025/...`) | Also loadable from the Zenodo files directly |
| BM25 | Pyserini/Anserini, PyTerrier, or `bm25s` for a sub-corpus | Decide in plan 02/04 |
| Dense encoder | `sentence-transformers` | Model choice in plan 04 |
| ANN/vector search | `faiss` (CPU or GPU) | Matches course Module 7 |
| Cross-encoder | `sentence-transformers` CrossEncoder, or `transformers` | Plan 08 |
| Evaluation | `ranx`, cross-checked with `trec_eval`/`pytrec_eval` | Plan 03 |
| LLM | Gemini or OpenAI SDK | Behind one wrapper (below) |

---

## 5. Repository & Drive Layout

Based on scope §23, extended with the folders the later plans need.

```text
tot_project/
├── README.md                      # how to reproduce, in order
├── requirements.txt
├── configs/
│   ├── base.yaml                  # paths, seeds, depths, model names
│   └── decisions.md               # the decision log (section 7)
├── data/
│   ├── dev3-queries.jsonl
│   ├── dev3-qrels.txt
│   ├── dev1-queries.jsonl         # pilot
│   ├── train-queries.jsonl        # prompt development
│   └── MANIFEST.md                # source URLs + md5 sums
├── prompts/
│   ├── decomposition_prompt.txt
│   ├── reformulation_prompt.txt
│   └── CHANGELOG.md               # every prompt version, why it changed
├── cache/
│   ├── decompositions.jsonl
│   ├── reformulations.jsonl
│   └── ce_scores/                 # cross-encoder score cache (plan 08)
├── retrieval/
│   ├── bm25/   ├── dense/   └── hybrid/
├── reranking/
├── runs/                          # TREC-format run files, one per experiment
├── evaluation/
│   ├── metrics.py
│   ├── tests/                     # metric unit tests (plan 03)
│   ├── results.csv                # aggregate
│   └── per_query/                 # one CSV per run
├── analysis/                      # error-analysis sheets, figures
├── logs/
└── notebooks/
    └── tot_project.ipynb          # thin driver; real logic lives in .py modules
```

**Rule:** the notebook calls functions; it does not contain the only copy of any logic. Put code in `.py` modules under the repo so it can be diffed, tested and reused after a session reset.

---

## 6. Configuration, Seeds, Guards

### 6.1 `configs/base.yaml` (minimum keys)

```yaml
seed: 42
splits: {train: train, pilot: dev1, final: dev3}   # 'test' deliberately absent
depths: {retrieve: 1000, fuse_out: 1000, rerank_k: 100}
tuning: {splits: [train, dev1], robust_alpha: 0.5}   # add dev2 only if amendment A1 is approved
rrf: {k: 60}
bm25: {k1: null, b: null}        # fixed after Dev1 tuning (plan 04)
dense: {model: null, max_doc_tokens: null, prefix_query: null, prefix_doc: null}
llm: {provider: null, model: null, temperature: 0, max_output_tokens: null}
cross_encoder: {model: null, max_len: null, batch_size: null}
```

Fill the `null`s as decisions are made and logged. A run is only valid if its config snapshot is saved next to its run file.

### 6.2 Split guard (enforces rule 1)

```python
ALLOWED = {"train", "dev1", "dev3"}   # add "dev2" only if amendment A1 is approved
def load_split(name):
    if name not in ALLOWED:
        raise ValueError(f"Split '{name}' is not allowed in this project")
    ...
```

### 6.3 Seeds

Set Python, NumPy and PyTorch seeds at notebook start. BM25 is deterministic; FAISS flat search is deterministic; IVF/HNSW training is not unless seeded, so record index parameters.

### 6.4 LLM wrapper contract (shared by plans 05 and 06)

- One function `llm_call(task, query, prompt_version) -> dict`.
- **Cache key** = hash of `(provider, model_id, prompt_version, prompt_text_hash, temperature, query_text)`.
- Cache stored as JSONL: `{key, qid, task, model, prompt_version, raw_output, parsed, timestamp}`.
- Retries with exponential backoff; log failures; never silently drop a query.
- Temperature 0 (or lowest allowed). Record the exact model identifier string, since providers update models over time.
- Write to the cache after **every** call, so a disconnect loses nothing.
- A `--dry_run` mode that counts uncached queries and estimates calls before spending anything.

### 6.5 Run naming & format

- Run files in standard TREC format: `qid Q0 docid rank score run_name`.
- Run name pattern: `E{n}_{split}_{corpusTier}_{promptVer}_{YYYYMMDD}`, e.g. `E6_dev3_tierA_p3_20260301`.
- Every run gets a sidecar `*.json` with the config snapshot, git commit hash, wall time, and counts (queries, docs per query).

---

## 7. Decision Log

Keep `configs/decisions.md` as a table. Every plan file lists decisions to add. Never edit a past decision silently; add a new row that supersedes it.

| ID | Date | Decision | Options considered | Evidence (split used) | Frozen? |
|---|---|---|---|---|---|
| D-01 | | Corpus tier (plan 02) | | | |
| D-02 | | BM25 parameters, tuned for Recall@1000 (plan 04) | | Train/Dev1 | |
| D-03 | | Dense model (E5 family included in bake-off) + doc truncation (plan 04) | | Train/Dev1 | |
| D-04 | | RRF k and depth | | Dev1 | |
| D-05 | | LLM provider/model (plan 05–06) | | Train | |
| D-06 | | Decomposition: facet schema, extractive vs. predictive, original-list rule, E6 fusion (plan 05) | | Train/Dev1 | |
| D-07 | | Reformulation: keyword (R-K) vs. descriptive (R-N) style (plan 06) | | Train/Dev1 | |
| D-08 | | E8 fusion protocol (plan 07) | | Dev1 | |
| D-09 | | Cross-encoder model, K, query side (plan 08) | | Dev1 | |
| D-10 | | Dev-split tuning protocol: robust aggregate μ − 0.5σ across allowed dev splits; Dev2 amendment A1 yes/no | | — | |

---

## 8. Deliverables & Definition of Done

- [ ] Notebook opens on a fresh Colab runtime and reaches "environment OK" in one run-all.
- [ ] Repo contains layout above, with README steps in order.
- [ ] `base.yaml`, split guard, LLM wrapper with caching, and run-naming helper exist and have been used once.
- [ ] Decision log created with D-01…D-09 rows (empty is fine).
- [ ] Printed contract (section 3) is the first markdown cell of the notebook.

## 9. Risks for This Plan

| Risk | Mitigation |
|---|---|
| Session loss destroys progress | Cache and checkpoint to Drive after every unit of work |
| JVM / Pyserini install fails on Colab | Test in session 1; fall back to `bm25s` or PyTerrier-free options (plan 02) |
| API key leaked | Colab Secrets only; add `.gitignore` for keys and data |
| Notebook becomes unmaintainable | Logic in `.py` modules, notebook as a driver |
