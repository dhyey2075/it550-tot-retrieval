# 04 — Baselines: BM25, Dense, Hybrid RRF (E1–E3)

| | |
|---|---|
| **Scope reference** | §6 Proposed Retrieval Pipeline, §11 (E1–E3), §17 Phase 1 |
| **Depends on** | 01, 02 (corpus tier D-01), 03 (evaluator) |
| **Produces** | reusable `bm25_search()`, `dense_search()`, `rrf_fuse()`; runs E1, E2, E3; decisions D-02, D-03, D-04 |
| **Suggested effort** | 3–5 work sessions (indexing time dominates) |
| **Course link** | Module 1 (inverted index), Module 2 (BM25, ranked retrieval), Module 7 (dense retrieval, FAISS, ANN), Module 8 (hybrid retrieval) |
| **Owner** | `[fill in]` |

Everything in plans 05–08 reuses the three functions built here, so build them as clean, tested modules.

---

## 1. Goal & Exit Criteria

- [ ] E1 (Original + BM25), E2 (Original + Dense), E3 (Original + Hybrid) evaluated on Dev1 for development and on Dev3 for the final table.
- [ ] Retrieval functions accept **any query string** and return top-K `(docid, score)`, so the same code serves original queries, clues, and reformulations.
- [ ] BM25 parameters, dense model, and RRF `k` chosen on **Dev1 only** and frozen (D-02, D-03, D-04).
- [ ] Results saved as TREC run files and evaluated by plan 03's script.

---

## 2. E1 — Original Query + BM25

### 2.1 Engine choice (depends on D-01)

| Tier | Practical option |
|---|---|
| A or B (full corpus) | Pyserini/Anserini or PyTerrier index of the full corpus, or an official prebuilt index if Step A of plan 02 found one |
| C (sub-corpus) | the same engines, or an in-memory BM25 library on a small corpus |

Avoid pure-Python BM25 libraries that rescore every document per query for large corpora; they will not scale.

### 2.2 Tasks
- [ ] Index `title + body` using the document representation chosen in plan 02. Keep the doc ID field as the Wikipedia page ID string, exactly as in the qrels.
- [ ] Decide analyzer settings (lowercasing, stemming, stopwords) once; they apply to every query variant later. Log them.
- [ ] **Long-query handling:** ToT queries are verbose. Some engines cap the number of query terms/clauses. Detect whether truncation or an error occurs, and handle it identically for the original query, the clues, and the reformulation. Log the maximum query length actually handled.
- [ ] Parameter choice (D-02): evaluate a **small** grid of `(k1, b)` on Dev1 (for example the engine defaults plus 3–4 alternatives). Choose by nDCG@10 and MRR, then freeze. A bigger grid just overfits 142 queries.
- [ ] Retrieve top-1000 per query; write run file; evaluate.

### 2.3 Sanity checks
- A BM25 run should clearly beat the random baseline (plan 03, T7). If not, suspect ID mismatches or an indexing bug.
- Pick 5 Dev1 queries; manually read the top 3 results and confirm they look lexically plausible.
- Confirm every returned `docid` is in the qrels ID format.

---

## 3. E2 — Original Query + Dense Retrieval

### 3.1 Model selection (D-03)

Choose from a short list that fits a Colab T4 and is available through `sentence-transformers`. Practical picks by size:

| Class | Examples | Trade-off |
|---|---|---|
| Small | MiniLM-class, BGE-small, E5-small | fastest to encode, smallest vectors |
| Base | BGE-base, E5-base | usually better quality, 768-dim doubles storage |

Run a **mini bake-off on Dev1** (see §3.4) with at most 2–3 models, then freeze one. Do not tune the model on Dev3.

**Prefix requirements matter.** Some model families expect specific prefixes (for example E5 expects `query: ` / `passage: `, and BGE recommends an instruction prefix for short queries). Wrong prefixes silently degrade results. Read each model card and store the prefixes in `base.yaml`.

### 3.2 Document encoding plan
- [ ] Represent each doc as `title + lead text`, truncated to `max_doc_tokens` (start with the model's practical limit, commonly 256–512 tokens). Longer is slower and may not help.
- [ ] Encode in **shards** (for example 100k docs per shard) with fp16 on GPU, saving each shard to disk/Drive immediately. Skip shards already present on restart.
- [ ] L2-normalize embeddings if the model expects cosine/inner product.
- [ ] Log docs/second so you can predict the remaining time.

### 3.3 Search side
- [ ] Choose a FAISS index appropriate to the memory you computed in plan 02:
  - exact **flat inner-product** index if vectors fit in RAM;
  - **IVF** and/or **PQ** (or scalar quantization) if they do not; record `nlist`, `nprobe`, and the recall loss versus exact search on a 1,000-doc check;
  - HNSW is an option but its build is memory-hungry.
- [ ] Encode queries with the query prefix; search top-1000.
- [ ] If D-01 is Tier B: dense scores are computed **only for BM25's top-1000 candidates**. Name this run `E2-R` and put the tier in the table.

### 3.4 Mini bake-off protocol (Dev1)
1. Same Dev1 queries, same doc representation, same truncation.
2. Compare 2–3 models on nDCG@10, MRR, R@100.
3. Prefer the smaller model if the difference is within noise (use plan 03's bootstrap).
4. Record the winner and the numbers in D-03.

---

## 4. E3 — Original Query + Hybrid (BM25 + Dense via RRF)

### 4.1 RRF definition (write exactly this in the report)

For each document `d` appearing in any input list:

```text
RRF(d) = Σ over lists L containing d of   1 / (k + rank_L(d))        rank starts at 1
```

Documents are sorted by `RRF(d)` descending; ties are broken deterministically (for example by smaller best rank, then by docid). `k = 60` is a common default. Check on Dev1 whether a small set such as `{20, 60, 100}` changes anything materially; if not, keep 60 and say so (D-04).

### 4.2 Implementation notes
- [ ] Input lists are truncated to the same depth (1000). A doc missing from one list contributes nothing from that list.
- [ ] Write `rrf_fuse(list_of_ranked_lists, k, out_depth)`. It will be reused for **per-clue fusion** (plan 05) and **combined fusion** (plan 07), so it must accept an arbitrary number of lists.
- [ ] Unit tests: two lists with a known answer; a doc present in only one list; identical lists; empty list.
- [ ] The scope's hybrid uses unweighted RRF. Weighted variants are optional ablations on Dev1 only.

---

## 5. Run Procedure

```text
1. Develop on Dev1       → fix k1/b, dense model, truncation, RRF k          (log D-02..D-04)
2. Freeze configs        → git tag baselines-v1
3. Run E1, E2, E3 once   → on Dev3, save runs + config snapshots
4. Evaluate with plan 03 → append rows to results.csv
```

Baselines do not use the LLM, so unlike plans 05–06 there is no prompt to freeze, but the same discipline applies: do not retune after seeing Dev3.

---

## 6. What E1–E3 Should Let You Answer (scope Q1–Q3)

| Question | Evidence to collect now |
|---|---|
| Does BM25 work for ToT? | E1 metrics, rank-of-target distribution |
| Does dense retrieval fix vocabulary mismatch? | E2 vs E1 win/tie/loss table; queries where only E2 finds the target in top 100 |
| Is hybrid more robust? | E3 vs max(E1, E2); count queries where hybrid rescues a miss, and where it hurts |

Save the per-query CSVs; plan 09 will mine them. Do not write conclusions yet.

---

## 7. Deliverables

1. `retrieval/bm25/`, `retrieval/dense/`, `retrieval/hybrid/` modules with unit tests.
2. Index artifacts (or a script that rebuilds them) plus an index manifest: engine, version, analyzer, parameters, doc count.
3. Run files and config snapshots for E1–E3 (Dev1 and Dev3).
4. Decision log rows D-02, D-03, D-04 filled.
5. A short `baselines_notes.md`: anything surprising (truncation behavior, ID quirks, timing).

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Docid format mismatch (int vs string, leading zeros) | One normalization function; assert against qrels |
| Dense encoding never finishes | Sharded, resumable encode; time 10k docs first; step down tier if needed |
| Wrong model prefixes | Read model cards; test with and without prefix on a few Dev1 queries |
| FAISS quantization hides real quality | Check approximate vs exact recall on a small slice |
| Overfitting BM25/RRF settings to Dev1 | Tiny grids; prefer defaults when differences are within noise |
| Label confusion between dense retrieval and dense reranking | Tier in run name, table caption, and report |

## 9. Definition of Done

- [ ] E1–E3 rows exist for Dev3 in `results.csv` with the correct tier label.
- [ ] `rrf_fuse` is unit-tested and accepts N lists.
- [ ] Anyone can rerun E1–E3 from the repo and obtain identical run files (modulo documented nondeterminism in ANN training).
