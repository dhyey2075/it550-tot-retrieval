# 02 — Data & Corpus Strategy

| | |
|---|---|
| **Scope reference** | §3 Dataset, §15.2 Corpus Strategy |
| **Depends on** | 01 (environment, split guard, decision log) |
| **Produces** | verified Dev3/Dev1/Train files, `data_report.md`, **decision D-01 (corpus tier)** |
| **Suggested effort** | 2–3 work sessions (the corpus decision is the riskiest step in the project) |
| **Owner** | `[fill in]` |

---

## 1. Goal & Exit Criteria

Get the queries, qrels and a usable retrieval corpus in place, and decide **how retrieval will be run on Colab Free** without silently changing what the experiments mean.

Exit criteria:

- [ ] Dev3, Dev1 and Train queries and qrels loaded, checksummed, and inspected.
- [ ] Corpus structure understood (fields, size, length distribution).
- [ ] One of the corpus tiers in §4 chosen, tested end-to-end on 20 queries, and logged as D-01.
- [ ] `data_report.md` written (facts only, no results).

---

## 2. What the Data Is (as of the official documentation)

| Item | Detail |
|---|---|
| Corpus | 6,407,814 Wikipedia articles (2023 dump subset), one JSONL record per doc, doc ID = Wikipedia page ID |
| Corpus contains | the correct answer for every train, dev and test query |
| Dev3 | 536 queries + 536 qrels, derived from the 2024 TREC-ToT test set |
| Dev1 | 142 queries + qrels (derived from the 2023 dev set) |
| Train | 143 queries + qrels |
| Official access | `ir_datasets` ids such as `trec-tot/2025/dev3`, Hugging Face, or the Zenodo files |

Files the official page lists with checksums (verify against the Zenodo record you actually download from):

| File | Entries | md5 |
|---|--:|---|
| `dev3-2025-queries.jsonl` | 536 | `259c11645694a3c5230b66c7852d4d80` |
| `dev3-2025-qrel.txt` | 536 | `48ab0d24a5946861546e54064238477f` |
| `dev1-2025-queries.jsonl` | 142 | `b87c2f51d058de844e258a69b02e70fc` |
| `dev1-2025-qrel.txt` | 142 | `0c913ce8b5b287c73a6dfac662971e82` |

**Note on links:** your scope lists the Zenodo record `18235970`, while the `ir_datasets` download configuration points at record `15356599` for the 2025 files. Open both, confirm which one hosts Dev3 and the corpus, and write the winner into `data/MANIFEST.md`. Trust the md5 sums, not the record number.

Key reminder from scope §3.2: **Dev3 is a query/qrels split, not a smaller corpus.** Any Dev3 experiment still needs the large Wikipedia corpus, or a justified substitute (§4).

---

## 3. Data Tasks

### 3.1 Acquire and verify
- [ ] Download Dev3, Dev1, Train queries and qrels (small files, a few hundred KB).
- [ ] Compute md5; compare to the table above; record in `data/MANIFEST.md`.
- [ ] Do **not** download Test queries. (Test qrels are not public anyway, but the guard in plan 01 also blocks the split.)

### 3.2 Inspect queries (write findings to `data_report.md`)
- [ ] Field names in the query JSONL (the id field, the query text, and any metadata such as source or domain). Print three raw records.
- [ ] Query length distribution in characters and tokens (mean, median, max, percentiles). ToT queries are long, which matters for BM25 query handling in plan 04.
- [ ] Count duplicate query IDs and empty queries (expect none; confirm).
- [ ] Check whether any domain label (movie / landmark / celebrity) is present. If not, record that and see plan 03 for how to derive one.

### 3.3 Inspect qrels
- [ ] Format check: `qid 0 docid rel`. Confirm **exactly one relevant doc per query**, and which relevance value is used.
- [ ] Confirm every qrels doc ID exists in the corpus (use the offsets file or a streaming scan).
- [ ] This single-relevant-document structure is what makes Success@K equal Recall@K later (plan 03).

### 3.4 Inspect the corpus
- [ ] Print the first 3 records; list fields (the official description names `doc_id` and `url`; confirm title/text/other fields yourself).
- [ ] Sample 5,000 docs (random seeds fixed) and measure document length in characters and tokens.
- [ ] Decide the **document representation** for all retrievers: for example `title + first N characters`. Wikipedia pages are long; the lead section usually carries the identifying facts. Log the choice (it feeds D-02/D-03).
- [ ] Use the offsets file for random access to individual docs instead of re-scanning the whole gzip.

---

## 4. Corpus Strategy: Decision Tree

Scope §15.2 says: do not blindly build a full 6.4M-document index; inspect official resources first; and if full dense retrieval is impossible, say so and label it differently.

### Step A — Look for ready-made resources first (1 session)

Check, in this order:
1. The official benchmark repository (`TREC-ToT/bench`) and the track page for prebuilt indices, baseline runs, or notebooks.
2. PyTerrier / Anserini prebuilt-index registries and Hugging Face datasets for this corpus.
3. Published baseline runs for Dev3 (the track coordinators provided BM25 and dense baselines).

Record what exists, its size, and whether it fits in Colab disk and Drive.

### Step B — Compute the cost of each full-corpus option

Do the arithmetic before attempting anything:

| Item | Estimate |
|---|---|
| Dense vectors, 384-dim, fp16 | 6,407,814 × 384 × 2 B ≈ **4.9 GB** |
| Dense vectors, 384-dim, fp32 | ≈ **9.8 GB** |
| Dense vectors, 768-dim, fp16 | ≈ **9.8 GB** |
| Lucene BM25 index | unknown until measured; build a 100k-doc sample index and extrapolate ×64 |
| Dense encoding time | time 10,000 docs on the actual GPU, then extrapolate ×640 |

Compare with: Colab session length, free disk, system RAM, and the 15 GB Drive quota. A resumable, sharded encode that survives disconnects is mandatory for any full-corpus dense attempt.

### Step C — Pick a tier (this is D-01)

| Tier | Description | Valid label in the report | When to choose |
|---|---|---|---|
| **A** | Full-corpus BM25 **and** full-corpus dense (maybe quantized/IVF/PQ) | "full-corpus" | Only if Step B shows it is realistic and resumable |
| **B** | Full-corpus BM25 (or a prebuilt one) → candidates → **dense reranking** of those candidates | "dense reranking of BM25 candidates" | BM25 over the full corpus is feasible, dense indexing is not. This mirrors how a published ToT pipeline used BM25 first and then dense models as rerankers |
| **C** | Reduced **sub-corpus** shared by all experiments | "reduced-corpus setting; not comparable to official numbers" | Neither full index is feasible |

If you end up in Tier B, then E2 and E5 are **not** dense *retrieval*. Name them `E2-R`, `E5-R`, and say so in every table caption.

### Tier C: how to build a sub-corpus without cheating

A badly built sub-corpus makes everything look good. Rules:

1. **Never build it from qrels targets alone.** A corpus that contains little beyond the answers makes retrieval artificially easy and inflates recall.
2. If you include targets (they must exist in the corpus), also include a large and *hard* set of distractors:
   - the union of BM25 top-N documents from **every** query variant (original, clues, reformulations) for all Dev3 queries, so the pool is not tuned to one experiment; plus
   - documents from the same domains/categories (if domain information is available); plus
   - a random background sample.
3. Choose the pool **once**, before running E1–E9, and share it across all experiments.
4. State the sub-corpus size, how it was built, and that BM25 statistics (IDF) now differ from the full corpus.
5. Report that absolute scores are **not comparable** with published full-corpus numbers; only comparisons *within* your study are valid.
6. If pooling uses BM25 outputs, state that this favors BM25-style recall and may disadvantage dense-only methods. Mention it in threats to validity.

### Gate test (all tiers)

Run 20 Dev1 queries end-to-end with the chosen tier: retrieve → fuse → evaluate. Measure seconds/query and peak RAM. Extrapolate to 536 queries × (1 original + ~4 clues + 1 reformulation) × 2 retrievers. If the projected time exceeds what one or two Colab sessions can do with checkpointing, step down one tier.

---

## 5. Split Discipline

| Split | Allowed use | Forbidden use |
|---|---|---|
| Train | prompt development, few-shot examples, dataset understanding | final numbers |
| Dev1 | pilot runs, BM25/RRF/cross-encoder hyperparameters, pipeline debugging | final numbers |
| Dev3 | **final evaluation only**, error analysis after freezing | choosing prompts or parameters |
| Test | nothing | everything |

Practical consequences:
- Few-shot examples in prompts must come from **Train** (never Dev3).
- Dev1 is small (142) and noisy; treat tuning conclusions there as modest. Keep grids small to avoid overfitting Dev1 itself.
- Error analysis on Dev3 is fine, but if it leads you to change something, record the change as post-hoc and rerun only as a clearly labeled extra experiment.

---

## 6. Deliverables

1. `data/MANIFEST.md`: source URLs, record IDs, md5 sums, download dates.
2. `data_report.md`: query/qrels/corpus statistics, representation choice, anomalies.
3. `configs/decisions.md` row **D-01** with the tier, the Step B arithmetic, and the gate-test timings.
4. A reusable `corpus.py` with: streaming reader, random access by doc ID, and the chosen document representation.
5. If Tier C: `pool_construction.md` describing exactly how the sub-corpus was built, plus the script.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Full corpus does not fit or finishes too slowly | Decide tier from measurements (Step B) in week 1, not week 4 |
| Index exceeds Drive's 15 GB | Keep indices on the Colab local disk, rebuild from shards, or keep only compressed shards on Drive |
| Silent leakage via a target-heavy sub-corpus | Follow Tier C rules; have a teammate review the pool script |
| Misreading qrels (multiple levels, wrong IDs) | Assert one relevant doc per query and that all IDs exist in the corpus |
| Link/record mismatch between sources | Verify by md5, record in MANIFEST |

## 8. Definition of Done

- [ ] You can state, in one sentence, exactly which retrieval setting each of E1–E9 will use (e.g. "full-corpus BM25, then dense reranking of the top-1000 BM25 candidates").
- [ ] A 20-query end-to-end dry run succeeded in the chosen tier.
- [ ] No Test data was touched.
