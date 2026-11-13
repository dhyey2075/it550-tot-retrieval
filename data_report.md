# Data Report (facts only)

Date: 2026-11-08  
Sources: Zenodo record `18235970` downloads under `data/` (see `data/MANIFEST.md`); field descriptions cross-checked with https://trec-tot.github.io/guidelines.html  
No retrieval scores. Corpus file not downloaded yet.

---

## 1. Query files

### 1.1 Schema

Every Train / Dev1 / Dev3 query record has **exactly two fields**:

| Field | Type | Role |
|---|---|---|
| `query_id` | string (numeric characters) | Query identifier (matches qrels column 1) |
| `query` | string | Full ToT request text (may contain newlines) |

No domain label (`movie` / `landmark` / `celebrity`) is present in the query JSONL. Domain labels, if needed later, must be derived at analysis time (plan 03), not used as system input.

### 1.2 Counts and integrity

| Split | Queries | Duplicate `query_id` | Empty `query` | Overlap with other downloaded splits |
|---|--:|--:|--:|---|
| Train | 143 | 0 | 0 | none with Dev1/Dev3 |
| Dev1 | 142 | 0 | 0 | none with Train/Dev3 |
| Dev3 | 536 | 0 | 0 | none with Train/Dev1 |

### 1.3 Three raw Dev3 records (truncated for readability)

```json
{"query_id": "2001", "query": "I remember this old building I used to pass by in the heart of a bustling financial district..."}
{"query_id": "2002", "query": "<1624 characters>"}
{"query_id": "2003", "query": "<1855 characters>"}
```

Train sample `query_id` values observed: `763`, `802`, `950`.  
Dev1 sample `query_id` values observed: `152`, `531`, `473`.

### 1.4 Length distribution

Lengths measured on the raw `query` string. **Tokens = whitespace-separated pieces** (not a model tokenizer).

| Split | Unit | mean | median | min | max | p25 | p75 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | chars | 773.3 | 657 | 132 | 2386 | 515 | 966 | 1497 |
| Train | ws-tokens | 147.5 | 127 | 26 | 437 | 101 | 186 | 271 |
| Dev1 | chars | 735.1 | 649.5 | 75 | 2851 | 442 | 886 | 1552 |
| Dev1 | ws-tokens | 140.7 | 121 | 10 | 553 | 86 | 171 | 290 |
| Dev3 | chars | 1308.1 | 1413.5 | 121 | 3540 | 1212 | 1555 | 1823 |
| Dev3 | ws-tokens | 237.6 | 252.5 | 23 | 703 | 222 | 279 | 329 |

Observation (fact): Dev3 queries are substantially longer than Train/Dev1 on both characters and whitespace tokens.

### 1.5 Published difficulty note (not our results)

Official track baselines and the 2026 reformulation paper report much lower BM25 nDCG@10 on Dev1 than on Dev3 under full-corpus settings (paper cites ≈0.083 on Dev1 vs ≈0.337 on Dev3; track table lists PyTerrier BM25 Dev1 nDCG@10 = 0.084 and Dev3 = 0.337). **Implication for us:** Dev1 absolute scores are a poor preview of Dev3 absolutes. These numbers are external references only.

### 1.6 Lexical overlap with targets

Deferred: requires reading target page text from the corpus (analysis-only; not a pipeline input). To be filled after corpus access in Step 4 / later analysis.

---

## 2. Qrels

### 2.1 Format

Standard TREC four-column qrels, whitespace-separated:

```text
qid 0 docid rel
```

| Property | Train | Dev1 | Dev3 |
|---|---|---|---|
| Lines | 143 | 142 | 536 |
| Iteration column | always `0` | always `0` | always `0` |
| Relevance values | always `1` | always `1` | always `1` |
| Judgments per `qid` | exactly 1 | exactly 1 | exactly 1 |
| `qid` set vs queries | exact match | exact match | exact match |
| `docid` form | digit strings (Wikipedia page IDs) | same | same |

Example Dev3 lines (first three):

```text
2001 0 9046237 1
2002 0 1203157 1
2003 0 402866 1
```

Consequence for metrics (plan 03): with one relevant document per query, Success@K equals Recall@K.

### 2.2 Target presence in corpus

Not verified yet (corpus / offsets not downloaded). Official documentation states the corpus contains the correct answer for every train, dev, and test query. Verification is a Step 4 / indexing prerequisite.

---

## 3. Corpus

| Item | Official value |
|---|---|
| File | `trec-tot-2025-corpus.jsonl.gz` |
| Documents | 6,407,814 |
| Compressed size | ≈ 9.14 GB (9,139,724,653 bytes on Zenodo) |
| md5 | `a2c82398aa86df6a68c8706b9b462bf2` |
| Offsets file | `trec-tot-2025-offsets.jsonl.gz` (md5 `00678e3155d962bb244e034e6401b79b`) |

Full gzip was **not** kept locally. Streamed samples: `data/samples/corpus_sample_10k.jsonl` (10,000 docs) and `data/samples/corpus_sample_100k.jsonl` (**62,624** docs; stream ended early). Feasibility write-up: `logs/feasibility/corpus_feasibility.md`.

### 3.1 Document fields (confirmed on streamed sample)

On-disk JSON keys: **`id`, `url`, `title`, `text`**.

- Guidelines prose says `doc_id`; the live JSONL uses **`id`**.
- `ir_datasets` maps this to `doc_id`.
- Qrels use digit-string Wikipedia page IDs; normalize with `str(doc["id"])`.

Example (first sample doc): `id=12`, title `Anarchism`, url `https://en.wikipedia.org/wiki/Anarchism`.

### 3.2 Document representation choice

Not frozen yet (feeds D-02/D-03). Feasibility timings used **`title + first 1000 characters of text`** as a working default (mean ≈ 926 chars on the 62k sample).

### 3.3 Length sample (62,624 docs from stream)

Raw `text` characters: mean 5012, median 2520, min 6, max 211728, p95 17288.  
Title characters: mean 18.7, median 16.

---

## 4. Split discipline reminder

| Split | Role in this project | Present in `data/` |
|---|---|---|
| Train | Prompt development | yes |
| Dev1 | Pilots / tuning | yes |
| Dev2 | Only if A1 approved | no |
| Dev3 | Final evaluation only | yes |
| Test | Never | no |

---

## 5. Open items after Step 4

1. ~~Confirm corpus JSON field name~~ → **`id`** on disk.
2. Freeze document representation (working default: title + 1k chars) when logging D-02/D-03.
3. Verify all Dev3/Dev1/Train qrels `docid`s exist via offsets or Anserini index.
4. **You decide D-01** — recommendation is Tier B (`logs/feasibility/corpus_feasibility.md`).
