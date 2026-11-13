# Corpus Feasibility (plan_v2/02 §4 Steps A & B)

Date: 2026-11-08 (measurements) / completed 2026-11-12  
Environment for BM25 sample timing: local macOS arm64 (not Colab). Dense full-corpus encode rates below are **scenarios**, not measured on a T4 — re-measure on Colab before claiming GPU throughput.  
No experiment scores. Recommendation for **D-01** only — not frozen until you approve.

---

## Step A — Ready-made resources

### Official baselines (`TREC-ToT/bench`)

| Resource | Location | Notes |
|---|---|---|
| Code + runs | https://github.com/TREC-ToT/bench (`trec25/`) | Anserini BM25, PyTerrier BM25, Lightning IR dense |
| Prepared indices | https://github.com/TREC-ToT/bench/tree/main/trec25#prepared-indices | Also mirrored as HF dataset `webis/TREC-ToT-Baselines` |
| Dev3 baseline metrics (official table) | track guidelines / bench README | PyTerrier BM25 nDCG@10=0.337, R@1000=0.771; Anserini BM25 weaker; dense full-corpus nDCG@10=0.014 on Dev3 (sanity check only) |

### Prepared index sizes (HTTP Content-Length verified 2026-11-08)

| Index | Framework | Compressed download | Official md5 | Fits free Drive (15 GB)? | Fits typical Colab local disk? |
|---|---|---:|---|---|---|
| `trec-tot-2025-anserini-index.zip` | Anserini | **1.77 GB** (`1765121233` B) | `b04afdf33519013bf08857005a6cbd88` | Yes | Yes |
| `trec-tot-2025-pyterrier-index.zip` | PyTerrier | **11.6 GB** (`11617237183` B) | `a9a22ed35abb6cea842a7c5734987c82` | Tight (≈ Drive quota alone) | Risky / often no |
| `trec-tot-2025-index.zip` | Lightning IR dense | **18 GB** (stated in bench README) | `9c61195c678e16a6146d09566f62229d` | **No** | Unlikely on Free |

URLs:

- https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip
- https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip
- https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-index.zip

### Other access

| Path | Finding |
|---|---|
| `ir_datasets` `trec-tot/2025` | Docs typed as `(doc_id, title, url, text)`; downloads Zenodo corpus on first use |
| Hugging Face | Track points at Wikipedia dump example; coordinator baselines live under `webis/TREC-ToT-Baselines` |
| 2026 reformulation paper | Full-corpus BM25 + **dense rerank of BM25 top-1000**; hardware was 500 GB RAM + 2× RTX 6000 — not Colab Free |

**Step A conclusion:** A usable **full-corpus BM25** path exists via the **1.77 GB Anserini** prebuilt index (or build with `bm25s` / Pyserini). A usable **full-corpus dense index** does **not** fit Drive (18 GB prebuilt; self-built vectors ≈ 4.6–18 GB plus FAISS). Published ToT pipelines already treat dense as a **reranker on BM25 candidates**.

---

## Step B — Cost arithmetic & sample timings

### Corpus sample used

| Sample | Docs | How obtained |
|---|--:|---|
| `data/samples/corpus_sample_10k.jsonl` | 10,000 | Streamed from Zenodo gzip (partial download) |
| `data/samples/corpus_sample_100k.jsonl` | **62,624** (stream cut by EOF before 100k) | Same; treat as ~63k sample |

On-disk JSON fields (confirmed on sample): **`id`, `url`, `title`, `text`**.  
`ir_datasets` exposes `id` as `doc_id`. Use `str(id)` to match qrels.

Document text length on 62,624-doc sample (raw `text`): mean 5012 chars, median 2520, p95 17288, max 211728.  
Working representation timed: `title + first 1000 chars of text` (mean ≈ 926 chars).

### Dense storage (full 6,407,814 docs)

| Setting | Vector matrix size |
|---|---:|
| 384-dim fp16 | ≈ **4.6 GB** |
| 384-dim fp32 | ≈ **9.2 GB** |
| 768-dim fp16 | ≈ **9.2 GB** |
| 768-dim fp32 | ≈ **18.3 GB** |

Plus FAISS index overhead, doc-id maps, and the need to **encode** every doc. Encode wall-time scenarios for 6.4M docs (not measured here):

| Assumed docs/s | Full-corpus encode |
|---|---|
| 50 (slow CPU) | ≈ 36 h |
| 200 | ≈ 9 h |
| 400 (plausible small model on T4) | ≈ 4.4 h |
| 800 (optimistic) | ≈ 2.2 h |

Any full dense encode needs **sharded, resumable** jobs across Colab disconnects, and likely exceeds comfortable Drive+session budget even if vectors are quantized.

### BM25 sample indexing (`bm25s`, representation = title + 1k chars)

Logged in `logs/feasibility/bm25s_timing.json`:

| n | tokenize+index | docs/s | saved index bytes | Extrapolated time @ 6.4M | Extrapolated size @ 6.4M |
|--:|---:|---:|---:|---:|---:|
| 10,000 | 0.48 s | ≈ 20.7k | 8.4 MB | ≈ 0.09 h | ≈ 5.0 GB |
| 50,000 | 4.12 s | ≈ 12.1k | 36.9 MB | ≈ 0.15 h | ≈ 4.4 GB |
| 62,624 | 4.46 s | ≈ 14.0k | 45.5 MB | ≈ 0.13 h | ≈ 4.3 GB |

Caveats: this is **in-memory `bm25s`**, not Lucene; extrapolation of size/time is approximate; Anserini prebuilt (1.77 GB zip) is the practical full-corpus BM25 artifact for Colab.

### Tier B candidate dense cost (order-of-magnitude)

Dev3 originals only: 536 × 1000 candidates = 536k encodes.  
At 400 docs/s ≈ **22 min**; with ~6 query variants (original + clues + reformulation) ≈ **2 h** of GPU encode if naively re-encoded every list (cacheable / dedupe by docid reduces this a lot).

Much cheaper and more Colab-compatible than encoding 6.4M docs.

### Drive / Colab constraints (design inputs)

| Constraint | Implication |
|---|---|
| Drive free ≈ 15 GB | Anserini index OK; PyTerrier zip alone nearly fills Drive; dense prebuilt does not fit |
| Colab session wipe + disconnects | Prefer downloadable BM25 index; dense must be sharded or candidate-only |
| Literature precedent | Tier B cascade matches 2026 ToT reformulation pipeline |

---

## Step C — Tier recommendation for D-01

| Tier | Feasible on Colab Free? | Label if chosen |
|---|---|---|
| **A** Full-corpus BM25 **and** full-corpus dense | BM25 yes (via Anserini prebuilt). Dense **no** under Drive/session constraints without heroic sharding and still-questionable permanence | "full-corpus" |
| **B** Full-corpus BM25 → dense **reranking** of BM25 top-K | **Yes** — preferred | "dense reranking of BM25 candidates" |
| **C** Shared reduced sub-corpus | Fallback only if Anserini index + BM25 retrieval prove unusable | "reduced-corpus setting; not comparable to official numbers" |

### Recommendation: **Tier B**

Reasons (evidence-based):

1. Prebuilt **Anserini 1.77 GB** index makes full-corpus BM25 realistic.
2. Prebuilt dense index is **18 GB**; self-built dense vectors are **4.6–18 GB** plus multi-hour encode — poor fit for Colab Free + 15 GB Drive.
3. Tier B is the cascade used in the 2026 reformulation paper and is explicitly allowed by scope §15.2 if labeled.
4. Under Tier B, name dense experiments **`E2-R` / `E5-R`** (reranking, not retrieval) in every table.

**Not freezing D-01** until you confirm. If you reject B, next choice is C (with pool rules in plan 02); A only if you accept owning a resumable full dense encode outside Free constraints.

### Suggested follow-ups after you freeze D-01 = B

1. Download Anserini index to Colab local disk (optional copy of zip to Drive).
2. Verify md5 `b04afdf33519013bf08857005a6cbd88`.
3. Smoke-retrieve 20 Dev1 queries; measure s/query and peak RAM (gate test).
4. On Colab T4, time encoding 10k sample docs with the shortlisted dense model; replace the scenario rates above with measured docs/s.
5. Log the frozen row in `configs/decisions.md`.
