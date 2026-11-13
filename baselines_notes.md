# Baselines notes (plan 04 / 05)

## Frozen so far

- **D-01 = Tier B** (`dense_rerank_bm25`): full-corpus BM25 → dense rerank of BM25 candidates.
- Dense experiment ids: **E2-R**, **E5-R** (not full-corpus dense retrieval).

## Not frozen yet (need Dev1 evidence)

| ID | Default in code until freeze | How to freeze |
|---|---|---|
| D-02 | Anserini/Pyserini engine defaults for k1, b | Coarse grid on Train/Dev1, primary metric Recall@1000 |
| D-03 | Bake-off preset start: `e5-small` (`intfloat/e5-small-v2` with query/passage prefixes) | Mini bake-off vs MiniLM / BGE-small on Dev1 |
| D-04 | RRF k=60 | Try {20,60,100} on Dev1; keep 60 if within noise |
| D-06 | Extractive facets; E4–E6 without original list; E6 protocol=`flat` | Compare +/- original and flat vs two_level on Dev1 |

## Implementation map

| Piece | Path |
|---|---|
| RRF | `retrieval/rrf.py` |
| BM25 | `retrieval/bm25/search.py` (`pyserini` / `bm25s`) |
| Anserini download | `retrieval/bm25/download_index.py` |
| Dense rerank | `retrieval/dense/rerank.py` |
| E1–E3 | `retrieval/pipeline_e1e3.py` |
| Facets | `tot/decomposition.py` + `prompts/decomposition_extractive.txt` |
| E4–E6 | `retrieval/pipeline_e4e6.py` |

## Surprises / quirks

_(fill during Colab runs)_
