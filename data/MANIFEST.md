# Data Manifest

Download date (UTC): 2026-11-08  
Downloader: `python -m tot.data_download`  
Policy: **Train / Dev1 / Dev3 only. Test was not downloaded.**

## Which Zenodo record?

| Record | DOI | Role |
|---|---|---|
| **[18235970](https://zenodo.org/records/18235970)** | 10.5281/zenodo.18235970 | **Used here** (listed in `scope.md`). Version dated 2026-01-13. Hosts corpus, offsets, train/dev1/dev2/dev3, **and Test queries**. |
| [15356599](https://zenodo.org/records/15356599) | 10.5281/zenodo.15356599 | Earlier version (2025-05-07) of the same concept DOI `10.5281/zenodo.15356598`. Same md5s for train/dev1/dev3/corpus/offsets. File list at fetch time did **not** include Test queries. |
| Parent concept | 10.5281/zenodo.15356598 | Version family for “TREC-ToT 2025”. |

**Verdict:** Query/qrel bytes for Train/Dev1/Dev3 are identical across these records (matching official md5s). We pin downloads to **record 18235970** because that is what `scope.md` cites and it is the fuller file set. Trust **md5**, not the record number alone.

Official checksum table also published at: https://trec-tot.github.io/guidelines.html

## Files downloaded (verified)

| Local path | Zenodo remote name | Entries | Bytes | Expected md5 | Actual md5 | Match |
|---|---|--:|--:|---|---|---|
| `data/train-queries.jsonl` | `train-2025-queries.jsonl` | 143 | 118816 | `288b7707b4e897f7447aac2cc2f613be` | `288b7707b4e897f7447aac2cc2f613be` | OK |
| `data/train-qrels.txt` | `train-2025-qrel.txt` | 143 | 2341 | `10a3c727fc5806ec4510f7a071b57cd7` | `10a3c727fc5806ec4510f7a071b57cd7` | OK |
| `data/dev1-queries.jsonl` | `dev1-2025-queries.jsonl` | 142 | 112237 | `b87c2f51d058de844e258a69b02e70fc` | `b87c2f51d058de844e258a69b02e70fc` | OK |
| `data/dev1-qrels.txt` | `dev1-2025-qrel.txt` | 142 | 2332 | `0c913ce8b5b287c73a6dfac662971e82` | `0c913ce8b5b287c73a6dfac662971e82` | OK |
| `data/dev3-queries.jsonl` | `dev3-2025-queries.jsonl` | 536 | 717935 | `259c11645694a3c5230b66c7852d4d80` | `259c11645694a3c5230b66c7852d4d80` | OK |
| `data/dev3-qrels.txt` | `dev3-2025-qrel.txt` | 536 | 9151 | `48ab0d24a5946861546e54064238477f` | `48ab0d24a5946861546e54064238477f` | OK |

Download URL pattern:

```text
https://zenodo.org/records/18235970/files/<remote-name>?download=1
```

## Not downloaded (by design)

| Remote name | Reason |
|---|---|
| `test-2025-queries.jsonl` | Contract: Test split never loaded |
| `test-2025-qrel.txt` | Same (and would be evaluation leakage) |
| `test-2025-query-mapping.tsv` | Test-related |
| `dev2-2025-queries.jsonl` / `dev2-2025-qrel.txt` | Optional; only if amendment A1 (D-10) is approved |
| `trec-tot-2025-corpus.jsonl.gz` (~9.1 GB) | Deferred to corpus feasibility (plan 02 Step A/B / Step 4) |
| `trec-tot-2025-offsets.jsonl.gz` (~62 MB) | Deferred with corpus |

Corpus remote checksums (for later verification, not yet downloaded):

| Remote name | Entries | md5 |
|---|--:|---|
| `trec-tot-2025-corpus.jsonl.gz` | 6,407,814 | `a2c82398aa86df6a68c8706b9b462bf2` |
| `trec-tot-2025-offsets.jsonl.gz` | 6,407,814 | `00678e3155d962bb244e034e6401b79b` |

## Alternate access (not used for these bytes)

- `ir_datasets` ids: `trec-tot/2025/train`, `trec-tot/2025/dev1`, `trec-tot/2025/dev3`
- Hugging Face / IR-Datasets links listed on the track guidelines page
