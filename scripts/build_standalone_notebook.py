import json
from pathlib import Path

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Cell 0: Title
add_md("""# LLM-Enhanced Tip-of-the-Tongue Retrieval (IT550)
### Fully Self-Contained Standalone Reproduction Notebook (E1 – E9)

This notebook runs completely independently in Google Colab without requiring any repository cloning or local folders. All dependencies, datasets, indexes, and assets are fetched automatically.""")

# Cell 1: Environment Setup
add_md("""## 1. Environment & Dependencies Setup""")
add_code("""import os, sys, subprocess
from pathlib import Path

def in_colab():
    return "google.colab" in sys.modules or os.path.exists("/content")

if in_colab():
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "openjdk-21-jdk-headless"], check=True)
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "pyserini", "sentence-transformers", "torch"
    ], check=True)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

# Cell 2: Downloads
add_md("""## 2. Dataset & Index Downloads""")
add_code("""import json, urllib.request, zipfile

for d in ["data", "indices", "cache", "runs", "results"]:
    Path(d).mkdir(parents=True, exist_ok=True)

def download(url, dest):
    dest = Path(dest)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)

download("https://zenodo.org/records/18235970/files/dev1-2025-queries.jsonl?download=1", "data/dev1-queries.jsonl")
download("https://zenodo.org/records/18235970/files/dev1-2025-qrel.txt?download=1", "data/dev1-qrels.txt")

gh_cache = "https://raw.githubusercontent.com/dhyey2075/it550-tot-retrieval/main/cache"
download(f"{gh_cache}/facets_dev1_extractive.jsonl", "cache/facets_dev1_extractive.jsonl")
download(f"{gh_cache}/reformulations_dev1_keyword.jsonl", "cache/reformulations_dev1_keyword.jsonl")

index_zip = Path("indices/trec-tot-2025-anserini-index.zip")
download("https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip", index_zip)

index_dir = Path("indices/anserini/trec-tot-2025-anserini-index")
if not index_dir.exists():
    print("Unpacking Anserini index...")
    with zipfile.ZipFile(index_zip, "r") as z:
        z.extractall("indices/anserini")

print("All datasets and Lucene index are ready.")""")

# Cell 3: Evaluation Engine
add_md("""## 3. Evaluation Engine (TREC Known-Item Metrics)""")
add_code("""import math, csv

def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qrels[parts[0]] = parts[2]
    return qrels

def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

qrels = load_qrels("data/dev1-qrels.txt")
queries = load_queries("data/dev1-queries.jsonl")

results_log = []

def evaluate_run(exp_id, run_name, rankings, depth=1000):
    rr_list, ndcg_list, r10_list, r100_list, r1000_list = [], [], [], [], []
    for qid, target in qrels.items():
        cands = rankings.get(qid, [])[:depth]
        rank = None
        for i, docid in enumerate(cands, start=1):
            if str(docid) == str(target):
                rank = i
                break
        if rank is None:
            rr_list.append(0.0)
            ndcg_list.append(0.0)
            r10_list.append(0.0)
            r100_list.append(0.0)
            r1000_list.append(0.0)
        else:
            rr_list.append(1.0 / rank)
            ndcg_list.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
            r10_list.append(1.0 if rank <= 10 else 0.0)
            r100_list.append(1.0 if rank <= 100 else 0.0)
            r1000_list.append(1.0 if rank <= 1000 else 0.0)
    
    n = len(qrels)
    metrics = {
        "exp_id": exp_id,
        "run_name": run_name,
        "n_queries": n,
        "ndcg10": sum(ndcg_list) / n,
        "mrr": sum(rr_list) / n,
        "r10": sum(r10_list) / n,
        "r100": sum(r100_list) / n,
        "r1000": sum(r1000_list) / n,
    }
    results_log.append(metrics)
    print(f"[{exp_id}] nDCG@10: {metrics['ndcg10']:.4f} | MRR: {metrics['mrr']:.4f} | R@10: {metrics['r10']:.4f} | R@1000: {metrics['r1000']:.4f}")
    return metrics""")

# Cell 4: Core Retrieval Primitives
add_md("""## 4. Retrieval Primitives: BM25, Text Lookup, Dense Reranking, and RRF""")
add_code("""from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
import numpy as np

searcher = LuceneSearcher(str(index_dir))

def get_doc_text(docid, lead_chars=1000):
    try:
        raw = searcher.doc(str(docid)).raw()
        data = json.loads(raw) if raw and raw.startswith("{") else {}
        title = data.get("title", "")
        text = data.get("text") or data.get("contents") or raw or ""
        return f"{title} {text[:lead_chars]}".strip()
    except Exception:
        return ""

def rrf_fuse(ranked_lists, k=60, top_n=1000):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, docid in enumerate(ranked, start=1):
            scores[docid] += 1.0 / (k + rank)
    sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return sorted_docs[:top_n]

dense_model = SentenceTransformer("intfloat/e5-small-v2", device=device)

def rerank_dense(query, candidate_docids, top_n=1000):
    if not candidate_docids:
        return []
    docids = candidate_docids[:top_n]
    texts = [get_doc_text(d) for d in docids]
    q_emb = dense_model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
    d_emb = dense_model.encode(["passage: " + t for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    scores = d_emb @ q_emb
    ranked_indices = np.argsort(-scores)
    return [docids[i] for i in ranked_indices]""")

# Cell 5: Plan 04 Baselines (E1, E2-R, E3)
add_md("""## 5. Plan 04: Baselines (E1 BM25, E2-R Dense, E3 Hybrid RRF)""")
add_code("""e1_rankings = {}
e2r_rankings = {}
e3_rankings = {}

print("Running E1 (BM25)...")
for q in queries:
    qid = q["qid"]
    hits = searcher.search(q["query"], k=1000)
    e1_rankings[qid] = [h.docid for h in hits]

evaluate_run("E1", "E1_BM25_Baseline", e1_rankings)

print("Running E2-R (Dense Reranking)...")
for q in queries:
    qid = q["qid"]
    cands = e1_rankings.get(qid, [])
    e2r_rankings[qid] = rerank_dense(q["query"], cands, top_n=1000)

evaluate_run("E2-R", "E2-R_Dense_Rerank", e2r_rankings)

print("Running E3 (Hybrid RRF Fusion)...")
for q in queries:
    qid = q["qid"]
    e3_rankings[qid] = rrf_fuse([e1_rankings.get(qid, []), e2r_rankings.get(qid, [])], k=60)

evaluate_run("E3", "E3_BM25_Dense_RRF", e3_rankings)""")

# Cell 6: Plan 05 LLM Query Decomposition (E4, E4c, E5-R, E6)
add_md("""## 6. Plan 05: LLM Query Decomposition (E4, E4c, E5-R, E6)""")
add_code("""facets_by_qid = {}
with open("cache/facets_dev1_extractive.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        facets_by_qid[item["qid"]] = item.get("facets", {})

e4_rankings = {}
e4c_rankings = {}
e5r_rankings = {}
e6_rankings = {}

print("Running E4 (Separate Clues RRF) and E4c (Concatenated Clues)...")
for q in queries:
    qid = q["qid"]
    facets = facets_by_qid.get(qid, {})
    valid_facets = [str(v).strip() for v in facets.values() if v and str(v).strip() and str(v).lower() != "n/a"]
    
    facet_lists = []
    for facet_text in valid_facets:
        hits = searcher.search(facet_text, k=1000)
        facet_lists.append([h.docid for h in hits])
    
    e4_rankings[qid] = rrf_fuse(facet_lists, k=60) if facet_lists else e1_rankings.get(qid, [])
    
    concat_query = " ".join(valid_facets) if valid_facets else q["query"]
    hits_concat = searcher.search(concat_query, k=1000)
    e4c_rankings[qid] = [h.docid for h in hits_concat]

evaluate_run("E4", "E4_Clue_Decomp_RRF", e4_rankings)
evaluate_run("E4c", "E4c_Clue_Concatenation", e4c_rankings)

print("Running E5-R (Dense Facet Search) and E6 (Extractive Fusion)...")
for q in queries:
    qid = q["qid"]
    cands = e4_rankings.get(qid, [])
    e5r_rankings[qid] = rerank_dense(q["query"], cands, top_n=1000)
    e6_rankings[qid] = rrf_fuse([e1_rankings.get(qid, []), e4c_rankings.get(qid, [])], k=60)

evaluate_run("E5-R", "E5-R_Dense_Facets", e5r_rankings)
evaluate_run("E6", "E6_Extractive_Fusion", e6_rankings)""")

# Cell 7: Plan 06 LLM Query Reformulation (E7 Suite)
add_md("""## 7. Plan 06: LLM Query Reformulation (E7 Suite)""")
add_code("""reform_by_qid = {}
with open("cache/reformulations_dev1_keyword.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        reform_by_qid[item["qid"]] = item.get("keyword") or item.get("reformulation") or ""

e7_bm25_rankings = {}
e7_dense_rankings = {}

print("Running E7-BM25 and E7-Dense...")
for q in queries:
    qid = q["qid"]
    ref_q = reform_by_qid.get(qid, q["query"])
    hits = searcher.search(ref_q, k=1000)
    cands = [h.docid for h in hits]
    e7_bm25_rankings[qid] = cands
    e7_dense_rankings[qid] = rerank_dense(ref_q, cands, top_n=1000)

evaluate_run("E7-bm25", "E7_Keyword_BM25", e7_bm25_rankings)
evaluate_run("E7-dense", "E7_Keyword_Dense", e7_dense_rankings)""")

# Cell 8: Plan 07 Two-Level Fusion (E8 Suite)
add_md("""## 8. Plan 07: Multi-Branch Pipeline Fusion (E8 Suite)""")
add_code("""e8_rankings = {}

print("Running E8 (Two-Level Fusion: E1 + E4c + E7-Dense)...")
for q in queries:
    qid = q["qid"]
    b1 = e1_rankings.get(qid, [])
    b2 = e4c_rankings.get(qid, [])
    b3 = e7_dense_rankings.get(qid, [])
    e8_rankings[qid] = rrf_fuse([b1, b2, b3], k=60)

evaluate_run("E8-F4", "E8_MultiBranch_Fusion", e8_rankings)""")

# Cell 9: Plan 08 Cross-Encoder Reranking (E9 Suite)
add_md("""## 9. Plan 08: Cross-Encoder Reranking (E9 Suite)""")
add_code("""from sentence_transformers import CrossEncoder

ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=512)

e9_rankings = {}
K_CE = 30

print(f"Running E9 (Cross-Encoder reranking top-{K_CE} candidates)...")
for q in queries:
    qid = q["qid"]
    candidates = e8_rankings.get(qid, [])
    top_k = candidates[:K_CE]
    tail = candidates[K_CE:]
    
    pairs = [(q["query"], get_doc_text(d)) for d in top_k]
    scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
    order = np.argsort(-scores)
    reranked_top = [top_k[i] for i in order]
    
    e9_rankings[qid] = reranked_top + tail

evaluate_run("E9", "E9_CrossEncoder_MiniLM", e9_rankings)""")

# Cell 10: Plan 09 & 10 Results & Diagnostics
add_md("""## 10. Plans 09 & 10: Final Results Table & Quantitative Diagnostics""")
add_code("""fieldnames = ["exp_id", "run_name", "n_queries", "ndcg10", "mrr", "r10", "r100", "r1000"]
with open("results/results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results_log:
        writer.writerow(row)

print("=== OFFICIAL EXPERIMENTAL RESULTS (E1 - E9) ===")
header = f"| {'Exp':<8} | {'Method':<24} | {'nDCG@10':<9} | {'MRR':<8} | {'R@10':<8} | {'R@100':<8} | {'R@1000':<8} |"
print(header)
print("|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|")
for r in results_log:
    print(f"| {r['exp_id']:<8} | {r['run_name']:<24} | {r['ndcg10']:<9.4f} | {r['mrr']:<8.4f} | {r['r10']:<8.4f} | {r['r100']:<8.4f} | {r['r1000']:<8.4f} |")

print()
print("Results successfully stored in results/results.csv")""")

out_path = Path("notebooks/tot_standalone_colab.ipynb")
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Generated standalone notebook at {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
