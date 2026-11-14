#!/usr/bin/env python3
"""Builds the comprehensive self-contained Google Colab notebook for IT550 ToT Retrieval.

Contains both:
- PATH A: Full 6.4M Wikipedia Corpus (Tier B & Upgrades: E1 - E12)
- PATH B: GPU Sub-Corpus Vector Retrieval (Tier C: B1 - B12)
"""

import json
from pathlib import Path

cells = []

def add_md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]})

def add_code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.split("\n")]})

# Header & Overview
add_md("""# LLM-Enhanced Tip-of-the-Tongue Retrieval (IT550)
## Complete Standalone Reproduction Notebook: Path A (6.4M Corpus) & Path B (GPU Sub-Corpus)

This notebook runs completely standalone in Google Colab (Tesla T4 GPU). It compares two parallel retrieval paradigms on the exact same 12 experimental methods:
- **PATH A (Full 6.4M Wikipedia Corpus):** Real-world web-scale retrieval using Lucene inverted index BM25, candidate text lookup, multi-stage dense reranking, and LLM expansion.
- **PATH B (GPU Sub-Corpus Vector Index):** In-memory 100% dense GPU vector search on an extracted sub-corpus (~10,000 documents containing all targets + hard distractors).""")

# Cell 1: Environment & Dependencies
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
        "pyserini", "sentence-transformers", "torch", "openai"
    ], check=True)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

# Cell 2: Downloads
add_md("""## 2. Dataset, Snippets, and Index Downloads""")
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
gh_data = "https://raw.githubusercontent.com/dhyey2075/it550-tot-retrieval/main/data"
download(f"{gh_cache}/facets_dev1_extractive.jsonl", "cache/facets_dev1_extractive.jsonl")
download(f"{gh_cache}/reformulations_dev1_keyword.jsonl", "cache/reformulations_dev1_keyword.jsonl")
download(f"{gh_data}/dev1_snippets.jsonl.gz", "data/dev1_snippets.jsonl.gz")

index_zip = Path("indices/trec-tot-2025-anserini-index.zip")
download("https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip", index_zip)

index_dir = Path("indices/anserini/trec-tot-2025-anserini-index")
if not index_dir.exists():
    print("Unpacking Anserini index...")
    with zipfile.ZipFile(index_zip, "r") as z:
        z.extractall("indices/anserini")

print("All datasets, snippets, and Lucene index are ready.")""")

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
                obj = json.loads(line)
                qid = str(obj.get("query_id") or obj.get("qid") or obj.get("id") or "").strip()
                query = str(obj.get("query") or obj.get("text") or "").strip()
                queries.append({"qid": qid, "query": query})
    return queries

qrels = load_qrels("data/dev1-qrels.txt")
queries = load_queries("data/dev1-queries.jsonl")

path_a_results = []
path_b_results = []

def evaluate_run(exp_id, run_name, rankings, target_qrels=qrels, depth=1000, store_list=path_a_results):
    rr_list, ndcg_list, r10_list, r100_list, r1000_list = [], [], [], [], []
    for qid, target in target_qrels.items():
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
    
    n = len(target_qrels)
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
    store_list.append(metrics)
    print(f"[{exp_id}] nDCG@10: {metrics['ndcg10']:.4f} | MRR: {metrics['mrr']:.4f} | R@10: {metrics['r10']:.4f} | R@1000: {metrics['r1000']:.4f}")
    return metrics""")

# Cell 4: Core Retrieval Primitives
add_md("""## 4. Retrieval Primitives: BM25, Candidate Snippet Lookup, Dense Model, and RRF""")
add_code("""import gzip
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
import numpy as np

searcher = LuceneSearcher(str(index_dir))

doc_text_cache = {}
snippets_path = Path("data/dev1_snippets.jsonl.gz")
if snippets_path.exists():
    with gzip.open(snippets_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                did = str(item.get("id") or "")
                title = item.get("title", "")
                txt = item.get("text", "")
                doc_text_cache[did] = f"{title}: {txt}".strip()
    print(f"Loaded {len(doc_text_cache)} candidate snippets into cache.")

def get_doc_text(docid, lead_chars=500):
    docid = str(docid)
    if docid in doc_text_cache:
        return doc_text_cache[docid][:lead_chars]
    try:
        raw = searcher.doc(docid).raw()
        data = json.loads(raw) if raw and raw.startswith("{") else {}
        title = data.get("title", "")
        text = data.get("text") or data.get("contents") or raw or ""
        res = f"{title}: {text}".strip()
        if res:
            doc_text_cache[docid] = res
            return res[:lead_chars]
    except Exception:
        pass
    return f"Document {docid}"

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

# PATH A SECTION
add_md("""# ============================================================
# PATH A: Full 6.4M Wikipedia Corpus (Tier B & Upgrades)
# ============================================================
In Path A, retrieval is performed over the complete 6.4 million document Wikipedia corpus using Lucene BM25 first-stage retrieval, multi-query expansion, and multi-stage rerankers.""")

# Cell 5: Path A E1, E2-R, E3
add_md("""### Path A — Experiment 1, 2 & 3: BM25 Baseline, Dense Reranking, and Hybrid RRF (E1, E2-R, E3)""")
add_code("""e1_rankings = {}
e2r_rankings = {}
e3_rankings = {}

print("Running Path A E1 (BM25 Baseline)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    hits = searcher.search(qtext, k=1000)
    e1_rankings[qid] = [h.docid for h in hits]

evaluate_run("E1", "E1_BM25_Baseline", e1_rankings, store_list=path_a_results)

print("Running Path A E2-R (Dense Reranking with E5-small)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    cands = e1_rankings.get(qid, [])
    e2r_rankings[qid] = rerank_dense(qtext, cands, top_n=1000)

evaluate_run("E2-R", "E2-R_Dense_Rerank", e2r_rankings, store_list=path_a_results)

print("Running Path A E3 (Hybrid BM25 + Dense RRF Fusion)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    e3_rankings[qid] = rrf_fuse([e1_rankings.get(qid, []), e2r_rankings.get(qid, [])], k=60)

evaluate_run("E3", "E3_BM25_Dense_RRF", e3_rankings, store_list=path_a_results)""")

# Cell 6: Path A E4, E4c, E5-R, E6
add_md("""### Path A — Experiment 4, 5 & 6: Clue Decomposition & Extractive Fusion (E4, E4c, E5-R, E6)""")
add_code("""facets_by_qid = {}
with open("cache/facets_dev1_extractive.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        facets_by_qid[item["qid"]] = item.get("facets", {})

e4_rankings = {}
e4c_rankings = {}
e5r_rankings = {}
e6_rankings = {}

print("Running Path A E4 (Separate Clues RRF) & E4c (Concatenated Clues)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    facets = facets_by_qid.get(qid, {})
    valid_facets = [str(v).strip() for v in facets.values() if v and str(v).strip() and str(v).lower() != "n/a"]
    
    facet_lists = []
    for facet_text in valid_facets:
        hits = searcher.search(facet_text, k=1000)
        facet_lists.append([h.docid for h in hits])
    
    e4_rankings[qid] = rrf_fuse(facet_lists, k=60) if facet_lists else e1_rankings.get(qid, [])
    
    concat_query = " ".join(valid_facets) if valid_facets else qtext
    hits_concat = searcher.search(concat_query, k=1000)
    e4c_rankings[qid] = [h.docid for h in hits_concat]

evaluate_run("E4", "E4_Clue_Decomp_RRF", e4_rankings, store_list=path_a_results)
evaluate_run("E4c", "E4c_Clue_Concatenation", e4c_rankings, store_list=path_a_results)

print("Running Path A E5-R (Dense Facet Search) & E6 (Extractive Fusion)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    cands = e4_rankings.get(qid, [])
    e5r_rankings[qid] = rerank_dense(qtext, cands, top_n=1000)
    e6_rankings[qid] = rrf_fuse([e1_rankings.get(qid, []), e4c_rankings.get(qid, [])], k=60)

evaluate_run("E5-R", "E5-R_Dense_Facets", e5r_rankings, store_list=path_a_results)
evaluate_run("E6", "E6_Extractive_Fusion", e6_rankings, store_list=path_a_results)""")

# Cell 7: Path A E7
add_md("""### Path A — Experiment 7: Keyword Query Reformulations (E7-bm25, E7-dense)""")
add_code("""reform_by_qid = {}
with open("cache/reformulations_dev1_keyword.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        reform_by_qid[item["qid"]] = item.get("keyword") or item.get("reformulation") or ""

e7_bm25_rankings = {}
e7_dense_rankings = {}

print("Running Path A E7-BM25 & E7-Dense...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    ref_q = reform_by_qid.get(qid, qtext)
    hits = searcher.search(ref_q, k=1000)
    cands = [h.docid for h in hits]
    e7_bm25_rankings[qid] = cands
    e7_dense_rankings[qid] = rerank_dense(ref_q, cands, top_n=1000)

evaluate_run("E7-bm25", "E7_Keyword_BM25", e7_bm25_rankings, store_list=path_a_results)
evaluate_run("E7-dense", "E7_Keyword_Dense", e7_dense_rankings, store_list=path_a_results)""")

# Cell 8: Path A E8
add_md("""### Path A — Experiment 8: Multi-Branch Pipeline Fusion (E8-F4)""")
add_code("""e8_rankings = {}

print("Running Path A E8 (Two-Level Fusion: E1 + E4c + E7-Dense)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    b1 = e1_rankings.get(qid, [])
    b2 = e4c_rankings.get(qid, [])
    b3 = e7_dense_rankings.get(qid, [])
    e8_rankings[qid] = rrf_fuse([b1, b2, b3], k=60)

evaluate_run("E8-F4", "E8_MultiBranch_Fusion", e8_rankings, store_list=path_a_results)""")

# Cell 9: Path A E9
add_md("""### Path A — Experiment 9: Cross-Encoder Reranking (MiniLM)""")
add_code("""from sentence_transformers import CrossEncoder

ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=512)

e9_rankings = {}
K_CE = 30

print(f"Running Path A E9 (MiniLM Cross-Encoder on top-{K_CE} of E8)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    candidates = e8_rankings.get(qid, [])
    top_k = candidates[:K_CE]
    tail = candidates[K_CE:]
    
    pairs = [(qtext, get_doc_text(d)) for d in top_k]
    scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
    order = np.argsort(-scores)
    reranked_top = [top_k[i] for i in order]
    
    e9_rankings[qid] = reranked_top + tail

evaluate_run("E9", "E9_CrossEncoder_MiniLM", e9_rankings, store_list=path_a_results)""")

# Cell 10: OpenAI LLM Configuration
add_md("""## OpenAI LLM Configuration (for E10 HyDE & E12 Listwise Reranking)""")
add_code("""import os
from openai import OpenAI

OPENAI_API_KEY = ""

try:
    from google.colab import userdata
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
except Exception:
    pass

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if openai_client:
    print("OpenAI client initialized successfully.")
else:
    print("Notice: Set OPENAI_API_KEY above to execute E10 and E12.")""")

# Cell 11: Path A E10
add_md("""### Path A — Experiment 10: HyDE & Candidate Title Expansion (E10)""")
add_code("""openai_cache_file = Path("cache/openai_hyde_cache.jsonl")
openai_cache = {}
if openai_cache_file.exists():
    with open(openai_cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                openai_cache[item["qid"]] = item

e10_rankings = {}

if openai_client:
    print("Running Path A E10 (HyDE Wikipedia leads & candidate titles with GPT-4o-mini)...")
    for q in queries:
        qid = str(q.get("qid") or q.get("query_id") or "")
        qtext = str(q.get("query") or q.get("text") or "")
        
        if qid not in openai_cache:
            prompt = (
                "You are an expert encyclopedic researcher. A user is describing an entity they forgot:\\n"
                + qtext + "\\n\\n"
                + "1. Guess 5 most likely titles (movie, book, song, or person).\\n"
                + "2. Write a short 2-3 sentence encyclopedic Wikipedia lead paragraph for this entity.\\n\\n"
                + "Respond in strict JSON:\\n{\\\"titles\\\": [\\\"Title 1\\\", \\\"Title 2\\\", ...], \\\"lead\\\": \\\"...\\\"}"
            )
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(resp.choices[0].message.content)
                openai_cache[qid] = data
                with open(openai_cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"qid": qid, **data}) + "\\n")
            except Exception as e:
                print(f"Error on {qid}: {e}")
                openai_cache[qid] = {"titles": [], "lead": ""}
        
        cached = openai_cache.get(qid, {})
        titles = " ".join(cached.get("titles", []))
        lead = cached.get("lead", "")
        expanded_query = f"{titles} {lead}".strip() or qtext
        
        hits_exp = searcher.search(expanded_query, k=1000)
        exp_cands = [h.docid for h in hits_exp]
        
        e10_rankings[qid] = rrf_fuse([e1_rankings.get(qid, []), exp_cands], k=60)

    evaluate_run("E10", "E10_HyDE_Title_Expansion", e10_rankings, store_list=path_a_results)
else:
    print("Skipping E10: OpenAI API key not provided.")""")

# Cell 12: Path A E11
add_md("""### Path A — Experiment 11: Modern Neural Cross-Encoder (BGE-Reranker-Base)""")
add_code("""from sentence_transformers import CrossEncoder

bge_model = CrossEncoder("BAAI/bge-reranker-base", device=device, max_length=512)

e11_rankings = {}
base_pool = e10_rankings if e10_rankings else (e8_rankings if e8_rankings else e1_rankings)

print("Running Path A E11 (BGE-Reranker-Base reranking top 30 of E10)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    candidates = base_pool.get(qid, [])
    top_k = candidates[:30]
    tail = candidates[30:]
    
    pairs = [(qtext, get_doc_text(d)) for d in top_k]
    scores = bge_model.predict(pairs, batch_size=32, show_progress_bar=False)
    order = np.argsort(-scores)
    reranked_top = [top_k[i] for i in order]
    
    e11_rankings[qid] = reranked_top + tail

evaluate_run("E11", "E11_BGE_CrossEncoder", e11_rankings, store_list=path_a_results)""")

# Cell 13: Path A E12
add_md("""### Path A — Experiment 12: Listwise LLM Reranking (GPT-4o-mini)""")
add_code("""listwise_cache_file = Path("cache/openai_listwise_cache.jsonl")
listwise_cache = {}
if listwise_cache_file.exists():
    with open(listwise_cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                listwise_cache[item["qid"]] = item["ranked_ids"]

e12_rankings = {}
pool_for_llm = e11_rankings if e11_rankings else (e10_rankings if e10_rankings else e1_rankings)
N_RERANK = 20

if openai_client:
    print(f"Running Path A E12 (Listwise LLM Reranking on top-{N_RERANK} candidates)...")
    for q in queries:
        qid = str(q.get("qid") or q.get("query_id") or "")
        qtext = str(q.get("query") or q.get("text") or "")
        candidates = pool_for_llm.get(qid, [])
        top_cands = candidates[:N_RERANK]
        tail = candidates[N_RERANK:]
        
        if qid not in listwise_cache:
            cand_text_list = []
            for idx, d in enumerate(top_cands, start=1):
                txt = get_doc_text(d, lead_chars=300)
                cand_text_list.append(f"[{idx}] {txt}")
            
            items_str = "\\n".join(cand_text_list)
            prompt = (
                "You are an expert search engine reranker for Tip-of-the-Tongue queries.\\n"
                + "Query description: \"" + qtext + "\"\\n\\n"
                + "Candidate Wikipedia articles:\\n" + items_str + "\\n\\n"
                + "Task: Rank ALL candidates from most likely to least likely to be the entity described.\\n"
                + "Respond in strict JSON with a list of the 1-based integer IDs in ranked order, e.g. {\\\"ranking\\\": [3, 1, 5, 2, ...]}"
            )
            
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(resp.choices[0].message.content)
                raw_order = data.get("ranking", [])
                valid_indices = []
                for x in raw_order:
                    if isinstance(x, int) and 1 <= x <= len(top_cands) and x not in valid_indices:
                        valid_indices.append(x)
                for i in range(1, len(top_cands) + 1):
                    if i not in valid_indices:
                        valid_indices.append(i)
                listwise_cache[qid] = valid_indices
                with open(listwise_cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"qid": qid, "ranked_ids": valid_indices}) + "\\n")
            except Exception as e:
                print(f"Listwise error on {qid}: {e}")
                listwise_cache[qid] = list(range(1, len(top_cands) + 1))
        
        ranked_indices = listwise_cache.get(qid, list(range(1, len(top_cands) + 1)))
        reranked_top = [top_cands[i - 1] for i in ranked_indices]
        e12_rankings[qid] = reranked_top + tail

    evaluate_run("E12", "E12_Listwise_GPT4o_Mini", e12_rankings, store_list=path_a_results)
else:
    print("Skipping E12: OpenAI API key not provided.")""")

# PATH B SECTION
add_md("""# ============================================================
# PATH B: GPU Sub-Corpus Vector Retrieval (Exact In-Memory Search)
# ============================================================
In Path B, we build a controlled sub-corpus (~10,000 documents) containing:
1. **All 140 true targets** from `qrels` (100% target reachability).
2. **Hard BM25 distractors** retrieved across all queries.

All document vectors fit directly into GPU VRAM (~70 MB), enabling **100% pure dense vector search across the entire corpus** without sparse BM25 pre-filtering. We run the exact same set of 12 experiments (B1 through B12).""")

# Cell 14: Sub-Corpus Construction & GPU Embedding
add_md("""### Path B — Setup: Sub-Corpus Extraction & GPU Vector Indexing""")
add_code("""subcorpus_ids = set(qrels.values())
for qid in e1_rankings:
    subcorpus_ids.update(e1_rankings[qid][:70])

subcorpus_docids = sorted(list(subcorpus_ids))
print(f"Constructed Path B sub-corpus with {len(subcorpus_docids)} documents (Targets + Hard BM25 Distractors).")

subcorpus_texts = [get_doc_text(d) for d in subcorpus_docids]

print("Encoding entire sub-corpus with e5-small-v2 on T4 GPU...")
doc_embs = dense_model.encode(
    ["passage: " + t for t in subcorpus_texts],
    batch_size=256,
    normalize_embeddings=True,
    convert_to_tensor=True,
    device=device,
    show_progress_bar=True
)
print(f"Sub-corpus vector index ready in GPU VRAM: {doc_embs.shape}")

def search_dense_subcorpus(query_strings, top_k=1000):
    q_embs = dense_model.encode(
        ["query: " + q for q in query_strings],
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False
    )
    sims = torch.matmul(q_embs, doc_embs.T)
    rankings = {}
    k = min(top_k, len(subcorpus_docids))
    for i, q in enumerate(queries):
        qid = str(q.get("qid") or q.get("query_id") or "")
        top_indices = torch.topk(sims[i], k=k).indices.cpu().numpy()
        rankings[qid] = [subcorpus_docids[idx] for idx in top_indices]
    return rankings""")

# Cell 15: Path B B1, B2, B3
add_md("""### Path B — Experiment 1, 2 & 3: Sub-Corpus BM25, Pure Dense GPU Search, and Hybrid RRF (B1, B2, B3)""")
add_code("""b1_rankings = {}
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    b1_rankings[qid] = [d for d in e1_rankings.get(qid, []) if d in subcorpus_ids]

evaluate_run("B1", "B1_BM25_Baseline", b1_rankings, store_list=path_b_results)

print("Running Path B B2 (Pure Dense GPU Vector Search on full sub-corpus)...")
raw_queries = [str(q.get("query") or q.get("text") or "") for q in queries]
b2_rankings = search_dense_subcorpus(raw_queries, top_k=1000)

evaluate_run("B2", "B2_Pure_Dense_GPU", b2_rankings, store_list=path_b_results)

print("Running Path B B3 (Hybrid BM25 + Dense RRF Fusion)...")
b3_rankings = {}
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    b3_rankings[qid] = rrf_fuse([b1_rankings.get(qid, []), b2_rankings.get(qid, [])], k=60)

evaluate_run("B3", "B3_Hybrid_RRF", b3_rankings, store_list=path_b_results)""")

# Cell 16: Path B B4, B4c, B6
add_md("""### Path B — Experiment 4, 5 & 6: Sub-Corpus Clue Decomposition & Extractive Fusion (B4, B4c, B5-R, B6)""")
add_code("""b4_rankings = {}
b4c_rankings = {}
b5r_rankings = {}
b6_rankings = {}

print("Running Path B B4 (Separate Clues RRF) & B4c (Concatenated Clues Dense GPU)...")
concat_clue_queries = []
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    facets = facets_by_qid.get(qid, {})
    valid_facets = [str(v).strip() for v in facets.values() if v and str(v).strip() and str(v).lower() != "n/a"]
    concat_clue_queries.append(" ".join(valid_facets) if valid_facets else qtext)
    
    # Filter Path A E4 hits to subcorpus
    b4_rankings[qid] = [d for d in e4_rankings.get(qid, []) if d in subcorpus_ids]

b4c_rankings = search_dense_subcorpus(concat_clue_queries, top_k=1000)

evaluate_run("B4", "B4_Clue_Decomp_RRF", b4_rankings, store_list=path_b_results)
evaluate_run("B4c", "B4c_Clue_Concatenation", b4c_rankings, store_list=path_b_results)

print("Running Path B B5-R (Dense Facets) & B6 (Extractive Fusion)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    cands = b4_rankings.get(qid, [])
    b5r_rankings[qid] = rerank_dense(qtext, cands, top_n=1000)
    b6_rankings[qid] = rrf_fuse([b1_rankings.get(qid, []), b4c_rankings.get(qid, [])], k=60)

evaluate_run("B5-R", "B5-R_Dense_Facets", b5r_rankings, store_list=path_b_results)
evaluate_run("B6", "B6_Extractive_Fusion", b6_rankings, store_list=path_b_results)""")

# Cell 17: Path B B7
add_md("""### Path B — Experiment 7: Sub-Corpus Keyword Query Reformulations (B7-bm25, B7-dense)""")
add_code("""b7_bm25_rankings = {}
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    b7_bm25_rankings[qid] = [d for d in e7_bm25_rankings.get(qid, []) if d in subcorpus_ids]

kw_queries = [reform_by_qid.get(str(q.get("qid") or q.get("query_id") or ""), str(q.get("query") or q.get("text") or "")) for q in queries]
b7_dense_rankings = search_dense_subcorpus(kw_queries, top_k=1000)

evaluate_run("B7-bm25", "B7_Keyword_BM25", b7_bm25_rankings, store_list=path_b_results)
evaluate_run("B7-dense", "B7_Keyword_Dense", b7_dense_rankings, store_list=path_b_results)""")

# Cell 18: Path B B8
add_md("""### Path B — Experiment 8: Sub-Corpus Multi-Branch Pipeline Fusion (B8-F4)""")
add_code("""b8_rankings = {}

print("Running Path B B8 (Two-Level Fusion: B1 + B4c + B7-Dense)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    b1 = b1_rankings.get(qid, [])
    b2 = b4c_rankings.get(qid, [])
    b3 = b7_dense_rankings.get(qid, [])
    b8_rankings[qid] = rrf_fuse([b1, b2, b3], k=60)

evaluate_run("B8-F4", "B8_MultiBranch_Fusion", b8_rankings, store_list=path_b_results)""")

# Cell 19: Path B B9
add_md("""### Path B — Experiment 9: Sub-Corpus Cross-Encoder Reranking (MiniLM)""")
add_code("""b9_rankings = {}

print(f"Running Path B B9 (MiniLM Cross-Encoder on top-{K_CE} of B8)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    candidates = b8_rankings.get(qid, [])
    top_k = candidates[:K_CE]
    tail = candidates[K_CE:]
    
    pairs = [(qtext, get_doc_text(d)) for d in top_k]
    scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
    order = np.argsort(-scores)
    reranked_top = [top_k[i] for i in order]
    
    b9_rankings[qid] = reranked_top + tail

evaluate_run("B9", "B9_CrossEncoder_MiniLM", b9_rankings, store_list=path_b_results)""")

# Cell 20: Path B B10
add_md("""### Path B — Experiment 10: Sub-Corpus HyDE & Expansion Retrieval (B10)""")
add_code("""b10_rankings = {}

if openai_cache:
    print("Running Path B B10 (HyDE Sub-Corpus Dense Vector Retrieval + BM25 Fusion)...")
    hyde_queries = []
    for q in queries:
        qid = str(q.get("qid") or q.get("query_id") or "")
        qtext = str(q.get("query") or q.get("text") or "")
        cached = openai_cache.get(qid, {})
        titles = " ".join(cached.get("titles", []))
        lead = cached.get("lead", "")
        hyde_queries.append(f"{titles} {lead}".strip() or qtext)
    
    b10_dense = search_dense_subcorpus(hyde_queries, top_k=1000)
    
    for q in queries:
        qid = str(q.get("qid") or q.get("query_id") or "")
        b10_rankings[qid] = rrf_fuse([b1_rankings.get(qid, []), b10_dense.get(qid, [])], k=60)
    
    evaluate_run("B10", "B10_HyDE_Title_Expansion", b10_rankings, store_list=path_b_results)
else:
    print("Skipping B10: HyDE cache not available.")""")

# Cell 21: Path B B11
add_md("""### Path B — Experiment 11: Sub-Corpus Modern Neural Cross-Encoder (BGE-Reranker-Base)""")
add_code("""b11_rankings = {}
base_pool_b = b10_rankings if b10_rankings else (b8_rankings if b8_rankings else b1_rankings)

print("Running Path B B11 (BGE-Reranker-Base reranking top 30 of B10)...")
for q in queries:
    qid = str(q.get("qid") or q.get("query_id") or "")
    qtext = str(q.get("query") or q.get("text") or "")
    candidates = base_pool_b.get(qid, [])
    top_k = candidates[:30]
    tail = candidates[30:]
    
    pairs = [(qtext, get_doc_text(d)) for d in top_k]
    scores = bge_model.predict(pairs, batch_size=32, show_progress_bar=False)
    order = np.argsort(-scores)
    reranked_top = [top_k[i] for i in order]
    
    b11_rankings[qid] = reranked_top + tail

evaluate_run("B11", "B11_BGE_CrossEncoder", b11_rankings, store_list=path_b_results)""")

# Cell 22: Path B B12
add_md("""### Path B — Experiment 12: Sub-Corpus Listwise LLM Reranking (GPT-4o-mini)""")
add_code("""b12_rankings = {}
pool_for_llm_b = b11_rankings if b11_rankings else (b10_rankings if b10_rankings else b1_rankings)

if openai_client:
    print("Running Path B B12 (Listwise LLM Reranking on top 20 candidates of B11)...")
    for q in queries:
        qid = str(q.get("qid") or q.get("query_id") or "")
        qtext = str(q.get("query") or q.get("text") or "")
        candidates = pool_for_llm_b.get(qid, [])
        top_cands = candidates[:N_RERANK]
        tail = candidates[N_RERANK:]
        
        # Reuse or call listwise LLM
        if qid not in listwise_cache:
            cand_text_list = []
            for idx, d in enumerate(top_cands, start=1):
                txt = get_doc_text(d, lead_chars=300)
                cand_text_list.append(f"[{idx}] {txt}")
            
            items_str = "\\n".join(cand_text_list)
            prompt = (
                "You are an expert search engine reranker for Tip-of-the-Tongue queries.\\n"
                + "Query description: \"" + qtext + "\"\\n\\n"
                + "Candidate Wikipedia articles:\\n" + items_str + "\\n\\n"
                + "Task: Rank ALL candidates from most likely to least likely to be the entity described.\\n"
                + "Respond in strict JSON with a list of the 1-based integer IDs in ranked order, e.g. {\\\"ranking\\\": [3, 1, 5, 2, ...]}"
            )
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(resp.choices[0].message.content)
                raw_order = data.get("ranking", [])
                valid_indices = []
                for x in raw_order:
                    if isinstance(x, int) and 1 <= x <= len(top_cands) and x not in valid_indices:
                        valid_indices.append(x)
                for i in range(1, len(top_cands) + 1):
                    if i not in valid_indices:
                        valid_indices.append(i)
                listwise_cache[qid] = valid_indices
            except Exception:
                valid_indices = list(range(1, len(top_cands) + 1))
        
        ranked_indices = listwise_cache.get(qid, list(range(1, len(top_cands) + 1)))
        reranked_top = [top_cands[i - 1] for i in ranked_indices]
        b12_rankings[qid] = reranked_top + tail
    
    evaluate_run("B12", "B12_Listwise_GPT4o_Mini", b12_rankings, store_list=path_b_results)
else:
    print("Skipping B12: OpenAI API key not provided.")""")

# Cell 23: Benchmark Synthesis & Results Comparison
add_md("""# ============================================================
# BENCHMARK SYNTHESIS & RESULTS COMPARISON
# ============================================================
Displays all experiment results side by side for both Path A (6.4M Wikipedia Corpus) and Path B (GPU Sub-Corpus).""")
add_code("""all_results = path_a_results + path_b_results

fieldnames = ["exp_id", "run_name", "n_queries", "ndcg10", "mrr", "r10", "r100", "r1000"]
with open("results/results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

def print_table(title, rows):
    print()
    print(f"=== {title} ===")
    header = f"| {'Exp':<10} | {'Method':<28} | {'nDCG@10':<9} | {'MRR':<8} | {'R@10':<8} | {'R@100':<8} | {'R@1000':<8} |"
    print(header)
    print("|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|")
    for r in rows:
        print(f"| {r['exp_id']:<10} | {r['run_name']:<28} | {r['ndcg10']:<9.4f} | {r['mrr']:<8.4f} | {r['r10']:<8.4f} | {r['r100']:<8.4f} | {r['r1000']:<8.4f} |")

print_table("PATH A: FULL 6.4M WIKIPEDIA CORPUS (TIER B & UPGRADES)", path_a_results)
print_table("PATH B: DEDICATED GPU VECTOR RETRIEVAL (TIER C SUB-CORPUS)", path_b_results)

# Side-by-side comparison table
print()
print("=== DIRECT SIDE-BY-SIDE COMPARISON (PATH A vs. PATH B) ===")
comp_header = f"| {'Method':<24} | {'Path A nDCG@10':<15} | {'Path B nDCG@10':<15} | {'Path A R@10':<12} | {'Path B R@10':<12} | {'Path A R@1000':<14} | {'Path B R@1000':<14} |"
print(comp_header)
print("|" + "|".join(["-" * (len(col) + 2) for col in comp_header.split("|")[1:-1]]) + "|")

a_dict = {r['exp_id'].replace('E', ''): r for r in path_a_results}
b_dict = {r['exp_id'].replace('B', ''): r for r in path_b_results}

for k in sorted(set(list(a_dict.keys()) + list(b_dict.keys()))):
    ra = a_dict.get(k)
    rb = b_dict.get(k)
    name = (ra or rb)['run_name'].split('_', 1)[-1]
    a_ndcg = f"{ra['ndcg10']:.4f}" if ra else "N/A"
    b_ndcg = f"{rb['ndcg10']:.4f}" if rb else "N/A"
    a_r10 = f"{ra['r10']:.4f}" if ra else "N/A"
    b_r10 = f"{rb['r10']:.4f}" if rb else "N/A"
    a_r1000 = f"{ra['r1000']:.4f}" if ra else "N/A"
    b_r1000 = f"{rb['r1000']:.4f}" if rb else "N/A"
    print(f"| {name:<24} | {a_ndcg:<15} | {b_ndcg:<15} | {a_r10:<12} | {b_r10:<12} | {a_r1000:<14} | {b_r1000:<14} |")

print()
print("All evaluation rows successfully exported to results/results.csv")""")

out_paths = [Path("notebooks/tot_standalone_colab.ipynb"), Path("tot_standalone_colab.ipynb")]
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
for p in out_paths:
    p.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated standalone notebook at {p} ({p.stat().st_size / 1024:.1f} KB)")
