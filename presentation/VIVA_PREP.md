# Viva Voce & Presentation Preparation Guide

**Course:** IT550 — Information Retrieval  
**Project:** Tree-of-Thought Retrieval for Tip-of-the-Tongue Queries  
**Target:** Oral Defense, Examiner Inquiries, and Whiteboard Technical Derivations  

---

## 1. The One-Minute Elevator Pitch

> *"Users seeking forgotten entities frequently recall episodic fragments and descriptive attributes while forgetting the formal title—a phenomenon known as Tip-of-the-Tongue (ToT). In this work, we investigate whether LLM-guided query transformation and neural reranking can bridge the severe vocabulary mismatch inherent in ToT queries.*  
> *Across 536 TREC-ToT 2025 Dev3 queries over a 6.4-million document Wikipedia collection, we systematically evaluate nine controlled configurations spanning lexical BM25, bi-encoder dense reranking, clue decomposition, guarded keyword reformulation, and cross-encoder reranking.*  
> *We discover that raw bi-encoders underperform BM25 due to semantic false positives, but guarded keyword reformulation (+24.9% nDCG@10) and clue concatenation (+0.71% Recall@1000) significantly outperform classical baselines. When fused into a two-level branch-wise architecture with top-30 cross-encoder reranking, our final system achieves **nDCG@10 = 0.0386**—a statistically significant **+26.6% improvement** ($p = 0.003$) over BM25 without ever leaking target titles or test data."*

---

## 2. Comprehensive Viva Voce Q&A (15 Core Questions)

### Q1: Why did you evaluate on Dev3 and not on the official TREC-ToT Test set?
**Answer:**  
In strict accordance with competitive IR protocol (TREC rules) and sound machine learning methodology, the official test set is completely held out without public relevance judgments (qrels) to prevent test-set overfitting and post-hoc leakage. Dev3 contains **536 natural queries** with verified ground-truth known-item qrels across multiple domains (Books, Movies, Music), providing ample statistical power ($N=536$) for bootstrap confidence intervals and paired permutation testing. We strictly separated splits: Train (50) and Dev1 (142) were used for hyperparameter exploration and prompt freezing; Dev3 was touched only for final frozen evaluation.

### Q2: Is Dev3 smaller than the corpus?
**Answer:**  
No, that conflates queries with documents. Dev3 is a **query split** (536 queries), whereas the retrieval corpus is the full 2024 Wikipedia snapshot containing **6.4 million documents** (indexed in Lucene/Anserini). Every Dev3 query searches against the entire 6.4M collection (Tier B initial retrieval), funneling down to 1,000 candidate documents for neural and fusion reranking.

### Q3: Did the LLM ever see the target answers or ground-truth titles?
**Answer:**  
**Absolutely not.** We instituted a verifiable zero-leakage protocol:
1. Prompts given to LLMs contained only the raw user query and fixed instructions. No target titles, document IDs, or qrels were ever injected into prompts or context.
2. In Plan 09, we ran `scripts/run_fairness_audit.py`, which systematically tokenized all prompt inputs and outputs across all 536 queries and asserted zero title substring leakage (`assert_fair: PASS` across all runs).
3. The LLM acts purely as a query transformer and clue extractor, completely blind to candidate documents and labels.

### Q4: What exactly is Reciprocal Rank Fusion (RRF) and why did you use it?
**Answer:**  
RRF is an unsupervised rank aggregation method introduced by Cormack et al. (SIGIR 2009). The score of document $d$ across a set of rankers $R$ is:
$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$
where $k$ is a smoothing constant (standard $k=60$).  
**Why RRF?** BM25 produces unbounded lexical IDF scores ($\in [0, 40]$), while bi-encoders output cosine similarities ($\in [-1, 1]$). Calibrating and normalizing these heterogeneous score distributions (e.g., via min-max or z-score) requires held-out validation data and is prone to distortion from outlier scores. RRF relies purely on ordinal rank positions, making it robust, parameter-light, and scale-invariant.

### Q5: Why can't a Cross-Encoder run over the whole 6.4 million document corpus?
**Answer:**  
Computational complexity. A bi-encoder embeds queries and documents independently:
$$\text{Score}(q, d) = \langle \mathbf{e}_q, \mathbf{e}_d \rangle$$
Document embeddings $\mathbf{e}_d$ are pre-computed offline, reducing retrieval to an inner product search via FAISS in milliseconds.  
A cross-encoder concatenates the query and document $[q; d]$ and feeds them through all $L$ transformer self-attention layers simultaneously:
$$\text{Attention Complexity} = \mathcal{O}((len(q) + len(d))^2)$$
Running a cross-encoder across 6.4 million documents for each query would require over 6.4 million forward passes of a multi-layer transformer per query, taking hours per search. Cross-encoders are strictly suitable as second-stage rerankers over top-30 to top-100 candidates.

### Q6: Why are nDCG@10 and MRR the primary metrics instead of MAP or Recall@1000?
**Answer:**  
ToT is a **known-item search task** where exactly one target document is relevant ($R=1$).
- **MAP** (Mean Average Precision) assumes multiple relevant documents and measures precision at each relevant recall point; when $R=1$, MAP collapses mathematically to Reciprocal Rank (RR).
- **nDCG@10** evaluates whether the single relevant item appears near the top of the first results page (discounts drop as $\frac{1}{\log_2(rank + 1)}$).
- **Recall@1000** measures candidate funnel capacity (whether the item is retrieved at all in stage 1), but cannot evaluate user experience. A system with high R@1000 but the target at rank 850 is useless to a user. Therefore, nDCG@10 and MRR reflect true search utility.

### Q7: Is this project presenting a brand-new retrieval architecture?
**Answer:**  
No. As declared in our Scope boundaries (§1 and §22), this is a rigorous **empirical integration and evaluation study**. We synthesized and adapted two landmark approaches—Lin et al. (EMNLP 2023) for clue decomposition and Mukhopadhyay et al. (ECIR 2026) for guarded keyword reformulation—and conducted the first head-to-head controlled comparison across 9 distinct configurations under identical corpus constraints.

### Q8: How does your clue decomposition differ from Lin et al. (2023)?
**Answer:**  
Lin et al. focused on Goodreads books, extracting book-specific fields (*protagonist, author, setting, era*) and training specialized supervised "expert" retrievers for each field.  
In contrast, our work:
1. Operates open-domain across Wikipedia (books, movies, music, history).
2. Uses general facets: *Category, Description, People, Place, Time, Distinctive Details*.
3. Uses zero-shot prompting with standard off-the-shelf BM25 and dense retrievers instead of training separate neural models.
4. Uses parameter-free RRF instead of learned linear combination weights.
5. Focuses on strict extractive prompting rather than hallucinated predictive prompting.

### Q9: Lin et al. found predictive prompting better; why did you use extractive as primary?
**Answer:**  
Predictive prompting instructs the LLM to speculate or generate facts not explicitly stated in the query. While occasionally guessing a true actor or author, it frequently hallucinates erroneous proper nouns (e.g., guessing Tom Hanks for a movie starring Kevin Bacon), which injects catastrophic false positives into BM25 and dense indices. In accordance with Scope §7, our primary system enforces strict extractive prompting (only extracting clues substantiated by the user's text).

### Q10: Why does keyword reformulation help BM25 more than dense retrieval?
**Answer:**  
- **BM25** is an exact keyword matching algorithm. Raw ToT queries contain conversational filler ("I can't remember the name but...", "it was kind of like..."). Reformulation filters out this lexical noise and injects targeted synonyms, directly boosting BM25's TF-IDF match.
- **Dense bi-encoders** (like E5) are pre-trained on natural language pairs (sentences/passages). Keyword-style lists ("dystopian novel fireman burning books") lack syntactic grammar and natural discourse structure, degrading dense embedding quality. Our empirical results confirm this: E6 (BM25 rewrite) gained +24.9% nDCG@10, while E7-R (Dense rewrite) suffered a -42.0% penalty.

### Q11: Why don't your absolute scores match Mukhopadhyay et al. (2026)?
**Answer:**  
The 2026 paper operated under an enterprise compute budget: two NVIDIA RTX 6000 Ada GPUs (96 GB VRAM), 500 GB system RAM, full-corpus dense vector index, proprietary GPT-4o API rewrites, and monoT5 / RankLLM listwise rerankers.  
Our study was engineered for reproducible academic execution under Google Colab free-tier hardware (16 GB T4 GPU, 12 GB RAM), using Tier B candidate reranking, open-source E5-Small, and MiniLM-L6 cross-encoders. We use the 2026 paper for relative trend sanity checks (e.g., lexical dominance, PRF failure), not absolute score matching.

### Q12: Did Pseudo-Relevance Feedback (PRF / RM3) help?
**Answer:**  
No. Consistent with the findings of Mukhopadhyay et al. (2026), PRF fails dramatically on ToT queries. Classical PRF assumes that the top-k retrieved documents in the initial round are relevant and expands the query using their frequent terms. In ToT, the top-k initial documents are almost always false positives due to vocabulary mismatch. Expanding the query with terms from false positives causes topic drift and pollutes downstream retrieval.

### Q13: What would you prioritize if given 10x more compute?
**Answer:**  
1. **Full-Corpus Dense Indexing:** Indexing all 6.4 million Wikipedia documents with 768-dim embeddings in a multi-GPU FAISS cluster, lifting the Tier B candidate funnel ceiling.
2. **Listwise LLM Reranking:** Running RankLLM (e.g., Llama-3-70B) over top-20 candidates.
3. **Iterative / Agentic Clarification:** Implementing interactive clarification loops where the system asks the user discriminating questions based on candidate cluster ambiguities.

### Q14: What failed most frequently in your error analysis?
**Answer:**  
Across 94 rule-stratified queries:
- **Vocabulary Mismatch (67.0%):** The user's descriptive vocabulary shared zero unigrams with the Wikipedia page text.
- **Semantic False Positives (23.4%):** Dense models retrieved topically adjacent landmark entities (e.g., *Star Wars* instead of an obscure 1980s space opera).
- **False Memory (7.4%):** The user recalled contradictory or incorrect plot facts.

### Q15: How do you know your observed differences are not statistical noise?
**Answer:**  
Every comparison was evaluated with **paired non-parametric permutation tests** and **5,000 bootstrap resamples** (reported in `analysis/planned_comparisons.csv`). Furthermore, we applied **Holm–Bonferroni correction** across our 11 planned hypotheses to control the family-wise error rate. The E1 vs. E9 gain ($\Delta = +0.0081$, $p = 0.003$) and E4 vs. E4c gain ($\Delta = +0.0152$, $p < 0.001$) remain statistically significant at $\alpha = 0.05$ after correction.

---

## 3. Whiteboard Technical Derivations

Examiners frequently ask for whiteboard derivations. Be prepared to derive the following four models from first principles:

### Derivation 1: BM25 Scoring Function (Module 2)
For a query $q = (t_1, t_2, \dots, t_m)$ and document $d$:
$$\text{Score}_{\text{BM25}}(q, d) = \sum_{i=1}^m \text{IDF}(t_i) \cdot \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

- **IDF Component (Probabilistic Lucene formulation):**
  $$\text{IDF}(t_i) = \ln \left( 1 + \frac{N - n(t_i) + 0.5}{n(t_i) + 0.5} \right)$$
  where $N$ is total documents in the collection, $n(t_i)$ is document frequency of term $t_i$.
- **Term Frequency Saturation:** Controlled by $k_1 \ge 0$ (default $0.9$ in our tuned grid). As $f(t_i, d) \to \infty$, the TF weight approaches $(k_1 + 1)$, preventing high term repetition in spam documents from dominating.
- **Document Length Normalization:** Controlled by $b \in [0, 1]$ (default $0.4$). When $b=1$, full normalization penalizes long documents linearly; when $b=0$, length normalization is disabled.

### Derivation 2: Reciprocal Rank Fusion (RRF) (Module 8)
Given rank positions $r_m(d)$ of document $d$ across $M$ distinct retrieval systems:
$$\text{RRF\_Score}(d) = \sum_{m=1}^M \frac{1}{k + r_m(d)}$$
- **Rank Indexing:** $r_m(d) \in \{1, 2, \dots, K\}$. If document $d$ is unretrieved by system $m$, $r_m(d) = \infty$, so its reciprocal contribution is 0.
- **Smoothing Parameter $k$:** $k=60$ prevents top-ranked items in a single run from dominating.
  - At $k=60$: Rank 1 gives $\frac{1}{61} \approx 0.01639$; Rank 2 gives $\frac{1}{62} \approx 0.01613$. The slope is smooth, rewarding consensus across multiple rankers over an outlier rank in one ranker.

### Derivation 3: Bi-Encoder vs. Cross-Encoder Architecture (Module 7)

```
Bi-Encoder (Dual Encoder):
Query q  -----> [ Transformer Enc ] -----> Vector e_q \
                                                        --> Similarity: <e_q, e_d> = cos(theta)
Doc d    -----> [ Transformer Enc ] -----> Vector e_d /
• Encoding is decoupled: e_d precomputed offline.
• Query inference: O(1) embedding + O(log N) MIPS vector search.
• Bottleneck: Fixed-size vector (e.g. 384 dims) must compress all semantic nuances.

Cross-Encoder (Joint Reranker):
[CLS] q [SEP] d [SEP] -----> [ Full Transformer Layers ] -----> Classification Head [Score]
• Full cross-attention between every token in q and every token in d.
• Complexity: O((|q| + |d|)^2) for every pair.
• Capability: Captures fine-grained interactions, contradictions, and exact lexical constraints.
```

### Derivation 4: Known-Item Evaluation (nDCG@10 & MRR) (Module 3)
In Tip-of-the-Tongue retrieval, exactly **one** document is relevant ($R=1$, relevance $r=1$):

1. **Mean Reciprocal Rank (MRR):**
   $$\text{RR}(q) = \begin{cases} \frac{1}{\text{rank}^*(q)} & \text{if } \text{rank}^*(q) \le K \\ 0 & \text{if } \text{target unretrieved or } \text{rank}^* > K \end{cases}$$
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

2. **Normalized Discounted Cumulative Gain at $K=10$ (nDCG@10):**
   $$\text{DCG@10} = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
   Since exactly one document is relevant ($rel = 1$), if the target is found at rank $r \le 10$:
   $$\text{DCG@10} = \frac{2^1 - 1}{\log_2(r + 1)} = \frac{1}{\log_2(r + 1)}$$
   The ideal ranking (IDCG) places the target at rank 1:
   $$\text{IDCG@10} = \frac{1}{\log_2(1 + 1)} = \frac{1}{\log_2(2)} = 1.0$$
   Therefore, for known-item retrieval:
   $$\text{nDCG@10}(q) = \begin{cases} \frac{1}{\log_2(r^*(q) + 1)} & \text{if } r^*(q) \le 10 \\ 0 & \text{otherwise} \end{cases}$$
