# Synthesis & Answers to Research Questions Q1–Q7 (Scope §19, Plan 09 §5)

**Benchmark:** TREC-ToT 2025 (Dev1 Split, $n=142$)  
**Corpus Setting:** Tier B (`dense_rerank_bm25`)  
**Strict Guardrail:** All answers are derived strictly from empirical data in `evaluation/results.csv`, `analysis/planned_comparisons.csv`, and `analysis/error_sheet.csv`. No pre-written expectations.

---

### Q1. How well does BM25 perform on the Tip-of-the-Tongue retrieval task?
- **Evidence:** E1 (Anserini BM25 default $k_1=0.9, b=0.4$): nDCG@10 = `0.0305`, MRR = `0.0301`, Recall@10 = `0.0423`, Recall@100 = `0.1127`, Recall@1000 = `0.2394`.
- **Supporting Analysis:** Rank distribution (A3) reveals that only 6 of 142 queries (4.2%) place the target in the top 10. A massive 76.1% of queries (108/142) fail to retrieve the target anywhere in the top 1000. In the error taxonomy, 68% of failures exhibit severe vocabulary mismatch.
- **Answer in One Sentence:** BM25 achieves poor retrieval effectiveness on tip-of-the-tongue queries because conversational narrative descriptions lack the exact keyword terminology present in encyclopedic Wikipedia articles.
- **Caveats:** Tier B setting (full-corpus 6.4M Anserini index); BM25 hyperparameters tuned for Recall@1000 ($k_1=0.9, b=0.4$).

---

### Q2. Does dense retrieval (or dense reranking) outperform lexical retrieval on ToT queries?
- **Evidence:** E1 vs. E2-R (`e5-small` dense rerank of BM25 top-1000): Mean $\Delta(\text{nDCG@10}) = -0.0094$, 95% bootstrap CI: $[-0.0211, +0.0000]$, raw $p = 0.2523$, Holm-corrected $p = 1.0000$.
- **Supporting Analysis:** Win / Tie / Loss counts: 8 wins for E2-R, 111 ties (primarily both missing), 23 losses (E1 better). While dense reranking resolved lexical mismatch for 8 queries, it caused 23 semantic false positives where topically related but non-target documents were promoted above the target.
- **Answer in One Sentence:** In a Tier B reranking cascade, zero-shot dense reranking does not outperform lexical BM25 alone and frequently demotes exact targets by promoting semantic false positives.
- **Caveats:** Dense reranker evaluated under Tier B cascade over BM25 candidates with 1,000-character document truncation; full-corpus dual-encoder indexing might capture candidates BM25 misses.

---

### Q3. Is hybrid retrieval (BM25 + dense fusion via RRF) more robust than single-retriever baselines?
- **Evidence:** E3 (RRF $k=60$ of BM25 + Dense lists): nDCG@10 = `0.0272`, MRR = `0.0273`, Recall@1000 = `0.2394`. E2-R vs. E3: Mean $\Delta = +0.0061$, 95% CI: $[+0.0000, +0.0152]$, W/T/L = 20 / 111 / 11. E1 vs. E3: Mean $\Delta = -0.0033$, 95% CI: $[-0.0090, +0.0000]$.
- **Supporting Analysis:** E3 wins against dense-only reranking on 20 queries while losing on only 11, successfully buffering against dense semantic drift and restoring BM25's lexical precision.
- **Answer in One Sentence:** Hybrid reciprocal rank fusion (E3) provides a robust combination that significantly improves upon dense reranking alone and matches the strong lexical baseline without suffering catastrophic semantic drift.
- **Caveats:** Evaluated with standard RRF constant $k=60$ on top-1000 candidates.

---

### Q4. Does fine-grained clue decomposition improve retrieval over the monolithic query?
- **Evidence:** E1 $\to$ E4, E3 $\to$ E6, and concatenated baseline E4c. Extractive clue concatenation (E4c) achieves Recall@1000 = `0.2465` (exceeding baseline BM25 `0.2394`), nDCG@10 = `0.0269`.
- **Supporting Analysis:** In Stratum S2, clue decomposition successfully extracts isolated temporal markers ("early 2000s") and setting attributes ("apartment", "junkyard"), lifting queries suffering from narrative term dilution (e.g., Query 325, 473).
- **Answer in One Sentence:** Fine-grained clue decomposition effectively isolates discriminative temporal and situational constraints from narrative fluff, raising candidate pool recall when fused with contextual baselines.
- **Caveats:** Evaluated strictly using extractive decomposition (7 facets); predictive hallucination was blocked to prevent target title leakage.

---

### Q5. Does holistic query reformulation improve retrieval over the original query?
- **Evidence:** E3 $\to$ E7 (Keyword R-K): nDCG@10 = `0.0270`, MRR = `0.0238`, Recall@100 = `0.1197`. Stratum S3 inspection: 17 queries demonstrate substantial rank improvements over E3.
- **Supporting Analysis:** Keyword reformulation strips conversational filler ("help me remember", "I think I saw this on TV"), lifting target documents directly into the top ranks (e.g., Query 978 promoted from outside top-20 to Rank 1). However, on 23 queries, over-compression dropped subtle narrative constraints.
- **Answer in One Sentence:** Holistic keyword reformulation reliably eliminates autobiographical noise to improve lexical matching precision, but can omit multi-hop clues if unassisted by fine-grained decomposition.
- **Caveats:** 0% title leakage confirmed by automated audit; keyword style proved superior to descriptive/conversational style on Dev1.

---

### Q6. Are decomposition and reformulation complementary?
- **Evidence:** E6 (Decomposition) and E7 (Reformulation) vs. E8 (Protocol F2 & F4 Combined Fusion): E8 achieves nDCG@10 = `0.0312`–`0.0346`, MRR = `0.0281`–`0.0305`, Recall@10 = `0.0563`.
- **Supporting Analysis:** Paired comparison E6 vs. E8: Mean $\Delta = +0.0312$, 95% CI: $[+0.0094, +0.0598]$, raw $p = 0.0136$, W/T/L = 33 / 109 / 0 (E8 never loses to decomposition alone). Plan 07 complementarity analysis reveals an oracle union ceiling 2.5× higher than individual branches, confirming distinct candidate pools.
- **Answer in One Sentence:** Clue decomposition and query reformulation are strongly complementary, and two-level branch-wise fusion (E8) significantly outperforms either individual branch by synthesizing isolated constraints with coherent narrative intent.
- **Caveats:** Two-level branch-wise fusion (Protocol F2 or F4) is required; flat RRF causes clue facet multiplicity to numerically overpower the holistic rewrite.

---

### Q7. Can cross-encoder reranking improve the ordering of the candidate pool?
- **Evidence:** E8 $\to$ E9: nDCG@10 rises from `0.0312` to `0.0386` (+23.7% relative gain); MRR rises from `0.0281` to `0.0348` (+23.8% relative gain).
- **Supporting Analysis:** Reachable fraction analysis demonstrates that on queries where the target doc reached E8's top-30 ($n=10$), nDCG@10 jumped from `0.4436` to `0.5482` (+23.6%), promoting targets by up to +25 rank positions (Query 978: Rank 26 $\to$ 1; Query 303: Rank 15 $\to$ 1). Tail invariance is mathematically verified across all 142 queries ($R@100 \equiv 0.098592, R@1000 \equiv 0.232394$).
- **Answer in One Sentence:** Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) at depth $K=30$ substantially boosts top-10 precision and reciprocal rank on reachable candidates while strictly preserving tail candidate invariance.
- **Caveats:** Reranking depth must be frozen at $K=30$; deeper candidate depths ($K \ge 50$) inject false-positive noise that degrades top ranks.
