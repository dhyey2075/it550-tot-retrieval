# Limitations and Threats to Validity (Scope §18, Plan 09 §6)

Every empirical study in Information Retrieval operates under constraints. This document details the key threats to validity, methodological caveats, and scope limitations of our LLM-enhanced tip-of-the-tongue (ToT) retrieval investigation.

---

### 1. Corpus Setting & Architecture (Tier B)
Due to computational constraints on local and Colab environments (prebuilt dense vector indices exceed 18 GB and require distributed GPU infrastructure), we operate under **Tier B (`dense_rerank_bm25`)**. Dense retrieval (`e5-small`) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) operate exclusively as rerankers over top-$K$ candidates retrieved by full-corpus BM25 (6.4M documents). Consequently, candidates missed entirely by BM25 are unreclaimable by downstream dense or cross-encoder stages.

### 2. Document Truncation
Both dense and cross-encoder models truncate Wikipedia documents to leading characters (lead 1,000 characters for dense reranking; 512 tokens for MiniLM). As noted by Mukhopadhyay et al. (2026), distinctive tip-of-the-tongue narrative clues often reside deep within the plot summary or production history sections rather than the lead paragraph, causing false negatives due to truncation.

### 3. Known-Item Cranfield Paradigm & Single Relevant Target
TREC-ToT qrels are strictly binary with exactly one relevant Wikipedia document per query ($n_{rel}=1$). In film and literature queries, near-duplicates, remakes (e.g., 1950 vs. 1980 versions), book adaptations, and sequels frequently match the user's narrative description closely but receive a strict relevance score of 0. This strictly penalizes systems that retrieve closely related alternative entities.

### 4. Small Development Tuning Sample
Hyperparameters (BM25 $k_1=0.9, b=0.4$; RRF $k=60$; Cross-Encoder depth $K=30$) were tuned and frozen on the Dev1 split ($n=142$). While split discipline was strictly maintained (Test split was never touched; Dev3 reserved strictly for held-out evaluation), small development sample sizes carry statistical variance.

### 5. Domain Shift & Coverage
Development queries are predominantly narrative descriptions of movies and television shows. While Dev3 and Test splits incorporate broader domains (landmarks, historical events, celebrities, and 50+ diverse categories), prompt schemas and clue extraction strategies were developed on narrative media queries, creating a potential domain shift.

### 6. LLM Contamination & Pretraining Leakage
LLM backends (GPT-4o-mini, Gemini 2.0 Flash) have extensive pretraining corpora covering Wikipedia and public internet forums (including Reddit r/tipofmytongue). While our strict prompting guardrails forbid guessing Wikipedia titles or predicting target names, LLM contextual understanding is inherently informed by pretraining exposure to ToT discussions.

### 7. Prompt Sensitivity
A single frozen prompt was employed for clue decomposition and query reformulation. Slight variations in prompt wording, few-shot examples, or instruction emphasis could alter the distribution of extracted clues and downstream retrieval efficacy.

### 8. LLM Drift and Nondeterminism
Commercial LLM APIs undergo silent updates and non-deterministic floating-point scheduling. We mitigated this by setting temperature $\tau=0$, pinning exact model snapshots, caching all responses by cryptographic SHA-256 key, and enforcing an offline fallback stub for deterministic local execution.

### 9. Single Retrieval Trajectory
Each experiment was evaluated over a single deterministic execution pass per query rather than repeated stochastic sampling. Caching ensures complete reproducibility, but prompt variations were not sampled across multiple random seeds.

### 10. Qualitative Annotator Subjectivity
Error taxonomy categorization (e.g., distinguishing between "Vocabulary mismatch", "Semantic false positive", and "Entity ambiguity") involves human qualitative judgment. Standardized operational decision rules (Plan 09 §4.3) were followed to minimize labeling drift.

### 11. Dev Split Difficulty vs. Held-Out Test Split
Prior literature indicates that Dev splits exhibit higher baseline performance than the official TREC test set (e.g., Mukhopadhyay et al. report BM25 nDCG@10 of 0.337 on Dev3 versus 0.122 on Test). Scores on development splits should not be extrapolated as absolute performance predictors on unreleased benchmarks.

### 12. Adaptation Rather than Exact Replication
Our architecture adapts the core conceptual findings of Lin et al. (2023) and Mukhopadhyay et al. (2026) to the TREC-ToT 2025 task by employing:
- A 7-facet extractive schema (category, description, people, place, time, details, names) rather than book-specific metadata.
- Pretrained zero-shot retrievers (`e5-small`, `MiniLM-L-6-v2`) rather than task-fine-tuned domain retrievers.
- Reciprocal Rank Fusion (RRF) rather than learned linear score combination weights.
