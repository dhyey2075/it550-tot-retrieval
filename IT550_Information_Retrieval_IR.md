# IT550: Information Retrieval (IR)

## Course Information

- **Instructor:** Sandip Modha, 2201, Faculty Block-2
- **Email:** Sandip_modha@dau.ac.in
- **Course Title:** Information Retrieval
- **Course Code:** IT550
- **Credit Structure (L-T-P-Cr):** 3-0-2-4
  - 3 hours lecture
  - 2 hours lab per week
- **Program:** M.Tech (ICT) III ML Elective; Technical elective for B.Tech VII (ICT and CS)

### Prerequisites / Desired Skill Set

- Linear algebra
- Basic probability and statistics
- Basic algorithms and data structures
- Python programming

---

## Course Description

This course focuses on the design and development of Information Retrieval systems.

The first part covers traditional Information Retrieval systems, including:

- Preprocessing
- Indexing
- Ranking
- Retrieval models
- Evaluation

The course then explores applications of Information Retrieval in:

- Text classification
- Text clustering
- Web search
- Link analysis algorithms such as PageRank and HITS

The second part covers modern Neural Information Retrieval techniques, including:

- Learning-to-rank
- Semantic search
- Dense retrieval
- Transformer-based retrieval models
- FAISS
- Vector databases

The course also explores applications of LLMs in modern IR, including:

- Retrieval-Augmented Generation (RAG)
- Hybrid retrieval
- Agentic retrieval
- Multimodal retrieval
- Conversational search

---

## Evaluation Scheme (Tentative)

| Component | Weightage |
|---|---:|
| Attendance and class participation | 5% |
| In-sem exam | 15% |
| Final End-Sem examination | 25% |
| 8–10 Lab Assignments | 20% |
| Course project | 35% |
| **Total** | **100%** |

---

## Suggested Textbooks / References

1. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008. ISBN-13: 978-0521865715
2. W. Bruce Croft, Donald Metzler, and Trevor Strohman. *Search Engines: Information Retrieval in Practice*. Pearson, 2010. ISBN: 9780131364899, 0131364898
3. Bhaskar Mitra and Nick Craswell. *An Introduction to Neural Information Retrieval*. Now Publishers Inc.
4. Jure Leskovec, Anand Rajaraman, Jeffrey D. Ullman. *Mining of Massive Datasets*. Cambridge University Press, 2011. ISBN: 978-1107077232

---

## Course Outcomes

After successful completion of the course, the student will be able to:

1. Understand basic concepts such as document indexing, query processing, similarity measures, and ranking techniques used in search-system design.
2. Evaluate the effectiveness of classical IR ranking models such as Vector Space Models, BM25, and Language Models for retrieval using:
   - Precision
   - Recall
   - F-measure
   - MAP
   - MRR
   - NDCG
3. Apply relevance feedback, query expansion, text classification, and clustering techniques to improve retrieval effectiveness.
4. Design and develop neural information retrieval systems using:
   - Learning-to-rank methods
   - Semantic search
   - Dense retrieval
   - Transformer-based retrieval models
   - Vector search techniques
5. Design and develop LLM-based Information Retrieval systems.

---

## Program Outcomes

The course document provides the following Program Outcome mapping:

| Program Outcome | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mapping | X | X | X | X | X |  |  | X | X | X |  | X |

---

# Detailed Course Content

**Total: 36 Lectures**

## Module 1 — Basics of IR

**Number of Lectures:** 5

Topics:

- Introduction to Information Retrieval
- Document collections and corpora
- Information needs
- Queries and relevance
- Handling unstructured text
- Preprocessing pipeline:
  - Tokenization
  - Stopword removal
  - Stemming
  - Lemmatization
- Inverted indexing:
  - Posting lists
  - Index construction

---

## Module 2 — Term Weighting and Classical IR Models

**Number of Lectures:** 8

Topics:

- Boolean models
- Query formulation
- Ranked retrieval
- TF-IDF (term frequency / inverse document frequency) weighting
- Cosine similarity
- Vector-space retrieval models
- Probabilistic Retrieval Model
- Binary Independence Model
- BM25 ranking function
- Language Models for IR
- Query Likelihood Model
- Smoothing techniques

---

## Module 3 — Experimental Evaluation of IR Systems

**Number of Lectures:** 3

Topics:

- Relevance judgments
- Cranfield paradigm
- Test collections
- Evaluation metrics:
  - Recall
  - Precision
  - F-measure
  - NDCG
  - MAP
  - MRR
  - Precision@K

---

## Module 4 — Relevance Feedback and Query Expansion

**Number of Lectures:** 2

Topics:

- Relevance feedback
- Rocchio algorithm for relevance feedback
- Pseudo relevance feedback
- Query expansion techniques

---

## Module 5 — Text Classification and Clustering

**Number of Lectures:** 2

Topics:

- The text classification problem
- Naive Bayes text classification
- Text clustering:
  - Flat clustering
  - Hierarchical clustering

---

## Module 6 — Web Search

**Number of Lectures:** 3

Topics:

- Search engine architecture
- Crawling
- Web as a graph
- Link analysis
- PageRank algorithm
- HITS algorithm

---

## Module 7 — Neural Information Retrieval

**Number of Lectures:** 9

Topics:

### Learning to Rank

- Pointwise ranking
- Pairwise ranking
- Listwise ranking
- LambdaMART

### Semantic and Neural Retrieval

- Latent Semantic Indexing
- Word embeddings
- Word2Vec
- Semantic search
- Dense retrieval

### Transformer-Based Retrieval

- Bi-Encoder
- Cross-Encoder
- Retrieval re-ranking

### Approximate Nearest Neighbor Search

- Approximate Nearest Neighbor (ANN) search
- FAISS
- Vector databases

---

## Module 8 — Information Retrieval in the LLM Era

**Number of Lectures:** 4

Topics:

- Limitations of classical retrieval systems
- Limitations of neural retrieval systems
- Retrieval-Augmented Generation (RAG):
  - Motivation
  - Architecture
- Hybrid retrieval:
  - Sparse retrieval
  - Dense retrieval
- Emerging trends:
  - Agentic retrieval
  - Multimodal retrieval
  - Conversational search

---

## Module-wise Lecture Summary

| Module | Title | Lectures |
|---:|---|---:|
| 1 | Basics of IR | 5 |
| 2 | Term Weighting and Classical IR Models | 8 |
| 3 | Experimental Evaluation of IR System | 3 |
| 4 | Relevance Feedback and Query Expansion | 2 |
| 5 | Text Classification and Clustering | 2 |
| 6 | Web Search | 3 |
| 7 | Neural Information Retrieval | 9 |
| 8 | Information Retrieval in the LLM Era | 4 |
| **Total** | | **36** |

---

## Course Structure at a Glance

### Classical Information Retrieval

1. Basics of IR
2. Preprocessing
3. Inverted indexing
4. Boolean and vector-space models
5. TF-IDF
6. Probabilistic retrieval
7. BM25
8. Language models
9. IR evaluation
10. Relevance feedback
11. Query expansion
12. Text classification and clustering
13. Web search
14. PageRank and HITS

### Neural Information Retrieval

1. Learning-to-rank
2. Latent Semantic Indexing
3. Word2Vec and word embeddings
4. Semantic search
5. Dense retrieval
6. Bi-Encoder and Cross-Encoder models
7. Retrieval re-ranking
8. ANN search
9. FAISS
10. Vector databases

### LLM-Based Information Retrieval

1. Limitations of classical and neural retrieval
2. RAG
3. Hybrid retrieval
4. Agentic retrieval
5. Multimodal retrieval
6. Conversational search
