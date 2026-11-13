# Project Scope
## LLM-Enhanced Tip-of-the-Tongue Information Retrieval using Query Decomposition, Reformulation and Hybrid Ranking

**Course:** IT550 – Information Retrieval  
**Project Type:** TREC-ToT based Information Retrieval project  
**Primary Benchmark:** TREC-ToT 2025 Dev3  
**Implementation Environment:** Google Colab Free + Python + LLM API

---

## 1. Project Overview

### 1.1 Problem

Tip-of-the-Tongue (ToT) retrieval is a **known-item retrieval** problem in which a user remembers an item but cannot recall its exact name. Instead of providing the entity name, the user describes it using incomplete, indirect, noisy, or partially remembered clues.

For example:

> "A science-fiction movie where people enter dreams and try to plant an idea in someone's mind."

The user knows the target but cannot formulate a precise query containing its name. This creates a difficult Information Retrieval problem because the words used in the query may have little lexical overlap with the target document.

The project will investigate whether **LLM-based query understanding before retrieval** can improve the retrieval of the correct Wikipedia page.

---

## 2. Objective

The primary objective is:

> **To improve Tip-of-the-Tongue known-item retrieval by transforming vague user queries using LLM-based query decomposition and query reformulation, and combining these transformations with conventional and neural retrieval techniques.**

The project will specifically study:

1. How well conventional retrieval methods such as BM25 perform on ToT queries.
2. Whether dense semantic retrieval performs better when lexical overlap is weak.
3. Whether combining BM25 and dense retrieval improves robustness.
4. Whether decomposing a complex ToT query into independent clues improves retrieval.
5. Whether LLM-based query reformulation makes vague queries more retrieval-friendly.
6. Whether decomposition and reformulation provide complementary benefits.
7. Whether neural cross-encoder reranking can further improve the final ranking.
8. Which approach works best on the TREC-ToT Dev3 benchmark.
9. What types of ToT queries remain difficult and why.

The project is primarily an **experimental integration and evaluation study**. We are not claiming to invent a completely new retrieval architecture. The contribution is to adapt and combine ideas from existing ToT research, evaluate them under a common experimental setup, and analyze their effectiveness on the TREC-ToT benchmark.

---

# 3. Dataset

## 3.1 TREC-ToT 2025

The project will use the **TREC Tip-of-the-Tongue (ToT) 2025 benchmark**.

Official TREC-ToT guidelines:

https://trec-tot.github.io/guidelines.html

Official dataset on Zenodo:

https://zenodo.org/records/18235970

The TREC-ToT 2025 task is an open-domain known-item retrieval task. Given a ToT request, the system must return a ranked list of Wikipedia page IDs, with the correct item ranked as highly as possible.

The 2025 corpus contains approximately **6.4 million Wikipedia documents** and the compressed corpus file is approximately **9.1 GB**.

---

## 3.2 Why We Are Using Dev3

The complete TREC-ToT 2025 corpus is too large for our intended **Google Colab Free** environment to comfortably download, index, and repeatedly process from scratch.

However, the important distinction is:

> **Dev3 is not a smaller copy of the 9.1 GB corpus. Dev3 is a query/evaluation split.**

The 9.1 GB figure refers to the full TREC-ToT 2025 Wikipedia corpus.

Dev3 itself contains:

- **536 queries**
- **536 qrels entries**
- Query files of approximately **718 KB**
- Queries originating from the **2024 TREC-ToT test set**
- A mixture of **movies, landmarks, and celebrities**

Official TREC documentation lists the 2025 splits as follows:

| Split | Queries | Description |
|---|---:|---|
| Train | 143 | 2025 training queries |
| Dev1 | 142 | Based on 2023 dev set |
| Dev2 | 143 | Based on 2023 test set |
| **Dev3** | **536** | **Based on 2024 test set** |
| Test | 622 | 2025 test queries |

Source:

https://trec-tot.github.io/guidelines.html

### Why Dev3 is appropriate

Dev3 is selected as the primary benchmark because:

1. **It is substantially larger than Dev1 and Dev2**, giving a more meaningful evaluation set.
2. It contains **536 queries**, which is large enough for comparing nine retrieval configurations.
3. Unlike Train, Dev1 and Dev2, which are movie-only, **Dev3 includes movies, landmarks and celebrities**, giving our system more varied retrieval scenarios.
4. It is part of the official TREC-ToT 2025 benchmark and therefore provides standardized queries and relevance judgments.
5. It allows us to evaluate the system without needing to treat the full 9.1 GB corpus as a locally indexed dataset.
6. It is particularly suitable for studying whether LLM query understanding generalizes across different entity types.

### Important experimental rule

We will use:

- **Train:** prompt development, debugging and understanding the dataset format.
- **Dev1:** initial pilot experiments and pipeline debugging.
- **Dev3:** primary/final project evaluation.
- **Test:** kept untouched and **not used for development or tuning**.

This prevents the project from accidentally optimizing against the official test set.

---

# 4. Research Foundation

The project combines two research directions that directly address the main weakness of ToT retrieval: the mismatch between the user's vague description and the representation expected by retrieval systems.

---

## 4.1 Research Paper 1 — Query Decomposition

### Paper

**Decomposing Complex Queries for Tip-of-the-tongue Retrieval**

Kevin Lin, Kyle Lo, Joseph E. Gonzalez, Dan Klein  
Findings of EMNLP 2023

Paper:

https://aclanthology.org/2023.findings-emnlp.367.pdf

GitHub:

https://github.com/kl2806/whatsthatbook

### Core idea

The paper observes that ToT queries often contain multiple different clues.

Instead of treating the entire query as one retrieval unit, an LLM decomposes it into individual clues.

For example:

**Original ToT query**

> "A science-fiction movie where people enter dreams and try to plant an idea in someone's mind."

Possible decomposition:

- Clue 1: science-fiction movie involving dreams
- Clue 2: people enter or manipulate another person's dreams
- Clue 3: an idea is implanted in someone's subconscious

Each clue can then be used as an independent retrieval query.

The results from different clues are finally combined.

### Why this helps

A single long query may contain several concepts that do not all occur together in the target document.

Decomposition allows the retrieval system to ask several more focused questions:

```text
Original Query
      |
      v
   LLM
      |
      +----> Clue 1 ----> Retriever ----+
      |                                  |
      +----> Clue 2 ----> Retriever ----+----> Fusion
      |                                  |
      +----> Clue 3 ----> Retriever ----+
```

The original paper used specialized retrievers for different clue types. Our project will adapt the core decomposition idea to the TREC-ToT Wikipedia setting, where the primary retrieval signals will be textual BM25 and dense semantic retrieval.

### What we will investigate

We will test whether:

> **retrieving independently from multiple LLM-generated clues produces better candidate documents than retrieving directly from the original ToT query.**

---

# 5. Research Foundation — Query Reformulation

## 5.1 Research Paper 2

### Paper

**Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for Tip-of-the-Tongue Known-Item Retrieval**

Debayan Mukhopadhyay, Utshab Kumar Ghosh, Shubham Chatterjee  
2026

Paper:

https://arxiv.org/pdf/2602.10321

TREC paper:

https://trec.nist.gov/pubs/trec34/papers/mst.tot.pdf

GitHub:

https://github.com/debayan1405/TREC-TOT-2025

### Core idea

Instead of decomposing a query into several clues, the system can ask an LLM to **rewrite the original vague query into a clearer, more retrieval-oriented query**.

Example:

```text
Original:
"A movie where people go inside dreams and put thoughts
into someone's mind."

             |
             v
            LLM
             |
             v

Reformulated:
"Science-fiction film involving shared dream environments,
dream manipulation, and implanting ideas in another person's
subconscious."
```

The reformulated query is then passed to retrieval systems.

The 2026 work reports that LLM-based reformulation can substantially improve retrieval recall, followed by further gains from multi-stage reranking.

### Why reformulation is different from decomposition

**Decomposition** asks:

> "What are the individual clues contained in this query?"

**Reformulation** asks:

> "How can I rewrite this query so that a retrieval system can understand the information need more effectively?"

Therefore, they represent two different ways of using an LLM before retrieval.

---

# 6. Proposed Retrieval Pipeline

The project will compare increasingly sophisticated retrieval pipelines.

## 6.1 Conventional Retrieval

### BM25

BM25 will be our primary lexical baseline.

It uses:

- term frequency
- inverse document frequency
- document length normalization

BM25 is strong when query terms overlap with the relevant document, but ToT queries can suffer from vocabulary mismatch.

---

## 6.2 Dense Retrieval

A dense retrieval model will represent queries and documents as vectors.

The system will retrieve documents using semantic similarity.

This can help when:

```text
Query wording != Document wording
```

but the underlying meaning is similar.

However, dense retrieval can also introduce semantic false positives and may struggle with rare entities or exact identifiers.

---

## 6.3 Hybrid Retrieval

We will combine:

- BM25
- Dense retrieval

using **Reciprocal Rank Fusion (RRF)**.

Conceptually:

```text
                Original Query
                 /          \
                /            \
             BM25          Dense
               \              /
                \            /
                 \          /
                   RRF Fusion
                       |
                       v
                 Final Ranking
```

RRF combines rankings instead of requiring the scores from different retrieval systems to be directly comparable.

---

# 7. LLM Query Decomposition

The decomposition branch will follow this pipeline:

```text
Original ToT Query
        |
        v
      LLM
        |
        v
  Multiple Clues
   /     |      \
  /      |       \
BM25   BM25    BM25
  \      |       /
   \     |      /
      RRF Fusion
          |
          v
   Candidate Ranking
```

We will additionally test dense retrieval for each clue and a hybrid BM25+dense setup.

### Design constraints

The LLM should:

- produce a small number of meaningful clues;
- preserve information from the original query;
- avoid introducing unsupported facts;
- avoid revealing the target entity name;
- produce retrieval-friendly textual clues;
- avoid unnecessary explanations.

The exact prompt will be documented and fixed before final Dev3 evaluation.

---

# 8. LLM Query Reformulation

The reformulation branch will follow:

```text
Original ToT Query
        |
        v
       LLM
        |
        v
Reformulated Query
        |
        +------> BM25
        |
        +------> Dense
        |
        v
      RRF
        |
        v
 Candidate Documents
```

The reformulation should:

- preserve the original information need;
- clarify ambiguous wording where possible;
- make implicit useful concepts explicit;
- avoid hallucinated facts;
- never insert the target answer;
- remain independent of the relevance judgment.

---

# 9. Combined Decomposition + Reformulation

After evaluating the two techniques independently, we will investigate whether they provide complementary information.

The combined approach will retrieve candidates from:

1. the original/decomposed clue queries;
2. the reformulated query;
3. BM25 retrieval;
4. dense retrieval.

These candidate lists will then be fused.

Conceptually:

```text
                    Original ToT Query
                    /              \
                   /                \
            Decomposition       Reformulation
                 |                    |
            Multiple Clues       One Clear Query
             /       \               /      \
          BM25      Dense          BM25    Dense
             \       /               \      /
              \     /                 \    /
               RRF /                  RRF
                   \                  /
                    \                /
                     ---- RRF -------
                          |
                          v
                   Final Candidates
```

The exact fusion protocol will be fixed before evaluation and reported clearly.

---

# 10. Cross-Encoder Reranking

The final stage will optionally apply a cross-encoder to the top candidate documents.

Unlike a bi-encoder, a cross-encoder directly evaluates:

```text
[Query, Document]
```

together.

This can provide a more precise relevance score, but it is computationally more expensive.

Therefore, it will only be applied to a small candidate set, such as the top 50–100 documents.

Pipeline:

```text
LLM Query Processing
        |
        v
BM25 + Dense Retrieval
        |
        v
RRF Candidate Fusion
        |
        v
Top 50–100 Documents
        |
        v
Cross-Encoder
        |
        v
Final Ranking
```

---

# 11. Nine Experiments

The project will contain nine controlled experiments.

| ID | Experiment | Retrieval Configuration | Main Question |
|---|---|---|---|
| **E1** | Original Query + BM25 | BM25 | How strong is the lexical baseline? |
| **E2** | Original Query + Dense | Dense retrieval | Does semantic retrieval help with vocabulary mismatch? |
| **E3** | Original Query + Hybrid | BM25 + Dense + RRF | Does combining lexical and semantic retrieval improve robustness? |
| **E4** | Decomposed Query + BM25 | Per-clue BM25 + RRF | Does decomposition help lexical retrieval? |
| **E5** | Decomposed Query + Dense | Per-clue Dense + RRF | Does decomposition help semantic retrieval? |
| **E6** | Decomposed Query + Hybrid | Per-clue BM25 + Dense + RRF | Is decomposition stronger when combined with hybrid retrieval? |
| **E7** | Reformulated Query + Hybrid | Reformulation + BM25 + Dense + RRF | Does LLM reformulation improve retrieval? |
| **E8** | Decomposition + Reformulation + Hybrid | Combined LLM processing + BM25 + Dense + RRF | Are decomposition and reformulation complementary? |
| **E9** | E8 + Cross-Encoder | Combined pipeline + neural reranking | Can reranking further improve the final ranking? |

---

# 12. Experimental Progression

The experiments are intentionally structured as an incremental study:

```text
E1
Original + BM25
       |
       v
E2
Original + Dense
       |
       v
E3
Original + Hybrid
       |
       v
E4–E6
Decomposition
       |
       v
E7
Reformulation
       |
       v
E8
Decomposition + Reformulation
       |
       v
E9
Cross-Encoder Reranking
```

This allows us to isolate the contribution of each component instead of only reporting the performance of one final system.

---

# 13. Evaluation Metrics

The TREC-ToT task is a known-item retrieval problem where the goal is to rank the correct Wikipedia page as highly as possible.

The official TREC guidelines recommend ranking-oriented metrics suitable for a setting with one relevant document, including:

- **nDCG@10**
- **MRR**
- **Success@K**
- **Recall@K**

We will primarily report:

| Metric | Purpose |
|---|---|
| **nDCG@10** | Measures quality of the top-ranked results |
| **MRR** | Measures how early the correct document appears |
| **Recall@10** | Whether the target appears in the top 10 |
| **Recall@100** | Whether the target appears in the top 100 |
| **Recall@1000** | Whether the retrieval stage finds the target at all |
| **Success@10** | Whether at least one relevant result is retrieved in top 10 |

MRR is particularly meaningful for ToT because a user benefits greatly if the correct item appears near the top.

---

# 14. Experimental Fairness

To make the comparison meaningful, all experiments will use:

- the same Dev3 queries;
- the same relevance judgments;
- the same document collection/index;
- the same evaluation scripts;
- the same candidate depth wherever possible;
- fixed LLM prompts during final evaluation;
- cached LLM responses to avoid accidental variation;
- identical preprocessing where applicable.

No experiment will use the test set for tuning.

---

# 15. Implementation Plan

## 15.1 Environment

The project will be implemented primarily in:

- Google Colab Free
- Python
- Google Drive for caching and intermediate files

Potential libraries/tools:

- `ir_datasets`
- PyTerrier / Anserini where appropriate
- `sentence-transformers`
- Cross-Encoder models
- `ranx` / `trec_eval` for evaluation
- Gemini or OpenAI API for LLM processing
- NumPy / Pandas
- custom RRF implementation

---

## 15.2 Corpus Strategy

We will **not automatically download and build a full 6.4M-document index from scratch** in Colab.

Instead, we will first inspect the official TREC-ToT resources and available prepared indices/runs.

The implementation will prefer:

1. official/prepared retrieval indices when they are feasible;
2. existing baseline runs when appropriate for comparison;
3. efficient candidate retrieval/reranking strategies when a full local index is not practical.

If a dense full-corpus index cannot be executed within Colab Free constraints, the final report will explicitly distinguish between:

- genuine full-corpus dense retrieval; and
- dense reranking of BM25-generated candidates.

These must not be presented as the same experiment.

---

# 16. LLM Usage and Caching

LLM calls will be used only for query processing:

### Decomposition

```text
Query -> LLM -> Clue 1, Clue 2, Clue 3, ...
```

### Reformulation

```text
Query -> LLM -> Reformulated Query
```

The output for every query will be cached in JSON/JSONL format.

This provides:

- reproducibility;
- lower API cost;
- faster experimentation;
- protection against accidental repeated calls;
- consistent inputs across retrieval experiments.

A small pilot set will be used to debug prompts before running the full Dev3 evaluation.

---

# 17. Planned Development Workflow

### Phase 1 — Dataset and Baseline

1. Load TREC-ToT queries and qrels.
2. Inspect Dev3 query structure.
3. Implement/evaluate BM25.
4. Implement/evaluate dense retrieval.
5. Implement hybrid RRF.
6. Establish baseline results.

### Phase 2 — Query Decomposition

1. Design decomposition prompt.
2. Generate clues for pilot queries.
3. Inspect decomposition quality manually.
4. Run BM25 per clue.
5. Run dense retrieval per clue.
6. Fuse results.
7. Evaluate E4–E6.

### Phase 3 — Query Reformulation

1. Design reformulation prompt.
2. Generate reformulated queries.
3. Validate that target names are not introduced.
4. Run hybrid retrieval.
5. Evaluate E7.

### Phase 4 — Combined Approach

1. Define combination strategy.
2. Retrieve using decomposed clues.
3. Retrieve using reformulated query.
4. Fuse candidate lists.
5. Evaluate E8.

### Phase 5 — Reranking

1. Select top candidate documents.
2. Apply cross-encoder.
3. Produce final ranking.
4. Evaluate E9.

### Phase 6 — Analysis

1. Compare all nine experiments.
2. Analyze improvements and failures.
3. Categorize difficult queries.
4. Inspect examples where decomposition helps.
5. Inspect examples where reformulation helps.
6. Inspect examples where both fail.
7. Analyze whether hybrid retrieval is consistently better than individual retrievers.

---

# 18. Error Analysis

Performance numbers alone will not be sufficient.

We will inspect difficult queries and classify failure cases such as:

### Vocabulary mismatch

The query describes the target using words that are absent from the target document.

### Missing clue

The user's description contains an important clue that the system fails to exploit.

### Incorrect decomposition

The LLM splits the query into weak, redundant, or misleading clues.

### Hallucinated clue

The LLM introduces information that was not present in the original query.

### Over-general reformulation

The rewritten query becomes too broad and retrieves many irrelevant documents.

### Over-specific reformulation

The LLM invents details that incorrectly narrow the retrieval space.

### Semantic false positive

Dense retrieval finds documents that are semantically similar but are not the target.

### Lexical mismatch

BM25 fails because the vocabulary of the query and target document differs substantially.

### Entity ambiguity

Multiple entities satisfy part of the description.

This analysis will help explain **why** an approach succeeds or fails rather than only reporting numerical scores.

---

# 19. Expected Outcomes

The project is expected to answer the following questions:

### Q1. Does BM25 work well for ToT retrieval?

We expect BM25 to provide a strong and interpretable lexical baseline but to struggle when query wording differs substantially from document wording.

### Q2. Does dense retrieval improve semantic matching?

We will determine whether dense retrieval retrieves relevant pages missed by BM25.

### Q3. Does hybrid retrieval improve robustness?

We expect BM25 and dense retrieval to provide complementary signals.

### Q4. Does query decomposition improve retrieval?

We will test whether separating multiple clues allows the system to retrieve evidence that a single query misses.

### Q5. Does query reformulation improve retrieval?

We will measure whether an LLM-generated retrieval-friendly query improves ranking and recall.

### Q6. Are decomposition and reformulation complementary?

This is one of the central questions of the project.

### Q7. Does reranking improve the final result?

We will determine whether a cross-encoder can improve the ordering of candidates already retrieved by the earlier stages.

---

# 20. Results Table

The final report will contain a table similar to:

| Exp. | Method | nDCG@10 | MRR | R@10 | R@100 | R@1000 |
|---|---|---:|---:|---:|---:|---:|
| E1 | Original + BM25 | TBD | TBD | TBD | TBD | TBD |
| E2 | Original + Dense | TBD | TBD | TBD | TBD | TBD |
| E3 | Original + Hybrid | TBD | TBD | TBD | TBD | TBD |
| E4 | Decomposition + BM25 | TBD | TBD | TBD | TBD | TBD |
| E5 | Decomposition + Dense | TBD | TBD | TBD | TBD | TBD |
| E6 | Decomposition + Hybrid | TBD | TBD | TBD | TBD | TBD |
| E7 | Reformulation + Hybrid | TBD | TBD | TBD | TBD | TBD |
| E8 | Decomposition + Reformulation + Hybrid | TBD | TBD | TBD | TBD | TBD |
| E9 | E8 + Cross-Encoder | TBD | TBD | TBD | TBD | TBD |

**Important:** Results will only be filled after actually running the experiments. No expected or fabricated numerical result will be presented as an experimental finding.

---

# 21. Major Outcomes / Achievements

At the end of the project, we aim to produce:

1. A reproducible TREC-ToT retrieval pipeline.
2. A BM25 baseline.
3. A dense retrieval baseline.
4. A hybrid BM25+dense retrieval system.
5. An LLM-based query decomposition system.
6. An LLM-based query reformulation system.
7. A combined decomposition + reformulation system.
8. A cross-encoder reranking stage.
9. A controlled comparison of nine experiments.
10. Quantitative evaluation using standard IR metrics.
11. Qualitative error analysis of difficult ToT queries.
12. A detailed analysis of when LLM query processing helps and when it hurts.

The final deliverable will therefore be more than a single retrieval model: it will be an **experimental study of how different forms of LLM-based query understanding affect Tip-of-the-Tongue retrieval**.

---

# 22. Scope Boundaries

The project will **not** attempt to:

- train a large language model from scratch;
- fine-tune a large language model;
- reproduce the complete infrastructure of the original research papers;
- build a production-scale 6.4M-document search engine from scratch on Colab Free;
- claim a new state-of-the-art retrieval architecture without evidence;
- use the official 2025 test set for tuning;
- introduce the target entity name through the LLM prompt or post-processing.

The emphasis is on **retrieval experimentation, controlled comparison, LLM query understanding, and analysis**.

---

# 23. Reproducibility

To make the project reproducible, we will maintain:

```text
project/
├── data/
│   ├── dev3-queries.jsonl
│   └── dev3-qrels.txt
│
├── prompts/
│   ├── decomposition_prompt.txt
│   └── reformulation_prompt.txt
│
├── cache/
│   ├── decompositions.jsonl
│   └── reformulations.jsonl
│
├── retrieval/
│   ├── bm25/
│   ├── dense/
│   └── hybrid/
│
├── reranking/
│
├── evaluation/
│   ├── metrics.py
│   └── results.csv
│
└── notebooks/
    └── tot_project.ipynb
```

All important prompts, model names, retrieval parameters, fusion parameters, candidate depths and evaluation settings will be documented.

---

# 24. Research Papers and Official Resources

## TREC-ToT 2025

**Official guidelines**

https://trec-tot.github.io/guidelines.html

**Official dataset — Zenodo**

https://zenodo.org/records/18235970

**Official TREC-ToT benchmark repository**

https://github.com/TREC-ToT/bench

---

## Query Decomposition Paper

**Decomposing Complex Queries for Tip-of-the-tongue Retrieval**

Lin, K., Lo, K., Gonzalez, J. E., & Klein, D. (2023).

https://aclanthology.org/2023.findings-emnlp.367/

**GitHub**

https://github.com/kl2806/whatsthatbook

---

## Query Reformulation / Hybrid Reranking Paper

**Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for Tip-of-the-Tongue Known-Item Retrieval**

Mukhopadhyay, D., Ghosh, U. K., & Chatterjee, S. (2026).

https://arxiv.org/abs/2602.10321

**TREC Proceedings Version**

https://trec.nist.gov/pubs/trec34/papers/mst.tot.pdf

**GitHub**

https://github.com/debayan1405/TREC-TOT-2025

---

# 25. Final Research Question

The entire project can be summarized by the following research question:

> **Can LLM-based query decomposition and query reformulation, combined with hybrid lexical-semantic retrieval and neural reranking, significantly improve Tip-of-the-Tongue known-item retrieval compared with conventional retrieval methods?**

The project will answer this through a controlled nine-experiment evaluation on the TREC-ToT 2025 Dev3 benchmark.

---

# 26. Final Project Pipeline

The final system can be summarized as:

```text
                         TREC-ToT Query
                              |
               +--------------+--------------+
               |                             |
               v                             v
        Original Query                 LLM Query Processing
                                             |
                              +--------------+--------------+
                              |                             |
                              v                             v
                       Decomposition                 Reformulation
                              |                             |
                         Clue 1..N                    Rewritten Query
                              |                             |
                    +---------+---------+           +-------+-------+
                    |                   |           |               |
                   BM25               Dense        BM25           Dense
                    |                   |           |               |
                    +---------+---------+           +-------+-------+
                              |                             |
                             RRF                           RRF
                              |                             |
                              +-------------+---------------+
                                            |
                                      Candidate Fusion
                                            |
                                      Top 50–100 Docs
                                            |
                                            v
                                      Cross-Encoder
                                            |
                                            v
                                      Final Ranking
                                            |
                                            v
                                  TREC-ToT Evaluation
                                            |
                    +-----------------------+----------------------+
                    |          |          |          |             |
                  nDCG@10     MRR       R@10      R@100        R@1000
                                            |
                                            v
                                      Error Analysis
                                            |
                                            v
                                   Final Conclusions
```

---

## 27. One-Sentence Project Summary

> **This project investigates whether LLMs can bridge the gap between how users naturally describe something they cannot remember and how information retrieval systems need queries to be expressed, by combining query decomposition, query reformulation, hybrid retrieval and neural reranking for TREC-ToT known-item retrieval.**
