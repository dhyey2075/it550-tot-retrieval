### Complementarity Analysis (Q6)

Evaluated on 142 queries at cutoff $K=100$:

| Category | Queries | Fraction | Interpretation |
|---|---:|---:|---|
| Targets retrieved ONLY by Decomposition (E6) | 0 | 0.0% | Unique signal from faceted clues |
| Targets retrieved ONLY by Reformulation (E7) | 9 | 6.3% | Unique signal from holistic rewrite |
| Targets retrieved by BOTH branches | 0 | 0.0% | Concordant queries |
| Targets MISSED by both | 133 | 93.7% | Hard / out-of-pool queries |

#### Oracle Fusion Ceiling (Best of E6 and E7 per query)
- **Oracle nDCG@10:** `0.0291`
- **Oracle MRR:** `0.0278`
- **Oracle Recall@100:** `0.0634`
- **E8 Improved Ranks:** `19` queries
- **E8 Diluted Ranks:** `19` queries
