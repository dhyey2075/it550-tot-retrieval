### Cross-Encoder Reranking Analysis (Plan 08, K=30)

#### 1. Reachable Fraction (Ceiling Analysis)
- **Reachable Queries (target in E8 top-30):** `10/142` (7.0%)
- **Unreachable Ceiling:** `132/142` (93.0%) queries cannot be solved by any reranker at depth $K=30$.

#### 2. Headline Performance Comparison (E8 vs E9)

| Metric | Full Dev1 (n=142) E8 | Full Dev1 (n=142) E9 | Reachable Subset E8 | Reachable Subset E9 | Interpretation |
|---|---:|---:|---:|---:|---|
| **nDCG@10** | `0.0312` | `0.0386` | `0.4436` | `0.5482` | Cross-encoder precision inside top-10 |
| **MRR** | `0.0281` | `0.0348` | `0.3867` | `0.4829` | Reciprocal rank of first target |

#### 3. Rank Movement Summary
- **Promoted (improved rank):** `4` queries (2.8%)
- **Demoted (degraded rank):** `4` queries (2.8%)
- **Unchanged:** `134` queries (94.4%)

#### 4. Sample Rank Shifts
- **Promoted Samples:**
  - Query `303`: Rank `15` → `1` (+14)
  - Query `325`: Rank `13` → `2` (+11)
  - Query `895`: Rank `7` → `2` (+5)
  - Query `978`: Rank `26` → `1` (+25)
- **Demoted Samples:**
  - Query `252`: Rank `1` → `3` (-2)
  - Query `428`: Rank `6` → `12` (-6)
  - Query `786`: Rank `1` → `10` (-9)
  - Query `868`: Rank `8` → `16` (-8)
