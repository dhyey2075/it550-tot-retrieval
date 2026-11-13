# 05 — LLM Query Decomposition (E4–E6)

| | |
|---|---|
| **Scope reference** | §4.1 (Lin et al. 2023), §7, §11 (E4–E6), §16, §17 Phase 2 |
| **Papers used** | Lin, Lo, Gonzalez, Klein — *Decomposing Complex Queries for Tip-of-the-tongue Retrieval* (Findings of EMNLP 2023), read in full |
| **Depends on** | 01 (LLM wrapper + cache), 03, 04 (`bm25_search`, `dense_search`, `rrf_fuse`) |
| **Produces** | frozen facet prompts, `cache/decompositions.jsonl` (Dev3 complete), runs E4–E6 plus variants, decision **D-06** |
| **Suggested effort** | 5–7 work sessions |
| **Course link** | Module 2 (query formulation), Module 4 (query expansion), Module 8 (LLMs in IR) |
| **Owner** | `[fill in]` |

Can run in parallel with plan 06 once plan 04 works.

---

## 1. Goal & Exit Criteria

Test scope hypothesis Q4: **retrieving separately from several LLM-generated sub-queries, then fusing, finds better candidates than retrieving from the original query alone.**

- [ ] Facet schema, extractive prompt, and (extension) predictive prompt developed on Train/Dev1, audited, **frozen** and tagged.
- [ ] Decompositions cached for all 536 Dev3 queries (and the pilot sets).
- [ ] E4, E5, E6 (scope-defined) evaluated; paper-faithful and concatenation variants evaluated (§5).
- [ ] Per-facet effectiveness table and decomposition audit completed (§8–9).

---

## 2. What Lin et al. Actually Do (verified from the full text)

1. **Decomposition is field-based, not free-form.** The decomposer outputs one sub-query *per metadata field* of the target (in their book collection: plot, date, cover, title, genre, author). Each field has its own prompt and few-shot examples (up to 8, GPT-3.5), and fields are prompted independently. The LLM answers "N/A" when the query has nothing for a field.
2. **Two prompting strategies.**
   - *Extractive:* sub-queries are pure extractions from the query; "do not guess"; N/A when absent.
   - *Predictive:* sub-queries are inferred to look like the content of the field and may be inferred metadata "or even hallucinations". Their prompts also ask the LLM to guess a title and an author.
   - **Predictive beat extractive overall** (Recall@5 on WhatsThatBook: 32.1 vs 26.6, against 26.5 for the best baseline retriever on the original query). Extractive is better *when it fires* (not N/A) and is more grounded, but fires rarely for some fields.
3. **Specialized retrieval experts.** Each field goes to its own retriever, fine-tuned on LLM-decomposed training pairs (Contriever for text fields, CLIP for covers, a symbolic rule for dates).
4. **Combination:** a **weighted sum of the experts' scores, with weights tuned on validation Recall@5**, and the final combination **includes a base retriever that receives the original query**.
5. **Other findings worth reusing.** Concatenating the sub-queries into one query also helped a little; a single LLM-rewritten query and LLM-generated hypothetical documents did *worse* than the original query for the dense retriever; the plot clue appears in over 90% of queries and is the strongest single expert; dates and genres are weak; author clues are rare but strong when present; the average number of extracted clues per query was about 2.3 (of 6 fields); performance falls as distractor documents are added to the corpus.
6. **Their error analysis (50 failures):** 54% failure to model the query–document relationship, 28% insufficient document representation, 18% the query itself contained an error.

### 2.1 Paper vs. our adaptation

| Aspect | Lin et al. | This project | Why |
|---|---|---|---|
| Targets | Books with structured metadata | Wikipedia pages (movies, landmarks, celebrities) with title + text | Different corpus |
| Fields | plot, date, cover, title, genre, author | **domain-agnostic facets** (§3) | No book metadata or covers here |
| Retrievers | Fine-tuned per-field experts | Zero-shot **BM25 and dense**, no training | Scope §22: no fine-tuning |
| Combination | Weighted score sum | **RRF** (unweighted; weighted RRF as an ablation) | Scope §6.3 |
| Original query in combination | Yes | Run **both** with and without (§5) | Scope E4 says "per-clue"; paper includes the base retriever |
| Predictive prompting | Allowed; best variant | **Extension only**, with guardrails | Scope §7 forbids unsupported facts |
| Title/author guessing | Yes | **Dropped** | Would risk naming the target (scope §22) |
| Prompts | One call per field | One JSON call per query (cheaper) | Free-tier API budget; flagged deviation |

State in the report that you adapt the idea; you do not reproduce their system.

---

## 3. Facet Schema (D-06)

Domain-agnostic, so one schema covers movies, landmarks and celebrities. Every facet may be `N/A`.

| Facet | Holds | Typical domains |
|---|---|---|
| `category` | entity type and genre/field ("science-fiction film", "suspension bridge", "jazz singer") | all |
| `description` | what happens, what it is, what it is known for | movies, landmarks |
| `people` | characters, cast, relationships, roles | movies, celebrities |
| `place_setting` | locations, settings, nationality | landmarks, movies, celebrities |
| `time_period` | release/construction/active era, including era hints from "I watched it in the 90s" | all |
| `distinctive_details` | notable objects, visual features, quotes, trivia | all |
| `name_fragments` | any partial memory of the name itself ("starts with S", "has the word night in it") | all; **extractive only** |

`name_fragments` mirrors the paper's title field (about 30% of their queries contained title guesses, usually uncertain). It is extractive only, so it can never introduce a name that is not already in the query.

Document the schema and rationale in `prompts/CHANGELOG.md`.

---

## 4. Prompt Design

### 4.1 Extractive prompt (primary; scope-compliant)

```text
Extract clues about an item a person is trying to remember. Give one value per field, copied or
lightly trimmed from the description. If the description has nothing for a field, output "N/A".
Do not guess. Do not add anything that is not in the description. Do not name the item.

Fields: category, description, people, place_setting, time_period, distinctive_details, name_fragments
Output JSON only, with exactly those keys.

[few-shot examples]
Description: {query}
```

### 4.2 Predictive prompt (extension)

```text
For each field, write a short phrase describing what a Wikipedia article about this item would
say for that field, inferring from the description. Do NOT name the item, and do NOT introduce any
person, place, or work that is not mentioned in the description. Output "N/A" only if nothing can
be inferred. Output JSON only. (No name_fragments field.)
```

The only change from the paper's predictive idea is the ban on proper names that are not in the query, which keeps leakage risk low while preserving inferred descriptors such as genre or setting.

### 4.3 Few-shot examples
- **4–6 examples from Train** (movies) showing the facet JSON, never an answer.
- **2–3 hand-written synthetic examples for landmarks and celebrities**, invented by you, not copied from Dev3 (Train, Dev1 and Dev2 are movie-only per scope; Dev3 is the only split with other domains, and it is off limits for tuning).
- No Dev3 query or target ever appears in a prompt.

### 4.4 Output handling
- Strict JSON parser with one retry; drop `N/A` and empty values; deduplicate; cap each facet's length.
- Queries with **no** usable facet fall back to the original query as the only list. Count and report these fallbacks.
- Record per-facet non-N/A rates (the paper's "% not N/A" column).

---

## 5. Runs and Fusion Definitions (freeze before Dev3)

Let the non-N/A facets of a query be `f1…fn` (expect about 2–4 on average).

### 5.1 Main experiments (scope-defined)

| Exp | Lists | Fusion |
|---|---|---|
| **E4** | BM25(fi) for every facet | RRF over n lists |
| **E5** | Dense(fi) for every facet | RRF over n lists |
| **E6** | BM25(fi) and Dense(fi) | one protocol, frozen (flat RRF over 2n lists, or two-level: per-facet RRF then across facets); choose on Dev1, reuse in plan 07 (D-08) |

All use extractive facets, no original-query list, depth 1000, frozen RRF `k`.

### 5.2 Paper-faithful and diagnostic variants (cheap: no extra LLM calls)

| Variant | Definition | Question it answers |
|---|---|---|
| **E4+ / E5+ / E6+** | same as E4–E6 **plus** the original query's list(s) | Does the paper's "base retriever on the original query" matter? |
| **E4c / E5c / E6c** | all facet texts **concatenated into one query**, then BM25 / dense / both | Does separating the clues matter, or is concatenation enough? (The paper's concatenation baseline) |
| **Facet-alone runs** | one run per facet type | Which facets carry the signal? (§9) |
| **Weighted RRF** (optional) | weights per facet type, tuned on Train/Dev1 | The paper's weighted combination, rank-based |
| **Facet pruning** (optional) | drop facet types whose standalone Dev1 recall is poor | Do weak facets (the paper found dates and genre weak) dilute the fusion? |

**Which variant is "E4–E6" in the main table?** Pre-register: the scope-defined version (no original list). If, on Dev1, the "+" version wins on nDCG@10 and MRR by a margin clearly outside bootstrap noise, state that, report both, and use the "+" version as the E4–E6 row in the main table with the other in the appendix. Record in D-06.

### 5.3 Predictive extension

| Exp | Definition |
|---|---|
| **E4P / E5P / E6P** | same as E4–E6 but with predictive facets |

Run on Dev1 first. Run on Dev3 as a clearly labeled extension (they violate the scope's no-unsupported-facts rule by design). Add its hallucination audit (§8).

### 5.4 Tier-B note (if plan 02 chose dense reranking of BM25 candidates)
Define precisely what "dense per facet" means (for example, dense reranking of that facet's BM25 top-1000) and label it `E5-R`. Short facets such as `category` produce generic BM25 candidates, so dense reranking cannot recover a target that BM25 missed for that facet; say so.

---

## 6. Prompt Development (Train + Dev1 only)

1. Pick 30–50 Train/Dev1 queries (short, long, noisy).
2. Generate extractive facets; read every output; score with the rubric in §8.1.
3. Revise; log each version in `prompts/CHANGELOG.md`; stop after 2–3 rounds.
4. Compare extractive vs predictive and the §5 variants on Dev1 retrieval metrics. Use a robust aggregate across the dev splits you are allowed to use (plan 01 §7, D-10).
5. Freeze, git-tag, then generate for all Dev3 queries.

**Domain-shift caution.** Your development data is movie-only, but Dev3 mixes movies, landmarks and celebrities. The 2026 reformulation paper used the same arrangement (tuned on movie-only splits, Dev3 as pseudo-test). Mitigate with domain-agnostic facets, synthetic non-movie few-shot examples, and a **format-only** check on Dev3 (JSON parse rate, N/A rate, facet count). Do not read Dev3 outputs for quality and then edit the prompt.

---

## 7. LLM Usage, Cost and Stability

- Calls for Dev3: 536 per variant (extractive, and predictive if run), plus pilot calls. Use `--dry_run` first.
- Temperature 0 and a recorded model id. API models are **not guaranteed deterministic**: the 2026 paper saw Recall@1000 vary (0.71, 0.70, 0.75; CV 3.67%) across three greedy runs on 100 queries.
- **Stability check:** generate facets three times for 100 seeded Train/Dev1 queries, run the retrieval pipeline each time, and report mean ± sd and CV of Recall@100 and nDCG@10. Final experiments read only from the cached first generation.
- Free-tier rate limits: backoff, resumable cache, write after every call.

---

## 8. Decomposition Audit

### 8.1 Manual rubric (50 Dev3 outputs, report-only, no prompt edits afterwards)

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Faithful** | invents facts | minor additions | fully grounded |
| **Complete** | misses main clue | misses minor clue | all kept |
| **Facet-correct** | clues in wrong facets | some mislabeled | correct |
| **Retrieval-friendly** | vague/chatty | acceptable | concrete |

### 8.2 Statistics to report (paper-style)
Non-N/A rate per facet; distribution of facet counts per query; fallback count; JSON failure count.

### 8.3 Leakage checks
1. **Heuristic (no qrels):** flag capitalized names or title-like strings in any facet that are absent from the query. For the predictive variant this is the main safety check.
2. **Qrels-based audit (Dev3, audit-only):** does any facet contain the target page title or a distinctive title token? Report the rate. Never filter, edit or regenerate individual rows (scope §22).
3. If the rate is high, fix the **prompt** on Train/Dev1 evidence, regenerate **everything**, re-freeze.

### 8.4 Contamination note
Dev3 derives from the 2024 test set, which an LLM may have seen. Record it as a limitation.

---

## 9. Evaluation & Analysis Tasks

- [ ] E4–E6 on Dev3 after freeze; planned comparisons E1→E4, E2→E5, E3→E6.
- [ ] Variant comparison table: E4–E6 vs "+" vs concatenation vs (optional) predictive.
- [ ] **Per-facet expert table** (analogue of the paper's Table 6): for each facet type, Recall@5/10/20/100 of its list alone, and the share of queries where it is non-N/A. Expect the description facet to dominate; test it rather than assume.
- [ ] Win/loss lists: queries rescued and broken by decomposition.
- [ ] Number of facets vs. gain; which facet produced the best rank for the target (single lucky facet or genuine fusion?).
- [ ] Ablations: flat vs. two-level fusion; weighted RRF; facet pruning.

---

## 10. Deliverables

1. `prompts/decomposition_extractive.txt`, `prompts/decomposition_predictive.txt`, `prompts/CHANGELOG.md`.
2. `cache/decompositions.jsonl` (variants keyed separately) for Dev3 and pilot sets.
3. Run files and config snapshots for E4–E6, the "+" and concatenation variants, and (extension) E4P–E6P.
4. `analysis/decomposition_audit.csv`, `analysis/facet_expert_table.csv`, stability results.
5. Decision row D-06 (schema, extractive vs predictive, original-list rule, E6 fusion protocol).

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Predictive prompting hallucinates or leaks | Keep it an extension; ban non-query proper names; heuristic + audit |
| Extractive prompting returns mostly N/A | Report N/A rates; the concatenation and "+" variants provide fallbacks |
| Weak facets (category, time) add noise to RRF | Per-facet table; facet pruning ablation |
| Movie-only dev data, mixed-domain Dev3 | Domain-agnostic facets; synthetic non-movie examples; format-only Dev3 check |
| Prompt overfit to Dev3 | Freeze tag before Dev3 generation |
| API nondeterminism | Cache; stability analysis |
| Single call per query differs from per-facet prompts of the paper | State as a deviation; optional 30-query comparison on Dev1 |

## 12. Definition of Done

- [ ] Prompts frozen and tagged; every Dev3 facet set comes from that version.
- [ ] E4–E6 and the variant table evaluated; per-facet table and audit complete.
- [ ] You can explain how your facets differ from Lin et al.'s fields, and why extractive is your primary variant even though the paper found predictive stronger.
