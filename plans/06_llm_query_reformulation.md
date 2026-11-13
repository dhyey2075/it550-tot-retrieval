# 06 — LLM Query Reformulation (E7)

| | |
|---|---|
| **Scope reference** | §5 (Mukhopadhyay et al. 2026), §8, §11 (E7), §16, §17 Phase 3 |
| **Depends on** | 01 (LLM wrapper + cache), 03, 04 |
| **Produces** | frozen reformulation prompt, `cache/reformulations.jsonl` (Dev3 complete), run E7; decision D-07 |
| **Suggested effort** | 3–4 work sessions |
| **Course link** | Module 4 (query expansion, pseudo-relevance feedback), Module 8 (LLM-era retrieval) |
| **Owner** | `[fill in]` |

Can run in parallel with plan 05 once plan 04 works.

---

## 1. Goal & Exit Criteria

Test scope question Q5: **does rewriting a vague ToT query into a clearer, retrieval-oriented query improve BM25, dense, and hybrid retrieval?**

- [ ] Reformulation prompt developed on Train/Dev1, audited, **frozen** and tagged.
- [ ] One reformulated query cached for each of the 536 Dev3 queries.
- [ ] E7 (Reformulation + BM25 + Dense + RRF) run and evaluated.
- [ ] Leakage and hallucination audit completed.

---

## 2. Reformulation vs. Decomposition (keep this distinction crisp in the report)

| | Decomposition (plan 05) | Reformulation (this plan) |
|---|---|---|
| Question the LLM answers | "What are the separate clues?" | "How would I rewrite this so a search system understands it?" |
| Output | several short queries | one clear, richer query |
| Retrieval | several lists fused | one BM25 list + one dense list fused |

The published work behind this plan used a single pre-retrieval LLM call (an 8B-parameter model with a model-agnostic prompt) and then a multi-stage ranking pipeline. Your setting differs: you use an API-hosted LLM, and BM25 + dense + RRF + a cross-encoder rather than their full stack. Say so, and avoid comparing your absolute numbers with theirs.

---

## 3. Prompt Design (D-07)

### 3.1 Requirements (from scope §8)
Preserve the information need; clarify ambiguous wording; make useful implicit concepts explicit; **no hallucinated facts**; **never insert the target answer**; independent of relevance judgments.

### 3.2 Suggested prompt skeleton

```text
A person is trying to find a Wikipedia page for something they cannot name. Rewrite their
description as one clear, specific search query for a search engine.

Rules:
- Keep every concrete detail from the description (genre, setting, time period, characters,
  events, objects, places).
- Remove memory-related chatter and uncertainty ("I think", "can't remember").
- Make implicit information explicit where it is a safe general inference
  (for example: "a movie about a ship that sank" → "film about the sinking of a ship").
- Do NOT add specific facts that the description does not support.
- Do NOT name, guess, or hint at the exact title, person, or place being described.
- Output only the rewritten query, 1-3 sentences, no explanation.

Description: {query}
```

### 3.3 Variants to compare on Dev1 (pick one; D-07)

| Variant | Idea | Risk |
|---|---|---|
| **R-A: concise rewrite** | clean restatement only | little extra recall |
| **R-B: rewrite + controlled expansion** | adds related generic concepts/synonyms | may over-specify or drift |

Keep the comparison small (two variants). The scope's E7 uses **one** reformulated query, so the winner becomes E7. The loser can be reported as an extra Dev1 note.

### 3.4 Output control
- Single string output; trim whitespace and quotes.
- Length cap (for example ≤ 80 words). If longer, truncate at the last full sentence and log how often this happens.
- Empty or failed output → fall back to the original query and **count fallbacks** (report the number).

### 3.5 Prompt development loop
Same as plan 05: 30–50 Train/Dev1 queries → read outputs → revise → log versions → freeze → then generate Dev3. No prompt edits after seeing Dev3 metrics.

---

## 4. Retrieval for E7

```text
reformulated_query ──► BM25  ──┐
                               ├──► RRF (frozen k) ──► top-1000 ──► E7 run
reformulated_query ──► Dense ──┘
```

- Same BM25 parameters, dense model, doc representation, depth and RRF `k` as E3, so **E3 vs. E7 differs only in the query text**. That is the clean test of Q5.
- Also run E7's BM25-only and dense-only lists as **diagnostic runs** (not part of the nine experiments, but they show *which* retriever benefits from reformulation). Label them `E7-bm25`, `E7-dense`.

If D-01 is Tier B, remember the dense step reranks BM25 candidates **retrieved with the reformulated query**, so the candidate set changes between E3 and E7. Note this in the report.

---

## 5. Audit: Hallucination & Leakage

### 5.1 Manual rubric (50 Dev3 outputs, report-only)

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Faithful** | invents specifics | minor additions | grounded |
| **Complete** | drops key clue | drops minor clue | keeps all |
| **Focused** | too broad/vague or too narrow | mostly good | well scoped |
| **Retrieval-friendly** | chatty | acceptable | concrete |

### 5.2 Automated checks
1. **Introduced-entity heuristic:** list capitalized tokens/phrases in the reformulation that are absent from the original. Review all flags manually. This heuristic does not use qrels, so it can also be used during prompt development.
2. **Qrels-based leak audit (Dev3, audit-only):** does the reformulation contain the target page's title or a distinctive title token? Report the rate. Never filter or edit outputs with this information, and never regenerate individual items (scope §22).
3. **Length statistics:** original vs. reformulated length. Very long outputs risk over-specific text.

### 5.3 What to do if leakage is high
Change the **prompt** using Train/Dev1 evidence, regenerate the **entire** Dev3 cache, re-run E7 and everything that depends on it, and document the incident. Do not repair individual rows.

### 5.4 Limitation to state
A strong LLM may recognize the item and encode it implicitly even without naming it (specific phrasing, rare terms). You cannot fully prove otherwise. Report the audit rates, say that the pipeline never receives titles or qrels, and note the training-data contamination risk for 2024-derived queries.

---

## 6. Analysis Tasks (feed plan 09)

- [ ] E3 → E7 paired comparison with correction; effect sizes with CIs.
- [ ] Win/tie/loss list: queries rescued, queries broken.
- [ ] Break down by query length, by domain, and by original-query rank bucket. Does reformulation help most when the original rank was far from the top?
- [ ] Which failure types appear: over-general, over-specific, hallucinated clue (scope §18 categories). Tag 20–30 failures now while the examples are fresh.
- [ ] Recall effect: compare R@100 and R@1000 for E3 vs. E7, since the published work emphasizes first-stage recall gains. Report your own numbers only.

### Optional stretch (course tie-in to Module 4)
Add a classic **RM3 pseudo-relevance-feedback** BM25 run as an extra baseline (`E-PRF`), supported by Anserini/Pyserini. It shows how a non-LLM expansion method behaves when initial recall is poor. Keep it outside the nine-experiment table, clearly labeled as an extra.

---

## 7. Deliverables

1. `prompts/reformulation_prompt.txt` (final) and `prompts/CHANGELOG.md` entries.
2. `cache/reformulations.jsonl` for all Dev3 queries (and pilot).
3. Run files and config snapshots for E7 and the diagnostic runs.
4. `analysis/reformulation_audit.csv` (rubric, flags, fallback count, leak rate).
5. Decision row D-07 (variant chosen, evidence from Dev1).

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reformulation hallucinates details and narrows retrieval | Faithfulness rule; audit; "over-specific reformulation" category |
| Becomes too generic and floods results | Concreteness rule; Dev1 comparison of R-A vs R-B |
| LLM names the answer | Prompt rule + audits; regenerate everything if prompt changes |
| Dev3 tuning by accident | Prompt frozen and tagged before Dev3 generation |
| Tier-B candidate shift confounds E3 vs. E7 | Document it; also report diagnostic lists |

## 9. Definition of Done

- [ ] Prompt frozen, tagged, and used for every Dev3 reformulation.
- [ ] E7 evaluated, with E3-vs-E7 planned comparison completed.
- [ ] Leakage and hallucination audit rates recorded.
