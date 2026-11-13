# 06 — LLM Query Reformulation (E7)

| | |
|---|---|
| **Scope reference** | §5 (Mukhopadhyay et al. 2026), §8, §11 (E7), §16, §17 Phase 3 |
| **Papers used** | Mukhopadhyay, Ghosh, Chatterjee — *Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for ToT Known-Item Retrieval* (arXiv 2602.10321), read in full; Lin et al. 2023 for the rewrite-vs-original finding |
| **Depends on** | 01 (LLM wrapper + cache), 03, 04 |
| **Produces** | frozen reformulation prompt(s), `cache/reformulations.jsonl` (Dev3 complete), run E7 plus diagnostics, decision **D-07** |
| **Suggested effort** | 4–5 work sessions |
| **Course link** | Module 4 (query expansion, pseudo-relevance feedback), Module 8 |
| **Owner** | `[fill in]` |

Can run in parallel with plan 05 once plan 04 works.

---

## 1. Goal & Exit Criteria

Test scope Q5: **does an LLM-rewritten query improve BM25, dense and hybrid retrieval?**

- [ ] Two reformulation styles developed on Train/Dev1, compared, one frozen for E7 (D-07).
- [ ] One reformulation cached for each of the 536 Dev3 queries.
- [ ] E7 (Reformulation + BM25 + Dense + RRF) and the per-retriever diagnostics evaluated.
- [ ] Stability, leakage, vocabulary and query-type analyses completed.

---

## 2. What the 2026 Paper Actually Does (verified from the full text)

**Rewriting step**
- A single LLM call per query, zero-shot, no fine-tuning. Three 8B-class instruction models compared: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct. Greedy decoding, temperature 0, at most 128 new tokens.
- The prompt turns a verbose ToT description into a **simple, keyword-focused search query**. Its guidelines, paraphrased: identify the implied entity type; add domain vocabulary; extract concrete details without adding facts; drop conversational filler; drop uncertainty and negative constraints unless crucial; follow a strict grounding rule (every token traceable to the query or a safe lexical transformation); allow cautious alias/canonical-form expansion only when an anchor entity is explicitly present; output only the query.
- Mistral was chosen as the default after the comparison below.

**Reported results (Dev3, BM25 first stage, full 6.4M corpus; their Table 6)**

| Query version | nDCG@10 | MRR | Recall@1000 |
|---|--:|--:|--:|
| Original | 0.3366 | 0.3129 | 0.7705 |
| Qwen2.5-7B rewrite | 0.3720 | 0.3455 | 0.7836 |
| Llama-3.1-8B rewrite | 0.3889 | 0.3597 | 0.7836 |
| Mistral-7B rewrite | 0.4609 | 0.4338 | 0.8284 |

**Other findings that shape this plan**
- **Pseudo-relevance feedback failed.** RM3, Bo1 and KL expansion (PyTerrier defaults: 3 feedback docs, 10 expansion terms, RM3 λ = 0.6) all scored below plain BM25 on original queries; RM3's Recall@1000 fell to 0.5093.
- **Rewrites help BM25 but can hurt dense models.** When a dense ensemble reranked BM25's top-1000 for Mistral-rewritten queries, nDCG@10 fell 18% (0.4609 → 0.3778) while Success@100 rose (0.7183 → 0.7724). The authors attribute this to keyword-style rewrites that dense models match poorly. Lin et al. likewise found a single rewritten query *worse* than the original for a dense retriever (their fine-tuned setting).
- **Stability:** three greedy reruns on 100 queries gave Recall@1000 of 0.71, 0.70, 0.75 (CV 3.67%); they treat CV below 2% as stable and 5–10% as needing explicit variability reporting.
- **By query type (100 labeled test queries, BM25):** gains were largest for false-memory queries (+0.29 Recall@1000, n = 7), also positive for visual/descriptive (+0.15, n = 54) and complex-constraint queries (+0.09, n = 34), and negative for simple entity recall (−0.20, n = 5). The group sizes are tiny, so treat these as hints.
- **Vocabulary:** 63% of 100 inspected rewrites used domain-specific terms, 37% generic phrasing.
- **Indirect references:** only 1.06% of indirect entity pointers were resolved to a name; 94.68% were paraphrased. The authors present this as intended conservatism.

### 2.1 Paper vs. our setting

| Aspect | Paper | This project |
|---|---|---|
| LLM | open 7–8B models, local | API LLM (Gemini or OpenAI), possibly stronger |
| Downstream | BM25 → dense ensemble rerank → monoT5 → 72B listwise | BM25 + dense + RRF (E7), later a cross-encoder (E9) |
| Rewrite style | keyword-focused | scope example is descriptive prose; **both styles tested** |
| Corpus | full 6.4M, 500 GB RAM machine | your tier from plan 02 |

Do not compare absolute numbers with theirs unless setting, LLM and tuning match.

---

## 3. Prompt Design (D-07)

### 3.1 Two styles (compare on Dev1)

| Style | Output | Aim |
|---|---|---|
| **R-K (keyword, paper-faithful)** | compact keyword-focused query with domain vocabulary | BM25-friendly |
| **R-N (descriptive, scope example)** | 1–3 sentences reading like a Wikipedia lead ("Science-fiction film involving shared dream environments…") | dense-friendly |

### 3.2 R-K skeleton (paraphrasing the paper's guidelines, adapted to three domains)

```text
You rewrite a vague description of something a person cannot name into a short search query.
- If the type of item is implied (film, landmark, person, ...), say it and add domain vocabulary
  that fits (e.g. "anthology film", "suspension bridge").
- Keep concrete details stated in the description: events, relationships, names, dates, places.
- Remove filler and hedges ("I think", "maybe", "I remember seeing").
- Remove uncertainty and "not X" statements unless essential.
- Every word must come from the description or be a safe synonym or domain term. Do not add facts.
- Add an alias or canonical form only if the description explicitly mentions that entity.
- Never guess or name the item being described.
Output only the query.
```

### 3.3 R-N skeleton
Same rules, but ask for "one to three plain sentences describing the item as an encyclopedia would, using only what the description supports".

### 3.4 Selection rule (pre-registered)
Run both styles on Dev1 through BM25, dense, and hybrid. Choose for **E7** the style with the better **hybrid** nDCG@10 and MRR (the scope defines E7 as hybrid), ties broken toward the simpler output. The other style stays as a Dev1 comparison and, if time permits, a Dev3 extension. Record in D-07.

### 3.5 Per-retriever routing extension
`E7-split`: R-K to BM25 and R-N to dense, fused by RRF. This directly tests the paper's explanation for the dense-stage drop. It is an extension, not one of the nine.

### 3.6 Output control
Single string; trim quotes; cap length (about 80 words); empty or failed output falls back to the original query, with the fallback count reported.

### 3.7 Development loop
30–50 Train/Dev1 queries → read outputs → revise → log versions → freeze → generate Dev3. No edits after seeing Dev3 metrics. Same domain-shift caution as plan 05 §6 (movie-only development data; add synthetic landmark and celebrity examples; Dev3 format-only checks).

---

## 4. Retrieval for E7

```text
reformulated_query ──► BM25  ──┐
                               ├──► RRF (frozen k) ──► top-1000 ──► E7
reformulated_query ──► Dense ──┘
```

- Same BM25 parameters, dense model, document representation, depth and RRF `k` as E3, so **E3 vs. E7 differs only in the query text**.
- **Required diagnostics:** `E7-bm25` and `E7-dense`. The papers suggest BM25 benefits and dense may not, and these runs show which.
- Tier B note: if dense only reranks BM25 candidates, the candidate set changes between E3 and E7 because BM25 now uses the rewrite. Say so.

---

## 5. Stability, Audit and Analysis

### 5.1 Rewrite stability (adopt the paper's protocol)
Rewrite 100 seeded Train/Dev1 queries three times with identical settings (an API LLM at temperature 0 is still not guaranteed identical). Report exact-match rate of rewrites, and mean ± sd and CV of Recall@100 and nDCG@10 across runs. Use the paper's thresholds as stated criteria (CV under 2% stable; 5–10% reported explicitly). Final experiments read the cached first generation.

### 5.2 Faithfulness and leakage (50 Dev3 outputs, report-only)

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Faithful** | invents specifics | minor additions | grounded |
| **Complete** | drops key clue | drops minor clue | keeps all |
| **Focused** | too broad or too narrow | mostly good | well scoped |
| **Retrieval-friendly** | chatty | acceptable | concrete |

Automated checks: (a) introduced-entity heuristic (capitalized phrases absent from the query); (b) qrels-based title-leak audit on Dev3, **audit-only, never used to filter or edit**; (c) length statistics. If leakage is high, fix the prompt on Train/Dev1 evidence and regenerate everything.

### 5.3 Paper-style audits (100 seeded Dev3 queries, manual, report-only)
- **Vocabulary:** domain-specific vs. generic phrasing (paper: 63% / 37%).
- **Indirect references:** among queries with a relational pointer ("the actor from …"), classify the rewrite as Resolved (a name was introduced), Paraphrased, Removed or Kept. Any "Resolved" case is also a leakage risk under scope §22; count them.
- **Query type:** label each query as simple entity recall, complex constraints, false memory, or visual/descriptive, then report E3 → E7 changes per type with counts and intervals. Keep the labeling guide in `analysis/`.

### 5.4 Core comparisons
- [ ] E3 → E7 paired test (planned comparison, corrected).
- [ ] Win/tie/loss lists; breakdown by query length, domain and original-rank bucket.
- [ ] Recall effect: R@100 and R@1000 for E3 vs. E7 (the paper stresses first-stage recall). Report your own numbers only.
- [ ] Failure tags (scope §18): over-general, over-specific, hallucinated clue.

---

## 6. Recommended Extras (outside the nine experiments, clearly labeled)

1. **`E-PRF` baseline.** BM25 with RM3, Bo1 and KL on original queries (3 feedback documents, 10 terms, RM3 λ = 0.6, as in the paper). The paper found all three below BM25 on Dev3 and RM3's recall collapsing; check whether you replicate that. It ties directly to course Module 4 and supports the argument that LLM reformulation addresses poor first-stage recall where PRF cannot.
2. **LLM comparison on Dev1.** The paper's three LLMs differed by up to 0.09 nDCG@10 on the same BM25 pipeline, so the LLM is a real experimental variable. If budget allows, compare your API model with one open 7–8B model run in 4-bit on Colab, using the paper's decoding settings. Report as a Dev1 side study, or as a Dev3 extension if you run it.

---

## 7. Deliverables

1. `prompts/reformulation_keyword.txt`, `prompts/reformulation_descriptive.txt`, `prompts/CHANGELOG.md`.
2. `cache/reformulations.jsonl` for Dev3 and pilot sets (styles keyed separately).
3. Run files and config snapshots for E7, `E7-bm25`, `E7-dense`, optional `E7-split` and `E-PRF`.
4. `analysis/reformulation_audit.csv`, stability results, query-type labels.
5. Decision row D-07 with the Dev1 style comparison.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Keyword rewrites help BM25 but hurt dense (as in the paper) | Two styles; per-retriever diagnostics; `E7-split` |
| Reformulation hallucinates or over-specifies | Strict grounding rule; audits; failure tags |
| LLM names the answer | Prompt rules; audits; regenerate everything if the prompt changes |
| Stronger API LLM may recognize items and encode them implicitly | Report audit rates; state as a limitation |
| Dev3 tuning by accident | Freeze tag before Dev3 generation |
| Tiny query-type groups mislead | Report counts; label 100–150 queries; call results exploratory |
| Tier-B candidate shift confounds E3 vs. E7 | Document; report diagnostics |

## 9. Definition of Done

- [ ] Style chosen on Dev1, prompt frozen and tagged, and used for every Dev3 rewrite.
- [ ] E7 and diagnostics evaluated; E3-vs-E7 planned comparison done.
- [ ] Stability, leakage, vocabulary, indirect-reference and query-type results recorded.
