# 05 — LLM Query Decomposition (E4–E6)

| | |
|---|---|
| **Scope reference** | §4.1 (Lin et al. 2023), §7, §11 (E4–E6), §16, §17 Phase 2 |
| **Depends on** | 01 (LLM wrapper + cache), 03, 04 (`bm25_search`, `dense_search`, `rrf_fuse`) |
| **Produces** | frozen decomposition prompt, `cache/decompositions.jsonl` (Dev3 complete), runs E4, E5, E6; decisions D-05, D-06 |
| **Suggested effort** | 4–6 work sessions |
| **Course link** | Module 2 (query formulation), Module 4 (query expansion), Module 8 (LLMs in IR) |
| **Owner** | `[fill in]` |

Can run in parallel with plan 06 once plan 04 works.

---

## 1. Goal & Exit Criteria

Test the hypothesis from scope §4.1: **retrieving separately from several LLM-generated clues, then fusing, finds better candidates than retrieving from the original query.**

- [ ] A decomposition prompt developed on Train/Dev1, reviewed manually, then **frozen** with a git tag.
- [ ] Clues generated and cached for all 536 Dev3 queries (and the Dev1 pilot).
- [ ] E4 (per-clue BM25 + RRF), E5 (per-clue dense + RRF), E6 (per-clue BM25 + dense + RRF) run and evaluated.
- [ ] Decomposition audit completed (§6).

---

## 2. How This Adapts the Original Paper

Lin et al. decompose a query into clues and send each clue to a retriever suited to that clue type, then combine the results. Their setting was book search with specialized retrievers. Your adaptation keeps the **core idea** (decompose, retrieve per clue, combine) but uses only general text retrievers (BM25 and dense) over Wikipedia. State this honestly in the report: you adapt the idea; you do not reproduce their system.

Before writing the prompt, read the paper's decomposition section and note: what clue types they use, how many clues they typically produce, and how they combine scores. Paste your notes in `prompts/CHANGELOG.md` under "inspiration".

---

## 3. Prompt Design (D-06)

### 3.1 Requirements (from scope §7)
The LLM must: produce a **small number** of meaningful clues; **preserve** information from the query; **not add** unsupported facts; **not reveal or guess** the target's name; produce **retrieval-friendly** text; avoid explanations.

### 3.2 Suggested prompt skeleton

```text
You help a search engine find a Wikipedia page that a person is trying to remember.
The person describes it vaguely. Split their description into 2-5 independent clues.

Rules:
- Each clue must be a short, self-contained search phrase or sentence.
- Use ONLY information stated in the description. Do not add facts.
- Do NOT guess, name, or hint at what the item is.
- Drop memory-related chatter ("I can't remember", "I watched it as a kid") unless it carries a
  real clue (for example, a time period).
- Prefer concrete attributes: type/genre, setting, time period, people/roles, plot events, notable
  objects, locations.
- Output JSON only: {"clues": ["...", "..."]}

Description: {query}
```

Few-shot examples (2–3), if used, must come from **Train** and must show decompositions, not answers. Never include a Dev3 query or Dev3 target in any prompt.

### 3.3 Output schema and parsing
- Strict JSON; a parser with a fallback (strip code fences, retry once with "return valid JSON only").
- Validate: `1 ≤ len(clues) ≤ N_max` (suggest `N_max = 5`), each clue non-empty, duplicates removed, clue length capped.
- If a call fails twice, **log the failure and fall back to the original query as the single clue**, and count how many queries took the fallback. Report that number.

### 3.4 Prompt development loop (Train + Dev1 only)
1. Pick **30–50** Train/Dev1 queries (include short, long, noisy ones).
2. Generate clues; read every output.
3. Score manually on a 4-point rubric (see §6.1). Keep a table.
4. Revise the prompt. Record each version in `prompts/CHANGELOG.md` with the reason.
5. After 2–3 iterations, stop. Freeze as `decomp_v_final`; git tag it.
6. Only then generate clues for all 536 Dev3 queries.

Do not iterate on Dev3 results. If you want to compare prompt variants, compare them on Dev1 retrieval metrics and log the choice (D-06).

---

## 4. Retrieval Per Clue (E4–E6)

### 4.1 Definitions (freeze before running; D-06 or D-08)

For a query with clues `c1…cn`:

| Exp | Lists produced | Fusion |
|---|---|---|
| **E4** | BM25(c1) … BM25(cn) | RRF over the n lists |
| **E5** | Dense(c1) … Dense(cn) | RRF over the n lists |
| **E6** | BM25(ci) and Dense(ci) for all i | choose one protocol below and use it everywhere |

E6 fusion protocol options:
- **Flat:** one RRF over all `2n` lists. Simplest; clue count influences weight.
- **Two-level:** RRF per clue across BM25 and dense first, then RRF across clues. Gives each clue equal weight.

Compare the two on **Dev1** only, pick one, record in D-08, and use it again in plan 07 so E6 and E8 are consistent.

### 4.2 Other design choices to fix
- **Is the original query also one of the lists?** Scope E4 says "per-clue", so the **primary** setting excludes the original query. Run "with original added" as an **ablation on Dev1 and, if time permits, Dev3 clearly labeled as an extra**. This directly answers whether decomposition is complementary to the raw query.
- **Per-clue depth:** retrieve 1000 per clue, fuse, output top 1000, to match E1–E3.
- **Clue weights:** equal (unweighted RRF). Weighting by clue type is out of scope unless time permits.
- **RRF k:** reuse the frozen value from D-04.

### 4.3 Efficiency
Total retrieval calls per query ≈ `n clues × 2 retrievers`. For 536 queries and ~4 clues that is a few thousand retrieval calls, which is small for BM25 and for batched dense query encoding. Batch-encode all clue strings once and cache the query vectors.

---

## 5. LLM Usage & Cost Control

- Number of Dev3 decomposition calls = 536 (one per query) plus pilot calls. Use `--dry_run` to count uncached queries before running.
- Respect free-tier rate limits with backoff; the cache makes the job resumable.
- Record: model id, prompt version, temperature, timestamp, and the raw model output beside the parsed clues.
- Run a **determinism spot-check:** regenerate clues for 20 queries; if they differ, note it. Final experiments read the cache only, so results stay reproducible regardless.

---

## 6. Decomposition Quality Audit

### 6.1 Manual rubric (on a sample of 50 Dev3 outputs, **report-only**, no prompt changes afterwards)

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Faithful** (no invented facts) | invents facts | minor additions | fully grounded |
| **Complete** (key clues kept) | misses main clue | misses minor clue | all kept |
| **Non-redundant** | mostly duplicates | some overlap | distinct |
| **Retrieval-friendly** | vague / chatty | acceptable | concrete, searchable |

### 6.2 Leakage check (scope §7: avoid revealing the target)
Two checks, kept separate:

1. **Pipeline-independent heuristic:** flag clues that contain capitalized multi-word names or title-like strings that do **not** appear in the original query. These are candidate "named entity introduced by the LLM" cases.
2. **Qrels-based audit:** after Dev3 clues are generated, compare clue text against the target page title. This is **audit-only**: it measures leakage rate for the report. It must **never** be used to filter, edit, or regenerate clues (scope §22).

If the leakage rate looks high, fix the **prompt** using Train/Dev1 evidence, regenerate **everything**, and re-freeze. Do not patch individual Dev3 outputs.

### 6.3 Contamination note
Dev3 queries come from the 2024 test set, and an LLM may have seen related content in training. Record this as a limitation. It is one more reason leakage must be measured, not assumed away.

---

## 7. Evaluation & Analysis Tasks

- [ ] Run E4, E5, E6 on Dev3 (after freeze); evaluate via plan 03.
- [ ] Planned comparisons: E1→E4, E2→E5, E3→E6 (paired tests with correction).
- [ ] Per-query: did decomposition rescue a miss? Did it break a hit? Save win/loss lists.
- [ ] Relationship checks (for plan 09): number of clues vs. gain; query length vs. gain; which clue was the "best" one (rank of target per clue list) to see whether gains come from one lucky clue or from fusion.
- [ ] Ablations (Dev1 or Dev3-labeled-extra): `N_max ∈ {2, 3, 5}`; with/without original query; flat vs. two-level fusion.

---

## 8. Deliverables

1. `prompts/decomposition_prompt.txt` (final) and `prompts/CHANGELOG.md`.
2. `cache/decompositions.jsonl` with all Dev3 queries (and Dev1/Train pilot items).
3. Run files and config snapshots for E4–E6.
4. `analysis/decomposition_audit.csv` (rubric scores, leakage flags, fallback count).
5. Decision rows D-05, D-06 (and the E6 fusion choice recorded under D-08).

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM leaks or guesses the answer | Strict prompt, heuristic + audit, regenerate everything if the prompt changes |
| Redundant or weak clues dilute fusion | Cap clue count; ablate N; manual review |
| Hallucinated clues misdirect retrieval | "Use only stated info" rule; rubric faithfulness check; error category "hallucinated clue" in plan 09 |
| Prompt overfit to Dev3 | Develop on Train/Dev1; freeze with git tag before Dev3 generation |
| API outage / rate limits | Resumable cache; backoff; start generation early |
| Flat RRF lets a many-clue query dominate | Compare two-level fusion on Dev1 |

## 10. Definition of Done

- [ ] Prompt frozen and tagged; every Dev3 clue list comes from that exact version.
- [ ] E4–E6 evaluated; planned comparisons run; audit table complete.
- [ ] You can explain, with a worked Dev3 example, one query where decomposition helped and one where it hurt (details go to plan 09).
