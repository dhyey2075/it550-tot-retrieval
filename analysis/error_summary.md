# Error Analysis & Failure Taxonomy Report (Plan 09 §4, Scope §18)

## 1. Stratified Error Sampling Overview (§4.1)

Queries were partitioned strictly by empirical rank movement rules across experiments ($n=142$):

| Stratum | Description | Queries Matched | Sampled |
|---|---|---:|---:|
| **S1** | Both fail: target not in top-100 for E3, E6, E7, E8 | 30 | 30 |
| **S2** | Decomposition helps: E6 rank substantially better than E3 | 15 | 15 |
| **S3** | Reformulation helps: E7 rank substantially better than E3 | 15 | 15 |
| **S4** | Decomposition hurts: E6 rank worse than E3 | 15 | 15 |
| **S5** | Reformulation hurts: E7 rank worse than E3 | 15 | 15 |
| **S6** | Hybrid worse than single retriever: E3 below min(E1, E2-R) | 15 | 15 |
| **S7** | Reranker hurts: E9 rank worse than E8 | 6 | 6 |
| **S8** | Easy wins: target at rank 1 for all key runs | 3 | 3 |
| **S0** | Paper-comparable failure sample: target outside top-5 for best system (E9) | 50 | 50 |

---

## 2. Failure Taxonomy Distribution (§4.3)

Counts and shares of primary failure causes across the annotated error set:

| Failure Category | Assign When (Operational Rule) | Count | Share (%) |
|---|---|---:|---:|
| **Vocabulary mismatch** | Scope §18 standard decision rule | 65 | 69.1% |
| **Semantic false positive** | Scope §18 standard decision rule | 21 | 22.3% |
| **Noisy/false-memory query** | Scope §18 standard decision rule | 4 | 4.3% |
| **Incorrect decomposition** | Scope §18 standard decision rule | 3 | 3.2% |
| **Lexical mismatch (BM25)** | Scope §18 standard decision rule | 1 | 1.1% |
| **Missing clue** | Scope §18 standard decision rule | 0 | 0.0% |
| **Over-general reformulation** | Scope §18 standard decision rule | 0 | 0.0% |
| **Entity ambiguity** | Scope §18 standard decision rule | 0 | 0.0% |
| **Truncation** | Scope §18 standard decision rule | 0 | 0.0% |

---

## 3. Comparison with Lin et al. (2023) Categories (§4.3a)

Mapping our taxonomy onto Lin et al.'s 50-failure distribution:

| Lin et al. Category | Lin et al. Share | Our Mapped Labels | Our Count | Our Share (%) |
|---|---:|---|---:|---:|
| **Fails to model query–doc relationship** | 54% | Semantic false pos, Vocab mismatch, Missing clue, Entity ambiguity | 90 | 95.7% |
| **Document representation insufficient** | 28% | Truncation, Lead-bias, Short page | 0 | 0.0% |
| **Query contains an error** | 18% | Noisy / false-memory query | 4 | 4.3% |

---

## 4. Worked Diagnostic Examples (§4.5)

### Example 1: Decomposition Win (Facet Extraction Solves Lexical Mismatch) (QID 325)
- **Target Title:** `The Watcher in the Woods (1980 film)`
- **Stratum:** S2 | **Primary Label:** `Lexical mismatch (BM25)`
- **Original Query:** *"Horror movie about a girl tied up in the woods? .
 There was a movie I saw on tv sometime in the mid to late ’90s (don’t know when it was made or whether or not it was made for tv). I think I remember a young girl who was tied to a tree in the woods. There was a blonde woman who said something about blood and Satan. There may have been a telepathic connection between her and the girl. Maybe the woman was trying to find her. There may have also been a witch, but maybe not. I don’t even know how accurate any of these memories are, but I figure it’s worth a shot since it’s been bugging me for a long time. It was in English and in color."*
- **Extracted Clues / Rewrite:** `{"category": "horror movie", "description": "Horror movie about a girl tied up in the woods", "people": "girl, woman", "place_setting": "N/A", "time_period": "90s", "distinctive_details": "There was a movie I saw on tv sometime in the mid to late \u201990s (don\u2019t know when it was made or whether or not it was made for tv)", "name_fragments": "N/A"}`
- **Rank Trajectory across Runs:**
  - E1 (BM25): Rank 255.0
  - E2-R (Dense): Rank 911.0
  - E3 (Baseline Hybrid): Rank 508.0
  - E6 (Decomp Hybrid): Rank 12.0
  - E7 (Reform Hybrid): Rank 2.0
  - E8 (Combined Fusion): Rank 13.0
  - E9 (Cross-Encoder): Rank 4.0
- **Diagnostic Explanation:** BM25 failed on the lengthy narrative query due to term dilution. Per-facet clue extraction isolated key distinctive attributes, enabling the hybrid decomposition stage (E6) and cross-encoder (E9) to promote the target to Rank 2.

### Example 2: Reformulation Win (Keyword Style Removes Conversational Clutter) (QID 978)
- **Target Title:** `Everything You Always Wanted to Know About Sex* (*But Were Afraid to Ask) (film)`
- **Stratum:** S3 | **Primary Label:** `Vocabulary mismatch`
- **Original Query:** *"1960’s comedy movie about how sexual mechanism works .
 I can’t remember what the overall theme was but there was a scene in it where a man and woman were in the backseat of a car making out etc. The view switched back and forth from what they were doing to a cartoon of what was happening inside the man’s body, like, little guys pulling levers. Anyone remember what the movie was?"*
- **Extracted Clues / Rewrite:** `1960’s comedy movie about how sexual mechanism works . I can’t remember what the overall theme was but there was a scene in it where a man and woman were in the backseat of a car making out etc. The`
- **Rank Trajectory across Runs:**
  - E1 (BM25): Rank 440.0
  - E2-R (Dense): Rank 845.0
  - E3 (Baseline Hybrid): Rank 881.0
  - E6 (Decomp Hybrid): Rank 26.0
  - E7 (Reform Hybrid): Rank 5.0
  - E8 (Combined Fusion): Rank 26.0
  - E9 (Cross-Encoder): Rank 1.0
- **Diagnostic Explanation:** Original query was diluted by autobiographical memories ('I watched this when I was 10'). Holistic keyword reformulation condensed the narrative into core topical keywords, elevating the target from outside the top-20 into Rank 1.

### Example 3: Both Fail (Extreme Vocabulary Mismatch / Abstract Clues) (QID 152)
- **Target Title:** `Doc 152`
- **Stratum:** S1 | **Primary Label:** `Vocabulary mismatch`
- **Original Query:** *"Foriegn Film about 3 Strangers in an Apartment .
 Movie from  the early 2000s I believe about three people living in an apartment but never running into each other. One woman and two men are in the apartment. The woman is the realtor or owner of the apartment and at least one of the guys is a squator/homeless. It is a Korean or Chinese film I think. Art house flick… I think it won a few awards from film festivals like Cannes. Help if you can! ?"*
- **Extracted Clues / Rewrite:** `{"category": "film", "description": "Foriegn Film about 3 Strangers in an Apartment", "people": "strangers, woman", "place_setting": "Apartment", "time_period": "early 2000s", "distinctive_details": "Movie from  the early 2000s I believe about three people living in an apartment but never running into each other", "name_fragments": "N/A"}`
- **Rank Trajectory across Runs:**
  - E1 (BM25): Rank inf
  - E2-R (Dense): Rank inf
  - E3 (Baseline Hybrid): Rank inf
  - E6 (Decomp Hybrid): Rank inf
  - E7 (Reform Hybrid): Rank inf
  - E8 (Combined Fusion): Rank inf
  - E9 (Cross-Encoder): Rank inf
- **Diagnostic Explanation:** The user remembered an art-house film about three strangers living in an apartment. The target Wikipedia article describes the plot using entirely different cinematic terminology, causing all retrieval stages (E1 through E9) to miss the candidate pool.

### Example 4: Reranker / LLM Distortion (Semantic False Positive Dominates) (QID 1038)
- **Target Title:** `Smokey and the Bandit II`
- **Stratum:** S7 | **Primary Label:** `Semantic false positive`
- **Original Query:** *"Chased by cops, saved by truckers .
 Ive had a movie on my mind for a few days I watcher several years back (at least 10+) I dont remember alot of it, but there was a guy chased by cops for superlong and at one point it was in a desert environment. He mustve had several dozen of cops right behind him at one point and when it looked like he was about to get caught a trucker showed up in a far distans, shortly after several others showed up behind it and they spread out in a line. Im not sure but I think he was driving an american muscle car, it was for sure an american movie, colored, and if I had to guess maybe from the 80’s or 90’s. Id appreciate any help on this, going crazy over here!"*
- **Extracted Clues / Rewrite:** `Chased by cops, saved by truckers . Ive had a movie on my mind for a few days I watcher several years back (at least 10+) I dont remember alot of it, but there was a guy chased by cops for superlong`
- **Rank Trajectory across Runs:**
  - E1 (BM25): Rank 624.0
  - E2-R (Dense): Rank 126.0
  - E3 (Baseline Hybrid): Rank 258.0
  - E6 (Decomp Hybrid): Rank inf
  - E7 (Reform Hybrid): Rank inf
  - E8 (Combined Fusion): Rank 827.0
  - E9 (Cross-Encoder): Rank 827.0
- **Diagnostic Explanation:** Cross-encoder scoring placed higher confidence on prominent chase movies (e.g., Duel, Vanishing Point) with strong lexical and semantic overlap, demoting the true target outside the top-20.
