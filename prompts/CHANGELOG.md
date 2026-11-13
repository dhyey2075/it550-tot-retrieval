# Prompt Changelog

Record every prompt version and why it changed. Do not edit past entries.

| Version | Date | Task | Change | Reason | Split used |
|---|---|---|---|---|---|
| decomp_extractive_v0 | 2026-11-12 | decomposition | Initial facet extractive prompt with 3 few-shots (2 Train-style movie + 1 synthetic landmark + 1 celebrity in file) | plan_v2/05 schema; primary scope-compliant | Train (examples) |
| decomp_predictive_v0 | 2026-11-12 | decomposition | Predictive extension; no name_fragments; ban novel proper names | plan_v2/05 labeled extension | — |
| reform_keyword_v0 | 2026-11-12 | reformulation | R-K keyword-focused prompt; domain vocabulary; drops hedges | plan_v2/06 §3.2 (paper-faithful) | Train |
| reform_descriptive_v0 | 2026-11-12 | reformulation | R-N descriptive encyclopedia lead style (1-3 sentences) | plan_v2/06 §3.3 (dense-friendly) | Train |

## Facet schema (D-06 candidate)

category, description, people, place_setting, time_period, distinctive_details, name_fragments (extractive only).

Inspiration (Lin et al.): field-based N/A decomposition; we adapt to domain-agnostic Wikipedia facets; one JSON call per query (API budget).
