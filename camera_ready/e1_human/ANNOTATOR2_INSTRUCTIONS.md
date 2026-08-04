# Second-Annotator C0-C8 Audit (120 Claims)

You are the second independent annotator. Work alone, using only the
Appendix B taxonomy rubric. Do NOT consult the first annotator, any
judge-result files, or `annotator2_private_id_mapping.json`.

For each claim in `annotator2_labels.csv`, fill in:

- `human_category`: one of C0-C8 (apply precedence C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0)
- `human_effective_date`: earliest date (YYYY-MM-DD) the claim was publicly
  knowable, or `timeless` for C0/C1, or `unresolved` if you cannot decide
- `human_is_leaked`: `yes` if effective date is after the cutoff (C7/C8 are
  always `yes`), otherwise `no`
- `confidence`: high / medium / low
- `note`: required only for low-confidence or ambiguous cases

## Category summary

- C0: timeless definition/logical truth
- C1: stable entity attribute
- C2: residual time-indexed state/status/trend
- C3: quantitative measurement
- C4: comparative/relational claim
- C5: discrete event
- C6: explicit publication/source claim
- C7: target outcome being predicted
- C8: post-outcome consequence

Estimated effort: 3-5 hours. You may look up public sources to establish
dates, but do not use LLM assistants for the category decision.