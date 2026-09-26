# Validation actually performed

- 14 CPU unit tests passed: T1 normalization, rubric/Strict/penalty handling, evidence validation, safe input fields, fixed targets, shards, output identity, capped text retention, image modes, completeness, frozen split and sorted target isolation.
- 11 Python files parsed; generic scheduler shell script passed bash syntax validation.
- 19 scientific kernel functions match the original function ASTs exactly.
- Local original release and local public Parquet format both loaded all 3,229 inputs; prompts and displayed image pixels matched.
- Existing anonymous scoring package: all 3,229 targets and operative rubric fields match the original release; T1 scoring equivalence checked. No answers/rubrics were regenerated or bundled.
- 1,615 Train and 308 Validation exports matched historical IDs, prompts, references, image hashes and source groups; zero Test targets exported.
- Training entry-point runtime import was attempted but not verified: dependency loading stalled on shared storage. Its Python syntax and scientific kernels were checked separately. All five non-training module help commands passed.
- The fixed membership has 227 nonoverlapping source groups and 1,306 Test items; held-out Challenge has 249 items.

## Limits

No new GPU training, model inference, automatic judging or paid API call was performed. Portable-wrapper checks do not prove bitwise GPU equivalence on other hardware. No clean-environment dependency installation was rerun. Historical training and scoring semantic limitations remain as documented in README. Source-group construction is reused from the frozen mapping, not recalculated.
