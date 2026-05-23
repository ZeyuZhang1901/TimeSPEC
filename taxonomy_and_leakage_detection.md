# Claim Taxonomy and Leakage Detection Design

## Purpose

This document specifies the taxonomy and leakage-detection component for the unified leakage evaluation framework. The component takes structured method outputs with evidence claims and source dates, then assigns a binary temporal leakage label to each evidence claim.

The design goal is to keep the final metric simple while making each claim-level decision auditable. Each claim receives exactly one taxonomy class, one effective source date, one binary leakage label, and metadata explaining the decision.

## Scope

The leakage detector operates over evidence claims from structured generation outputs. Each method output is expected to contain:

```json
{
  "evidence": [
    {
      "id": 1,
      "fact": "...",
      "source_date": "YYYY-MM-DD"
    }
  ],
  "prediction": "... task-specific ...",
  "reasoning": "..."
}
```

The detector focuses only on temporal leakage. It does not judge whether a claim is factually true. For leakage, the relevant question is:

> If the claim were true, when would the information asserted by the claim first have been publicly knowable?

The model-provided `source_date` is treated as a declared temporal anchor, not as ground truth.

## Core Definitions

Let `c` be an evidence claim and `t_ref` be the instance cutoff date. The verifier computes an effective public-availability date:

```text
effective_source_date(c)
```

The binary leakage label is:

```text
is_leaked(c, t_ref) = 1[effective_source_date(c) > t_ref]
```

For target-outcome and post-outcome consequence claims, leakage can be assigned by category rule even when an exact date is unavailable.

## Taxonomy

The taxonomy is defined over atomic factual evidence claims. It is general across domains and tasks. It is not specific to stock, salary, or legal prediction.

### Classes

`C0 Timeless Definition`

A claim whose truth is fixed by definition, convention, logic, or mathematical identity.

Examples:

- `AAV means average annual value.`
- `Brier score is squared probability error.`

Default leakage treatment: not leaked.

`C1 Stable Entity Attribute`

A durable identity, membership, or structural attribute of an entity that is not tied to the target event's resolution.

Examples:

- `Chevron is an integrated energy company.`
- `Kinder Morgan operates in the midstream sector.`
- `The Supreme Court is the highest U.S. court.`

Default leakage treatment: usually not leaked, unless the text or declared date indicates a time-sensitive status.

`C2 Time-Indexed State`

A claim about an entity, market, institution, person, or case at a time or during a period.

Examples:

- `By late 2019, Chevron had stabilized its balance sheet.`
- `The player was a restricted free agent.`
- `Schlumberger was experiencing declining demand.`

Default leakage treatment: requires date justification.

`C3 Quantitative Measurement`

A numerical value, statistic, price, score, probability, salary, revenue, rank, margin, vote count, or other measured quantity.

Examples:

- `Revenue was $25.8B.`
- `The stock closed at $57.32.`
- `The player averaged 10.8 points per game.`

Default leakage treatment: requires date justification. The date must correspond to when the measurement became public, not merely the period measured.

`C4 Comparative or Relational Claim`

A claim whose truth depends on comparing entities, groups, periods, or alternatives.

Examples:

- `Chevron had lower debt than peers.`
- `Company A was less diversified than Company B.`
- `This player was more efficient than comparable guards.`

Default leakage treatment: requires date justification, often with stricter verification because the claim depends on multiple inputs.

`C5 Discrete Event`

A localized occurrence, action, decision, filing, appointment, signing, announcement, injury, transaction, release, or change.

Examples:

- `The player signed with the Pacers.`
- `The Court granted certiorari.`
- `The company announced a merger.`

Default leakage treatment: requires event date or public announcement date.

`C6 Publication or Source Claim`

A claim about what a document, article, database, filing, opinion, report, press release, or other source stated.

Examples:

- `The Q3 report stated revenue was $25.8B.`
- `Spotrac listed the contract terms.`
- `The opinion was filed on June 4.`

Default leakage treatment: governed by source publication date.

`C7 Target Outcome Claim`

A claim that states, reveals, or directly entails the answer being predicted for the current instance.

Examples:

- `The petitioner prevailed.`
- `The player signed for $21.25M AAV.`
- `CVX had the highest six-month return.`

Default leakage treatment: leaked by category rule when the target outcome occurs after cutoff.

`C8 Post-Outcome Consequence Claim`

A claim about downstream effects, reactions, retrospective labels, or consequences that depend on the target outcome having occurred.

Examples:

- `The ruling became a landmark precedent.`
- `The contract was later viewed as a bargain.`
- `The stock surged after the pandemic crash.`

Default leakage treatment: leaked by category rule when the consequence depends on post-cutoff information.

### Exclusivity Rule

Natural-language claims can satisfy more than one class. Exclusivity is enforced through precedence:

```text
C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0
```

The classifier assigns the highest-priority applicable class. This order follows the most specific verification burden.

Examples:

- `The 2019 annual report stated revenue was $25.8B` is `C6`, not `C3`, because source publication controls availability.
- `Chevron had lower debt than peers` is `C4`, not `C2`, because the comparison is the verification burden.
- `The player signed for $21.25M AAV` is `C7` if that salary is the target outcome, even though it is also an event and a measurement.

### Practical Decision Tree

In implementation, the LLM should not be asked to choose among nine unordered labels. It should apply an ordered decision tree independently for each claim and stop at the first matching rule:

```text
1. If the claim states or directly entails the target answer, label C7.
2. Else if the claim describes a downstream effect or retrospective consequence of the target outcome, label C8.
3. Else if the claim is a definition, convention, logical truth, or mathematical identity, label C0.
4. Else if the claim is a durable entity identity or structural attribute, label C1.
5. Else if the claim is about what a document, report, filing, database, article, or opinion stated, label C6.
6. Else if the claim contains a numerical value, statistic, price, salary, revenue, margin, rank, probability, or measured quantity, label C3.
7. Else if the claim compares entities, groups, periods, or alternatives, label C4.
8. Else if the claim reports a localized event, announcement, filing, signing, decision, appointment, release, injury, transaction, or change, label C5.
9. Else label C2 as a time-indexed state.
```

This decision tree is sequential only within a single claim. Claims remain independent across items, so taxonomy and date verification can still be batched and parallelized across thousands of claims.

For validation runs, the judge may return `matched_rule` and a short explanation. For production runs, the explanation can be shortened or omitted to reduce output tokens.

### Completeness Argument

Every admissible factual claim is either time-independent or time-dependent.

Time-independent claims are definitions or stable attributes, covered by `C0` and `C1`.

Time-dependent claims either reveal the target (`C7`), reveal a post-target consequence (`C8`), refer to a source publication (`C6`), report a measurement (`C3`), compare entities or periods (`C4`), report a localized event (`C5`), or describe a state over time (`C2`).

`C2` is the residual class for time-dependent factual states. This keeps `C1` narrow and prevents stable-background labels from absorbing claims that require temporal verification.

## Date Verification

### Declared Date Versus Effective Date

Each evidence item contains a model-declared `source_date`. This date is useful but not sufficient. The detector audits whether the declared date is a plausible public-availability date for the claim class.

The detector stores:

```text
declared_source_date
effective_source_date
date_precision
date_verification_method
date_reason
date_confidence
```

### Plausibility Rules

The verifier applies claim-type-specific rules:

- `C3` stock price: the date may be the trading date.
- `C3` season statistic: the date cannot precede the date on which the relevant season or measurement window became complete.
- `C3` quarterly financial metric: the date cannot be the quarter end if the value became public only through an earnings release.
- `C3` annual financial metric: the date should correspond to an annual earnings release, annual report, or filing date.
- `C5` contract signing: the date should be the signing or public announcement date.
- `C5` court decision: the date should be the opinion, order, or filing date.
- `C6` publication claim: the date should be the publication date of the source.
- `C7` target outcome: leaked by category rule when the target resolves after cutoff.
- `C8` post-outcome consequence: leaked by category rule when the consequence depends on post-cutoff events.

### Effective Source Date Assignment

The effective source date is assigned by the following order:

```text
1. Use declared_source_date if it is plausible.
2. Use corrected_date if the verifier can correct an implausible declared date.
3. Use search_supported_date if search escalation resolves the claim.
4. Use conservative_latest_plausible_date if only coarse timing is available.
```

Coarse-date normalization uses conservative latest dates:

```text
YYYY       -> YYYY-12-31
YYYY-MM    -> last day of month
Qk YYYY    -> last day of quarter
early YYYY -> YYYY-04-30
mid YYYY   -> YYYY-08-31
late YYYY  -> YYYY-12-31
```

This policy prioritizes low false negatives. If a claim cannot be placed before the cutoff under a conservative public-availability rule, it is marked leaked.

## Leakage Detection Process

### Stage 1: Claim Normalization

Each evidence item is converted into a claim record:

```json
{
  "claim_uid": "...",
  "instance_id": "...",
  "claim_id": 1,
  "claim_text": "...",
  "declared_source_date": "YYYY-MM-DD",
  "cutoff_date": "YYYY-MM-DD",
  "task": "stock | salary | legal | ...",
  "target_description": "...",
  "model": "...",
  "method": "..."
}
```

The detector validates that `claim_text` is non-empty and that `declared_source_date` is parseable. Missing dates are escalated unless the claim is classified as `C0` or stable `C1`.

### Stage 2: Taxonomy Classification

Assign one class from `C0` to `C8` using the precedence rule. The classifier must use task context and target description because `C7` and `C8` are instance-relative.

### Stage 3: Category Shortcuts

Apply deterministic rules:

```text
C7 -> leaked
C8 -> leaked
C0 -> not leaked
C1 -> not leaked unless time-sensitive
```

All other categories continue to date plausibility verification.

### Stage 4: Date Plausibility Verification

For `C2-C6` and suspicious `C1`, verify whether the declared date is plausible. The verifier returns structured JSON:

```json
{
  "claim_id": 1,
  "category": "C3",
  "declared_date_plausible": true,
  "corrected_date": null,
  "date_precision": "exact",
  "confidence": "high",
  "reason": "The declared date matches the final date on which the season statistic became available."
}
```

### Stage 5: Search Escalation

Search is optional and selective. It is triggered when:

- `source_date` is missing;
- declared date is implausible and the verifier cannot correct it;
- verifier confidence is low;
- declared or corrected date is close to cutoff;
- category is high-risk, especially `C3`, `C4`, `C6`, `C7`, or `C8`;
- the claim has high Shapley importance in a later refinement pass.

Search returns source snippets and candidate dates. The date verifier then extracts the earliest reliable public-availability date or a conservative latest plausible date.

### Stage 6: Binary Leakage Label

The final claim-level decision is:

```text
is_leaked = effective_source_date > cutoff_date
```

The label is binary, but all supporting metadata is retained.

## Output Schema

Each claim receives:

```json
{
  "claim_id": 3,
  "claim_text": "...",
  "category": "C3",
  "declared_source_date": "2018-04-11",
  "effective_source_date": "2018-04-11",
  "is_leaked": false,
  "verification_method": "date_plausibility",
  "date_precision": "exact",
  "confidence": "high",
  "reason": "Season statistic date matches final regular-season availability."
}
```

Instance-level output contains the list of claim decisions plus summary counts:

```json
{
  "instance_id": "...",
  "cutoff_date": "YYYY-MM-DD",
  "total_claims": 30,
  "leaked_claims": 2,
  "leakage_rate": 0.0667,
  "claims": []
}
```

## Prompt Specifications

### Taxonomy and Date Plausibility Prompt

This prompt should be used with a low-cost structured-output judge. The recommended model is `google/gemini-3.1-flash-lite` through OpenRouter.

System message:

```text
You are a temporal provenance auditor. Your task is to classify evidence claims by temporal availability type and verify whether each declared source date is a plausible public-availability date for the claim.

You do not judge factual truth. Assume the claim may be true. Your task is only to determine when the information asserted by the claim would first have been publicly knowable.

Return only valid JSON matching the requested schema.
```

User message:

```text
Cutoff date: {cutoff_date}
Task: {task}
Target outcome description: {target_description}

Taxonomy:
C0 Timeless Definition: definition, convention, logic, or mathematical identity.
C1 Stable Entity Attribute: durable identity, membership, or structural attribute.
C2 Time-Indexed State: condition, status, exposure, quality, or relation at a time or during a period.
C3 Quantitative Measurement: numeric value, statistic, price, score, salary, revenue, margin, rank, vote count, or measured quantity.
C4 Comparative or Relational Claim: comparison across entities, periods, groups, or alternatives.
C5 Discrete Event: localized occurrence, action, decision, filing, appointment, signing, announcement, injury, transaction, release, or change.
C6 Publication or Source Claim: claim about what a document, filing, report, opinion, database, article, or source stated.
C7 Target Outcome Claim: states, reveals, or directly entails the answer being predicted for this instance.
C8 Post-Outcome Consequence Claim: downstream effect, reaction, retrospective label, or consequence depending on the target outcome.

Precedence:
C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0

Decision procedure:
Apply the following rules independently to each item. Stop at the first matching rule.
1. If the claim states or directly entails the target answer, label C7.
2. Else if the claim describes a downstream effect or retrospective consequence of the target outcome, label C8.
3. Else if the claim is a definition, convention, logical truth, or mathematical identity, label C0.
4. Else if the claim is a durable entity identity or structural attribute, label C1.
5. Else if the claim is about what a document, report, filing, database, article, or opinion stated, label C6.
6. Else if the claim contains a numerical value, statistic, price, salary, revenue, margin, rank, probability, or measured quantity, label C3.
7. Else if the claim compares entities, groups, periods, or alternatives, label C4.
8. Else if the claim reports a localized event, announcement, filing, signing, decision, appointment, release, injury, transaction, or change, label C5.
9. Else label C2 as a time-indexed state.

Date verification rule:
Treat declared_source_date as a candidate public-availability date, not ground truth. Check whether it is plausible for the assigned category. If implausible, provide a corrected date if possible. If only coarse timing is possible, use the conservative latest plausible date.

Items:
{items_json}

Output JSON:
{
  "results": [
    {
      "id": 1,
      "category": "C0|C1|C2|C3|C4|C5|C6|C7|C8",
      "matched_rule": 1,
      "declared_date_plausible": true,
      "corrected_date": null,
      "effective_source_date": "YYYY-MM-DD",
      "date_precision": "exact|declared|corrected|inferred|conservative|category_rule",
      "is_leaked": false,
      "confidence": "high|medium|low",
      "reason": "brief explanation"
    }
  ]
}
```

### Search Escalation Prompt

Search is handled by a separate search API. Search results are passed back into the verifier.

System message:

```text
You are a temporal date extraction auditor. Given a claim and search snippets, identify the earliest reliable public date on which the information asserted by the claim became knowable.

Do not judge whether the claim is true beyond the provided snippets. If snippets are insufficient, return a conservative latest plausible date and explain the uncertainty.

Return only valid JSON.
```

User message:

```text
Claim: {claim_text}
Declared source date: {declared_source_date}
Category: {category}
Cutoff date: {cutoff_date}
Task: {task}

Search snippets:
{search_snippets}

Return:
{
  "effective_source_date": "YYYY-MM-DD",
  "date_precision": "exact|corrected|inferred|conservative",
  "confidence": "high|medium|low",
  "is_leaked": true,
  "reason": "brief explanation"
}
```

## Model Choice and Cost

The recommended model for taxonomy and date-plausibility verification is `google/gemini-3.1-flash-lite` on OpenRouter.

At the listed OpenRouter price of `$0.25/M` input tokens and `$1.50/M` output tokens, this component is expected to be inexpensive for 500 questions with 30 claims each.

For 15,000 raw claims, assuming batches of 100 claims:

- about 150 judge calls before deduplication;
- roughly 0.9M to 1.5M input tokens;
- roughly 0.6M to 1.1M output tokens;
- estimated judge cost around `$1.50-$3.00`, with a conservative budget below `$5.00` including retries.

Gemini 3.1 Flash Lite should not be assumed to have native online search through OpenRouter. Search should be implemented as a separate tool or API. The verifier consumes search snippets only for escalated claims.

## Efficiency Plan

For 500 questions with 30 claims each, the raw claim count is 15,000. The pipeline should process claims globally rather than instance by instance.

### Flatten

Create a single global claim table containing all claims from all instances, methods, and models.

### Deduplicate

Normalize and deduplicate claims before any API call. Cache keys should include:

```text
normalized_claim_text
declared_source_date
task
target_description when needed
```

### Batch

Send 50 to 100 claims per taxonomy/date plausibility call. This keeps prompt size manageable while reducing per-call overhead.

Each batch prompt should instruct the judge to apply the decision tree separately to each claim. The ordered rules enforce exclusivity per claim; batching does not couple labels across claims.

### Parallelize

Taxonomy and date plausibility checks are claim independent. Run batches concurrently under API rate limits. Search escalation should use a smaller concurrency limit.

Suggested starting values:

```text
taxonomy_batch_size = 100
taxonomy_concurrency = 20
search_concurrency = 10
```

### Search Selectively

Search only for claims where cheap verification is insufficient. Expected search volume should be much lower than total claim volume, ideally below 10 to 20 percent after deduplication.

### Cache Persistently

Maintain persistent caches for:

```text
taxonomy_cache
date_plausibility_cache
search_result_cache
search_date_extraction_cache
```

The same evidence claims often recur across methods, models, and baselines, so persistent caching can reduce cost substantially.

## Expected Runtime

For 15,000 raw claims:

- Gemini-only taxonomy and date plausibility: about 150 batched calls before deduplication.
- Sequential execution: roughly 10 to 30 minutes depending on latency.
- Parallel execution with 20 workers: roughly 1 to 4 minutes under permissive rate limits.
- Production estimate with retries and rate limits: roughly 5 to 10 minutes.

Search escalation runtime depends on the external search API. If 10 to 20 percent of claims escalate, and search runs at 10 to 20 concurrent requests, a practical estimate is 5 to 20 additional minutes.

## Relation to Metrics

This module outputs the leakage labels used by downstream metric computation:

```text
OLR = mean(is_leaked_i)
Shapley-DCLR = sum(abs(phi_i) * is_leaked_i) / sum(abs(phi_i))
```

The taxonomy and leakage verifier do not compute Shapley values. They produce the per-claim leakage labels and metadata that Shapley-DCLR later consumes.

## Design Rationale

This design improves over date string matching by distinguishing three dates:

- the date mentioned in the claim text;
- the date declared by the model as `source_date`;
- the date on which the claim became publicly knowable.

The claim taxonomy determines which public-availability rule applies. This reduces false negatives by rejecting implausibly early declared dates, such as quarter-end dates for financial results that were released later. It also reduces false positives by using model-declared source dates when they are plausible, rather than blindly mapping vague phrases such as `late 2019` to the latest possible date.

The final leakage label remains binary, so metric aggregation stays simple. The supporting metadata makes each binary decision auditable.
