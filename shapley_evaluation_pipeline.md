# Shapley-DCLR Evaluation Pipeline — Design Document

**Paper:** All Leaks Count, Some Count More
**Purpose:** General-purpose evaluation pipeline that takes structured generation output from any model × method and computes Shapley-DCLR, OLR, and associated metadata.
**Last updated:** 2026-05-09

**Companion spec:** `LeakageEvaluationFramework/taxonomy_and_leakage_detection.md` is the authoritative design for claim taxonomy and binary leakage detection. This document uses that module's `C0-C8` taxonomy, effective-source-date verifier, and binary `is_leaked` labels.

---

## 1. Pipeline Overview

The production pipeline is **leakage-first**. Taxonomy and leakage detection run for every evidence claim because they are cheap, batched, and necessary for OLR. Shapley computation is expensive, so it is only run when it can affect Shapley-DCLR.

```text
If an instance has no leaked evidence claims:
    OLR = 0
    Shapley-DCLR = 0
    skip Shapley computation

If an instance has at least one leaked evidence claim:
    run cost-controlled Leverage SHAP
    compute Shapley-DCLR from abs(phi_i) and leakage labels
```

This gating is exact for DCLR: when all leakage labels are zero, the numerator of Shapley-DCLR is zero regardless of Shapley values.

```
INPUT (evidence, prediction, rationale — from any model × method)
         │
    ┌────▼────┐
    │ Stage 1 │  Taxonomy + Leakage Detection
    │         │  → applies C0-C8 taxonomy and assigns binary is_leaked per evidence item
    └────┬────┘
         │          (Shapley is skipped when no leaked claims are found)
    ┌────▼────┐
    │ Stage 2 │  Shapley Value Computation (Leverage SHAP)
    │         │  → computes φ_i for each evidence item via regression
    └────┬────┘
         │
    ┌────▼────┐
    │ Stage 3 │  Metric Aggregation
    │         │  → combines φ_i + ℓ(c_i) → Shapley-DCLR, OLR, metadata
    └────┬────┘
         │
       OUTPUT
```

### Production Execution Order

Production runs leakage detection before Shapley. This avoids unnecessary coalition evaluation: once all leakage labels are zero, Shapley-DCLR is known exactly to be zero.

```text
1. Parse structured outputs.
2. Normalize evidence items into claim records.
3. Run taxonomy + source-date leakage detection for all claims using taxonomy_and_leakage_detection.md.
4. Compute OLR immediately.
5. If no leaked claims: set Shapley-DCLR = 0 and skip Shapley.
6. If leaked claims exist: run Leverage SHAP with cost-controlled settings.
7. Aggregate Shapley-DCLR and emit claim-level metadata.
8. Run high-budget Shapley only for validation/audit subsets.
```

### Production Defaults

```yaml
leakage:
  judge_model: google/gemini-3.1-flash-lite
  taxonomy_batch_size: 100
  taxonomy_concurrency: 20
  search_concurrency: 10
  search_only_on_escalation: true

shapley:
  skip_if_no_leakage: true
  evaluator_backend: tinker
  evaluator_model_policy: same_as_generator
  renderer_name_policy: same_as_generator
  temperature: 0.0
  batch_different_coalitions: false
  coalition_budget_C: 15
  coalition_budget_cap: 1200
  exact_threshold_n: 8
  high_rigor_C: 25
  high_rigor_subset_size: 50
```

The default Shapley mode keeps coalitions isolated, one coalition per Tinker call, to avoid cross-coalition contamination. It also uses the same model or checkpoint family that generated the original output, so coalition predictions are attributed to the same predictive behavior as the evaluated method. Batched multi-coalition prompts can be enabled only after validation shows that batch outputs match single-coalition outputs.

---

## 2. Input Schema (Per Instance)

Every instance entering this pipeline has a standardized format, regardless of which model or method generated it.

```python
@dataclass
class EvalInput:
    # ─── Instance metadata ───
    instance_id: str
    task: str                             # "legal" | "stock" | "salary" | "forecast"
    question: str                         # full question text
    cutoff_date: str                      # YYYY-MM-DD temporal reference
    ground_truth: dict                    # task-specific ground truth

    # ─── From generation phase (model × method output) ───
    evidence: list[dict]                  # [{"id": int, "fact": str, "source_date": str|None}, ...]
    prediction: dict                      # task-specific prediction (ranking/salary/probability)
    rationale: str                        # free-text reasoning
    
    # ─── Run metadata ───
    model: str                            # model identifier
    method: str                           # method identifier, e.g. temporal_hint/rag/timespec/tempo
```

The **evidence items are the players** (claims) in the Shapley game. Each `evidence[i]["fact"]` is one atomic claim `c_i`.

---

## 3. Output Schema (Per Instance)

```python
@dataclass
class EvalOutput:
    # ─── Instance identity ───
    instance_id: str
    model: str
    method: str
    task: str

    # ─── Core metrics ───
    olr: float                            # Overall Leakage Rate (unweighted)
    shapley_dclr: float                   # Shapley-weighted Decision-Critical Leakage Rate
    performance: float                    # prediction quality vs ground truth

    # ─── Shapley details ───
    n_claims: int                         # number of evidence items (players)
    shapley_values: list[float] | None    # φ_i for each claim, null if Shapley skipped
    leakage_labels: list[int]             # ℓ(c_i) ∈ {0, 1} for each claim
    claim_categories: list[str]           # κ_i ∈ {C0..C8} from taxonomy_and_leakage_detection.md

    # ─── Computation metadata ───
    v_empty: float                        # normalized v(∅), always 0.0
    v_full: float | None                  # normalized closed-world v(N), null if Shapley skipped
    budget_used: int                      # number of v(S) evaluations performed
    confidence_se: list[float] | None     # optional bootstrap standard error per φ_i
```

---

## 4. Stage 1: Taxonomy + Leakage Detection

### Purpose
Determine the taxonomy class κ(c_i) and binary leakage status ℓ(c_i) ∈ {0, 1} for each evidence item.

This stage is specified in detail in `taxonomy_and_leakage_detection.md`. The key production rule is:

```text
is_leaked(c, t_ref) = 1[effective_source_date(c) > t_ref]
```

where `effective_source_date` is the model-declared `source_date` after claim-type-aware plausibility checking, correction, and optional search escalation.

### Inputs

Each evidence item is converted into a claim record:

```json
{
  "claim_id": 1,
  "claim_text": "...",
  "declared_source_date": "YYYY-MM-DD",
  "cutoff_date": "YYYY-MM-DD",
  "task": "legal|stock|salary",
  "target_description": "...",
  "model": "...",
  "method": "..."
}
```

### Taxonomy

The taxonomy is the `C0-C8` taxonomy from the companion spec:

| Class | Meaning | Default leakage handling |
|---|---|---|
| C0 | Timeless definition | clean |
| C1 | Stable entity attribute | clean unless time-sensitive |
| C2 | Time-indexed state | verify effective source date |
| C3 | Quantitative measurement | verify effective source date |
| C4 | Comparative or relational claim | verify effective source date |
| C5 | Discrete event | verify event/publication date |
| C6 | Publication or source claim | verify source publication date |
| C7 | Target outcome claim | leaked by category rule |
| C8 | Post-outcome consequence claim | leaked by category rule |

Labels are assigned by the ordered decision tree in `taxonomy_and_leakage_detection.md`; claims can be processed independently and batched across instances.

### Logic

```python
def stage_1_taxonomy_and_leakage(
    evidence,
    cutoff_date,
    task,
    target_description,
    judge_client,
    search_client,
    cache,
):
    """
    Input: evidence list with fact/source_date, task context, cutoff date.
    Output: per-claim taxonomy class, effective source date, binary leakage label, metadata.
    """
    claims = normalize_evidence_to_claim_records(
        evidence=evidence,
        cutoff_date=cutoff_date,
        task=task,
        target_description=target_description,
    )

    # Batch across claims and instances in production.
    # The judge applies the C0-C8 decision tree and checks date plausibility.
    first_pass = batched_taxonomy_and_date_plausibility(
        claims=claims,
        judge_client=judge_client,
        cache=cache.taxonomy_date_cache,
    )

    # Escalate only hard claims: missing date, implausible date without correction,
    # low confidence, near-cutoff date, or high-risk category.
    escalated = select_search_escalations(first_pass)
    search_results = search_and_extract_dates(
        claims=escalated,
        search_client=search_client,
        judge_client=judge_client,
        cache=cache.search_cache,
    )

    decisions = merge_first_pass_and_search(first_pass, search_results)

    for d in decisions:
        if d.category in {"C7", "C8"}:
            d.is_leaked = True
        elif d.category in {"C0", "C1"} and not d.time_sensitive:
            d.is_leaked = False
        else:
            d.is_leaked = d.effective_source_date > cutoff_date

    return decisions
```

### Output

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

### Cost Controls

Stage 1 is cheap relative to Shapley:

- flatten and deduplicate claims across all instances;
- batch 50-100 claims per Gemini 3.1 Flash Lite call;
- apply C0/C1/C7/C8 shortcuts;
- use search only on escalation;
- cache taxonomy, date plausibility, search results, and extracted dates.

See `taxonomy_and_leakage_detection.md` for prompt details, model choice, runtime estimates, and cache keys.

---

## 5. Stage 2: Shapley Value Computation

Stage 2 computes claim-level Shapley values only for instances where Stage 1 found at least one leaked evidence claim. If all `is_leaked` labels are false, then Shapley-DCLR is exactly zero and this stage is skipped.

### Purpose

Compute `phi_i` for each evidence item, measuring how much claim `i` contributes to normalized above-baseline prediction quality under a closed-world evaluator.

### Players

For each question, the players are the evidence items from the structured output:

```text
N = {1, ..., n}
player i = evidence[i].fact
```

The Shapley module does not extract additional rationale claims. Rationale-only leakage can be audited separately, but Shapley-DCLR uses evidence claims as players.

### Value Function

For a coalition `S ⊆ N`, define `v(S)` as the normalized above-baseline task score obtained when a closed-world evaluator predicts using only evidence claims in `S`.

```text
v(S) = (score(S) - score_base) / (score_ideal - score_base)
```

This gives:

```text
v(empty) = 0
v(ideal) = 1
```

Values may be negative when the coalition performs worse than the baseline.

Use the same closed-world evaluator for every non-empty coalition, including the full coalition. By default, this evaluator should be the same model or model checkpoint that generated the structured output. Do not use the original generator prediction as `v(full)`, because that would mix a free-form generation value with coalition-conditioned closed-world values. Store the original prediction separately and compute a reconstruction diagnostic:

```text
full_reconstruction_gap = distance(original_prediction, closed_world_full_prediction)
```

Task-normalized values:

```text
legal:
  score(S) = 1 - (p_hat - y)^2
  score_base = 0.75              # p_hat = 0.5
  v(S) = (score(S) - 0.75) / 0.25

stock:
  score(S) = (rho + 1) / 2
  score_base = 0.5               # random ranking, rho = 0
  v(S) = rho

salary:
  score(S) = max(0, 1 - abs(predicted_salary - actual_salary) / actual_salary)
  score_base = score from a fixed pre-declared salary baseline
  v(S) = (score(S) - score_base) / max(1 - score_base, eps)
```

For stock, negative `rho` is retained. A negative value means the coalition supports a ranking worse than random, and the corresponding claims may receive negative Shapley values.

### Closed-World Coalition Prompt

For every non-empty coalition, the evaluator receives the task context plus only the selected claims:

```text
CLOSED-WORLD CONSTRAINT:
You MUST base your prediction exclusively on the VALIDATED FACTS listed below.
Do NOT introduce facts, numbers, events, or entities not present in the list.
If the facts are insufficient, make the best prediction possible using only them.

VALIDATED FACTS:
[2] ...
[3] ...
[18] ...

Return only the task-specific JSON prediction.
```

For legal:

```json
{"probability_petitioner": 0.0}
```

For salary:

```json
{"predicted_salary": 0}
```

For stock:

```json
{"ranking": ["TICKER1", "TICKER2", "..."]}
```

### Exact Versus Approximate Shapley

If `2^n` is affordable, compute exact Shapley by evaluating every coalition.

Recommended rule:

```text
if n <= 10: exact enumeration
else: Leverage SHAP
```

For example:

```text
n = 5  -> 32 coalitions, exact
n = 18 -> 262,144 coalitions, Leverage SHAP
n = 30 -> 1.07B coalitions, Leverage SHAP
```

### Leverage SHAP Sampling

Leverage SHAP differs from KernelSHAP. KernelSHAP samples rows according to the SHAP kernel, which overweights very small and very large coalitions. Leverage SHAP samples rows according to statistical leverage scores. For the Shapley regression matrix, the leverage score of a coalition of size `s` is proportional to:

```text
1 / C(n, s)
```

Because there are `C(n, s)` coalitions of size `s`, this gives roughly equal total mass to each coalition size.

Practical sampler:

```text
1. Reserve empty and full coalition evaluations.
2. For each pair:
   a. sample size s uniformly from {1, ..., n-1};
   b. sample S uniformly among subsets of size s;
   c. include S and its complement N \ S.
3. Avoid duplicate masks.
4. Stop when the evaluation budget is reached.
```

Budget:

```text
m_total = min(cap, max(2*n + 2, floor(C * n * log(n))))
```

Recommended starting values:

```text
production C = 15
production cap = 1200
high-rigor audit C = 25
high-rigor audit cap = 2000
```

Example for `eval_0023` with `n = 18`:

```text
production m_total ≈ 15 * 18 * log(18) ≈ 780
2 evaluations reserved for empty and full
778 non-boundary evaluations
389 paired samples
```

Example sampled pair:

```text
S       = {2, 3, 18}
N \ S   = {1,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
```

### Projected Regression

After sampled coalitions are evaluated, build binary design matrix `Z`:

```text
Z[j, i] = 1 if claim i is in coalition S_j else 0
```

Let `y_j = v(S_j)` because `v(empty)=0`. Let:

```text
delta = v(full)
one = all-ones vector in R^n
base = (delta / n) * one
P = I - (1/n) * one * one^T
b = y - Z * base
```

Solve the projected weighted regression:

```text
theta = argmin_theta || W^{1/2} (Z P theta - b) ||_2^2
phi = P theta + base
```

This enforces efficiency by construction:

```text
sum_i phi_i = v(full)
```

For a fixed-budget implementation, the row weight should match the actual inclusion probability of the sampled row. If size is sampled uniformly and subset is sampled uniformly within that size, then:

```text
p(S) = 1 / ((n - 1) * C(n, |S|))
```

The full Shapley regression row scaling uses:

```text
w_shap(S) = 1 / (C(n, |S|) * |S| * (n - |S|))
```

The sampled regression weight is proportional to:

```text
w(S) = w_shap(S) / p(S)
```

If using paired without-replacement sampling, the implementation should use the actual pair inclusion probability. The simplified formula above is acceptable as the first engineering implementation and should be validated by a convergence study.

### Tinker Evaluation Mode

Coalition evaluation is the dominant cost. The default mode uses **one coalition per Tinker call** to avoid cross-coalition contamination. Prompts must be compact and prediction-only.

```text
Task: legal
Question: {compact question context}
Facts:
1. ...
2. ...

Use only these facts. Return JSON only: {"p": number}
```

Do not include source dates, leakage labels, taxonomy labels, or rationale requests in Shapley prompts unless needed by the task itself.

Optional multi-coalition batching can be enabled only after validation shows that batched predictions match single-coalition predictions. If enabled, use a Claim ID Reference System:

```text
CLAIMS:
[1] ...
[2] ...
...
[n] ...

SETS:
SET 1: [2, 3, 18]
SET 2: [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
...

For each SET, predict using ONLY the listed claim IDs.
Return a JSON array with exactly one prediction per SET.
```

Default single-coalition settings:

```text
batch_different_coalitions = false
max_tinker_concurrency = 20 to 100, depending on rate limits
temperature = 0
max_tokens = task-specific short JSON output
```

### Caching

Cache every coalition evaluation:

```text
cache_key = hash(instance_id, evaluator_model, prompt_version, coalition_mask)
```

Also cache full coalition predictions because they are used both as `v(full)` and for reconstruction diagnostics.

### Output Metadata

Per instance:

```json
{
  "n_claims": 18,
  "estimator": "exact|leverage_shap_projected_regression",
  "m_total": 78,
  "batch_different_coalitions": false,
  "v_empty": 0.0,
  "v_full": 0.42,
  "original_prediction": {},
  "closed_world_full_prediction": {},
  "full_reconstruction_gap": 0.0,
  "num_tinker_calls": 77,
  "claim_shapley": []
}
```

Per claim:

```json
{
  "claim_id": 18,
  "shapley_value": -0.041,
  "abs_shapley_value": 0.041,
  "shapley_rank": 1,
  "bootstrap_se": 0.006
}
```

### Cost and Runtime Notes

For 500 questions with about 30 claims each:

```text
claims per question: ~30
production coalitions per leaking question: 1200 with C=15 and cap=1200
Tinker calls per leaking question: ~1199 because empty coalition requires no call
```

Wall-clock time is governed by Tinker latency, retry rate, and concurrency. The production default reduces cost by skipping Shapley for no-leak instances and using compact prompts. High-budget `C=25` runs are reserved for validation and case studies.

Token cost depends on Tinker pricing and prompt size. The single-coalition prompt repeats selected facts, but avoids cross-coalition contamination. Multi-coalition batching can reduce cost further but must be audited before use.

---

## 6. Stage 3: Metric Aggregation

```python
def stage_3_aggregate(shapley_values, leakage_labels, prediction, task, ground_truth):
    """
    Combine Shapley values with leakage labels to produce final metrics.
    Pure local computation — no LLM calls.
    """
    labels = np.array(leakage_labels)
    
    # OLR: unweighted leakage rate
    olr = float(labels.mean()) if len(labels) > 0 else 0.0

    # If no leaked claims, DCLR is exactly zero and Shapley may be skipped.
    if labels.sum() == 0:
        return {
            "olr": olr,
            "shapley_dclr": 0.0,
            "performance": performance_score(task, prediction, ground_truth),
            "shapley_status": "skipped_no_leakage",
        }

    if shapley_values is None:
        raise ValueError("Shapley values are required when leaked claims exist.")

    phi = np.array(shapley_values)
    
    # Shapley-DCLR: importance-weighted leakage rate
    abs_phi = np.abs(phi)
    total_influence = abs_phi.sum()
    if total_influence < 1e-10:
        shapley_dclr = 0.0
    else:
        shapley_dclr = float((abs_phi * labels).sum() / total_influence)
    
    # Performance: prediction quality vs ground truth
    performance = performance_score(task, prediction, ground_truth)
    
    return {
        "olr": olr,
        "shapley_dclr": shapley_dclr,
        "performance": performance,
        "shapley_status": "computed",
    }
```

---

## 7. Production Orchestration

The production runner is organized around one principle: perform cheap global work first, then invoke expensive Shapley calls only where they can change the result.

```text
Run-level input:
  one method output file or directory
  one dataset metadata file with question, cutoff, and ground truth
  one run config

Run-level output:
  per-instance JSONL
  per-claim JSONL
  aggregate summary JSON
  caches and audit logs
```

### Orchestration Steps

```text
1. Load all method outputs and dataset metadata.
2. Parse each structured completion into evidence, prediction, and reasoning.
3. Build EvalInput records.
4. Flatten all evidence claims across the run.
5. Run Stage 1 taxonomy and leakage globally with batching, deduplication, and caching.
6. Attach claim decisions back to each instance.
7. Compute OLR for every instance.
8. For instances with no leaked claims:
     shapley_status = skipped_no_leakage
     shapley_dclr = 0
     shapley_values = null or []
9. For instances with leaked claims:
     sample coalitions
     evaluate coalitions with Tinker
     solve exact or Leverage SHAP
     compute Shapley-DCLR
10. Write per-claim, per-instance, and summary outputs incrementally.
11. Run validation/audit jobs on selected subsets.
```

### Pseudocode

```python
async def run_evaluation_job(config: EvalJobConfig) -> None:
    dataset = load_dataset_metadata(config.dataset_path)
    outputs = load_method_outputs(config.input_path)
    eval_inputs = build_eval_inputs(dataset, outputs)

    # Stage 1: global, cache-first, batched leakage detection.
    all_claims = flatten_claims(eval_inputs)
    claim_decisions = await taxonomy_leakage_detector.run_global(all_claims)
    attach_claim_decisions(eval_inputs, claim_decisions)

    for instance in eval_inputs:
        instance.olr = mean([c.is_leaked for c in instance.claims])

        if not any(c.is_leaked for c in instance.claims):
            result = build_no_leak_result(instance)
        else:
            shapley = await shapley_runner.run(instance)
            result = aggregate_result(instance, shapley)

        write_incremental_outputs(result)

    write_summary()
```

### Concurrency

Stage 1 concurrency is claim-batch based. Stage 2 concurrency is instance/coalition based.

```yaml
taxonomy_concurrency: 20
search_concurrency: 10
shapley_instance_concurrency: 4-16
tinker_call_concurrency: 20-100
```

Keep Tinker concurrency configurable because actual throughput depends on account limits, model latency, and retry behavior.

---

## 8. Implementation Structure

All future implementation code should live under `LeakageEvaluationFramework`.

### Package Layout

```text
LeakageEvaluationFramework/
  README.md
  shapley_evaluation_pipeline.md
  taxonomy_and_leakage_detection.md

  leakage_eval/
    __init__.py
    cli.py
    config.py
    schemas.py

    io/
      __init__.py
      loaders.py
      writers.py
      jsonl.py

    taxonomy/
      __init__.py
      classifier.py
      prompts.py
      date_verifier.py
      search_escalation.py
      cache.py

    shapley/
      __init__.py
      value_function.py
      coalition_sampler.py
      leverage_regression.py
      exact.py
      runner.py
      prompts.py

    evaluators/
      __init__.py
      tinker_client.py
      openrouter_client.py
      parsers.py

    metrics/
      __init__.py
      performance.py
      leakage.py
      dclr.py
      aggregation.py

    audit/
      __init__.py
      convergence.py
      evaluator_sensitivity.py
      batch_contamination.py

  configs/
    leakage_eval_default.yaml
    shapley_qwen30b_default.yaml
    audit_high_rigor.yaml

  runtime/
    cache/
      taxonomy_cache.jsonl
      date_plausibility_cache.jsonl
      search_cache.jsonl
      coalition_value_cache.jsonl
    results/
      <job_id>/
```

### Main Responsibilities

`schemas.py`

Defines `EvalInput`, `ClaimRecord`, `ClaimDecision`, `ShapleyResult`, `EvalOutput`, and summary schemas. These should be Pydantic models or dataclasses with JSON serialization.

`taxonomy/`

Implements the `C0-C8` decision tree, date plausibility verifier, selective search escalation, and persistent caches. This module follows `taxonomy_and_leakage_detection.md`.

`shapley/`

Implements normalized value functions, exact Shapley, Leverage SHAP sampling, projected regression, coalition evaluation, and Shapley metadata.

`evaluators/`

Wraps Tinker/OpenRouter calls and provides task-specific output parsers. Shapley should call this abstraction rather than raw APIs.

`metrics/`

Computes OLR, Shapley-DCLR, normalized task scores, reconstruction gaps, and aggregate summaries.

`audit/`

Runs convergence studies, evaluator sensitivity checks, and optional batch contamination checks.

### CLI Targets

```bash
python -m LeakageEvaluationFramework.leakage_eval.cli run \
  --config LeakageEvaluationFramework/configs/shapley_qwen30b_default.yaml \
  --input <method_output_dir_or_file> \
  --dataset <dataset_metadata_path> \
  --output LeakageEvaluationFramework/runtime/results/<job_id>

python -m LeakageEvaluationFramework.leakage_eval.cli audit-convergence \
  --config LeakageEvaluationFramework/configs/audit_high_rigor.yaml \
  --results LeakageEvaluationFramework/runtime/results/<job_id>

python -m LeakageEvaluationFramework.leakage_eval.cli summarize \
  --results LeakageEvaluationFramework/runtime/results/<job_id>
```

---

## 9. Results Structure

Each evaluation job writes a self-contained results directory.

```text
runtime/results/<job_id>/
  config.json
  manifest.json
  summary.json
  instance_results.jsonl
  claim_results.jsonl
  shapley_results.jsonl
  skipped_shapley.jsonl
  errors.jsonl

  audit/
    convergence.jsonl
    evaluator_sensitivity.jsonl
    batch_contamination.jsonl

  logs/
    run.log
    cost_estimate.json
    api_usage.jsonl
```

### `manifest.json`

```json
{
  "job_id": "...",
  "created_at": "...",
  "input_path": "...",
  "dataset_path": "...",
  "n_instances": 500,
  "models": ["Qwen/Qwen3-30B-A3B"],
  "methods": ["temporal_hint", "rag", "timespec", "tempo"],
  "taxonomy_spec": "taxonomy_and_leakage_detection.md",
  "shapley_spec": "shapley_evaluation_pipeline.md"
}
```

### `claim_results.jsonl`

One row per evidence claim:

```json
{
  "instance_id": "eval_0023",
  "claim_id": 18,
  "claim_text": "...",
  "declared_source_date": "2019-06-27",
  "category": "C5",
  "effective_source_date": "2019-06-27",
  "is_leaked": false,
  "verification_method": "date_plausibility",
  "confidence": "high",
  "reason": "..."
}
```

### `shapley_results.jsonl`

One row per instance where Shapley ran:

```json
{
  "instance_id": "eval_0023",
  "shapley_status": "computed",
  "estimator": "leverage_shap_projected_regression",
  "n_claims": 18,
  "m_total": 780,
  "v_empty": 0.0,
  "v_full": 0.42,
  "num_tinker_calls": 779,
  "original_prediction": {},
  "closed_world_full_prediction": {},
  "full_reconstruction_gap": 0.03,
  "claim_shapley": [
    {
      "claim_id": 18,
      "shapley_value": -0.041,
      "abs_shapley_value": 0.041,
      "shapley_rank": 1,
      "bootstrap_se": null
    }
  ]
}
```

### `skipped_shapley.jsonl`

One row per instance where Shapley was skipped:

```json
{
  "instance_id": "eval_0001",
  "reason": "no_leaked_claims",
  "olr": 0.0,
  "shapley_dclr": 0.0
}
```

### `instance_results.jsonl`

One row per instance:

```json
{
  "instance_id": "eval_0023",
  "task": "legal",
  "model": "Qwen/Qwen3-30B-A3B",
  "method": "tempo",
  "n_claims": 18,
  "leaked_claims": 1,
  "olr": 0.0556,
  "shapley_dclr": 0.38,
  "performance": 0.82,
  "shapley_status": "computed",
  "full_reconstruction_gap": 0.03
}
```

### `summary.json`

```json
{
  "overall": {
    "n_instances": 500,
    "olr": {"mean": 0.012, "std": 0.041},
    "shapley_dclr": {"mean": 0.009, "std": 0.037},
    "performance": {"mean": 0.74, "std": 0.12},
    "shapley_computed": 63,
    "shapley_skipped_no_leakage": 437,
    "estimated_cost_usd": 3.42
  },
  "by_task": {},
  "by_model": {},
  "by_method": {}
}
```

---

## 10. Cost Model

### Production Cost Model

The old full-Shapley estimate below is not the recommended execution plan. The production plan minimizes cost through four mechanisms:

1. Run Shapley only for instances with at least one leaked claim.
2. Use a calibrated Leverage SHAP budget (`C=15`, cap 1200) based on pilot convergence sweeps.
3. Keep prompts compact and generate prediction-only JSON.
4. Use high-budget `C=25` only for validation and case-study subsets.

For 500 questions with about 30 claims each:

```text
coalitions per leaking question ≈ min(1200, 15 * 30 * log(30)) = 1200
empty coalition requires no model call
Tinker calls per leaking question ≈ 1199
```

Compact single-coalition prompt estimate:

```text
input tokens per call: 700-1,500
output tokens per call:
  legal: 8-12
  salary: 8-12
  stock: 25-40
```

Worst case, if all 500 questions have leakage:

```text
calls ≈ 500 * 1199 = 599,500
input tokens ≈ 420M-899M
output tokens ≈ 4.8M-24.0M
```

With OpenRouter Qwen3-30B-A3B reference pricing of roughly `$0.08/M` input tokens and `$0.28/M` output tokens, this is about `$35-$79` per 500-question worst-case run if every question contains leakage. Tinker cost should be computed from the current Tinker Console because Tinker prices distinguish prefill and sample tokens. Tinker documentation defines prefill as prompt-token processing and sample as output-token generation, and recommends MoE models for cost-effective inference.

Leakage-gated cost scales with the fraction of instances containing leaked claims:

| Leaking-instance rate | Approx. Shapley cost per 500-question run |
|---|---|
| 10% | ~$0.50-$1.00 |
| 20% | ~$1-$2 |
| 50% | ~$2.50-$5 |
| 100% | ~$5-$10 |

For multiple method/model runs:

```text
total_cost ≈ per_run_worst_case_cost * number_of_runs * leaking_instance_rate
```

Example for 10 runs:

```text
100% leaking -> ~$50-$100
20% leaking  -> ~$10-$20
```

Taxonomy/leakage detection is separate. For 15,000 claims, Gemini 3.1 Flash Lite taxonomy/date plausibility is expected to be a few dollars with batching and caching.

### Tinker Evaluator Choice

Use the **same model or checkpoint as the generator** as the default Shapley evaluator:

- For a Qwen3-30B output, use `Qwen/Qwen3-30B-A3B` or the corresponding trained sampler/checkpoint.
- For a Qwen3.5-35B output, use `Qwen/Qwen3.5-35B-A3B` or the corresponding trained sampler/checkpoint.
- For a Kimi output, use the matching Kimi model.
- For TEMPO outputs, use the same trained checkpoint/sampler when available.

This is the most faithful setting because all coalition predictions come from the same model family that produced the original structured answer. It reduces attribution error from evaluator mismatch. A shared cheap evaluator such as `Qwen/Qwen3-30B-A3B` remains useful as a cost-saving sensitivity mode, but it is not the default for paper-quality DCLR.

### Recommended Qwen/Kimi Experiment Models

If the experiment is restricted to Qwen and Kimi families, use two Qwen models and two Kimi models:

| Role | Model | Why use it | Cost note |
|---|---|---|---|
| Qwen cost-effective model | `Qwen/Qwen3-30B-A3B` | Best Qwen cost/quality tradeoff; MoE with low active parameter count; already integrated in this repository | Reference OpenRouter price: ~$0.08/M input, ~$0.28/M output |
| Stronger Qwen comparison | `Qwen/Qwen3.5-35B-A3B` | Newer/stronger Qwen-family model; still MoE and cost-conscious relative to dense large models | Reference OpenRouter price: ~$0.14-$0.1625/M input, ~$1.00-$1.30/M output |
| Kimi reasoning representative | `moonshotai/Kimi-K2-Thinking` | Strong reasoning model; useful for legal and complex closed-world prediction validation | Reference OpenRouter price: ~$0.60/M input, ~$2.50/M output |
| Newer Kimi comparison | `moonshotai/Kimi-K2.6` | Newer Kimi-family model with long context; useful as high-quality comparison | Reference OpenRouter price: ~$0.75/M input, ~$3.50/M output |

The exact Tinker IDs and prices must be verified in the Tinker Console before running. The public Tinker documentation defines the billing units (`prefill`, `sample`, and `train`) and notes that MoE models are priced by active parameters, but the attached public page does not expose the full current per-model table.

**Default recommendation:** use these four as generation/evaluation subjects and use the same model/checkpoint as the Shapley evaluator for each subject. This is more expensive than a shared evaluator, but it is the most faithful attribution setting. Use `Qwen/Qwen3-30B-A3B` as a shared evaluator only for a cost-saving sensitivity run.

### Shapley Cost By Evaluator Model

The estimates below use the production single-coalition setup:

```text
500 questions
~30 claims/question
C = 15
cap = 1200
~1199 model calls per leaking question
input/call = 700-1,500 tokens
output/call = 8-40 tokens
```

Worst case if every question has at least one leaked claim:

```text
calls ≈ 599,500
input tokens ≈ 420M-899M
output tokens ≈ 4.8M-24.0M
```

Estimated Shapley cost per 500-question run:

| Shapley evaluator | Reference input price | Reference output price | 100% leaking | 20% leaking | 10% leaking |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-30B-A3B` | ~$0.08/M | ~$0.28/M | ~$4.4-$10.0 | ~$0.9-$2.0 | ~$0.4-$1.0 |
| `Qwen/Qwen3.5-35B-A3B` | ~$0.14-$0.1625/M | ~$1.00-$1.30/M | ~$8.0-$22.4 | ~$1.6-$4.5 | ~$0.8-$2.2 |
| `moonshotai/Kimi-K2-Thinking` | ~$0.60/M | ~$2.50/M | ~$33.3-$75.9 | ~$6.7-$15.2 | ~$3.3-$7.6 |
| `moonshotai/Kimi-K2.6` | ~$0.75/M | ~$3.50/M | ~$41.9-$96.0 | ~$8.4-$19.2 | ~$4.2-$9.6 |

These are Shapley-only estimates. Taxonomy and leakage detection are separate and expected to cost only a few dollars for 15,000 raw claims when using Gemini 3.1 Flash Lite with batching.

If all four Qwen/Kimi models are used as generation subjects and each model uses its **own matching Shapley evaluator**, the cost is approximately the sum of each row in the table above, multiplied by the number of methods:

```text
per_method_cost_for_4_models ≈
  Qwen3-30B cost
  + Qwen3.5-35B cost
  + Kimi-K2-Thinking cost
  + Kimi-K2.6 cost
```

For example, four models and five methods produce 20 model-method runs:

```text
100% leaking: ~$438-$1,022
20% leaking:  ~$88-$204
10% leaking:  ~$44-$102
```

If all outputs use a **shared Qwen3-30B-A3B Shapley evaluator**, the cost is much lower but attribution is less faithful to each generator model:

```text
total_shared_evaluator_cost ≈ $4.4-$10.0 * num_500_question_runs * leaking_instance_rate
```

Use the matching-evaluator setup for final reported paper metrics. Use the shared-evaluator setup only as a cheaper ablation/sensitivity analysis.

Optional cheaper evaluator candidates, if non-Tinker fallback is allowed:

| Candidate | Use |
|---|---|
| Qwen3-30B-A3B | default quality/cost balance |
| Qwen3-14B | cheaper validation candidate |
| GPT-OSS-20B | cheapest fallback candidate |

Before changing the evaluator, run a calibration study measuring:

```text
rank_corr(abs_phi_matching, abs_phi_shared)
abs(DCLR_matching - DCLR_shared)
closed_world_full_prediction_gap
batch_vs_single_coalition_gap
```

### Prompt Compression Rules

For single-coalition Tinker calls:

```text
Task: legal
Question: compact task context
Facts:
1. ...
2. ...

Use only these facts. Return JSON only: {"p": number}
```

Do not include:

- rationale requests;
- source dates unless needed for the task prediction;
- taxonomy metadata;
- leakage labels;
- long explanations;
- repeated methodological instructions.

Maximum output tokens:

```yaml
legal: 8-12
salary: 8-12
stock: 25-40
```

### Optional Multi-Coalition Batch Mode

Multi-coalition batching is cheaper but risks cross-coalition contamination. It should be disabled by default:

```yaml
batch_different_coalitions: false
```

If enabled for a later cost-saving run:

```yaml
legal_batch_size: 32-64
salary_batch_size: 64-128
stock_batch_size: 64-128
duplicate_set_audit_rate: 0.02
single_call_audit_rate: 0.02
```

Enable it only if batch predictions match single-coalition predictions on the audit subset.

### Historical Full-Shapley Cost (Not Default)

Running high-budget Shapley for every instance is not the production plan. It is retained only as an upper-bound warning.

If Shapley were run on every instance with a high budget, cost would scale as:

```text
total_calls ≈ num_instances * num_model_method_runs * C * n * log(n)
```

For 500 instances, 20 model-method runs, n≈30, and C=25:

```text
total_calls ≈ 500 * 20 * min(2000, 25 * 30 * log(30)) ≈ 20M coalition calls
```

This mode is unnecessary because no-leak instances have Shapley-DCLR exactly equal to zero. Use leakage-gated Shapley instead.

---

## 11. Validation and Audit Protocols

Run validation before the full experiment to calibrate budget, evaluator model, and optional batching. Validation is run on a subset, not the whole dataset.

### Budget Convergence

Goal: choose the smallest budget that gives stable Shapley-DCLR and stable high-influence claim rankings.

Recommended pilot:

```text
pilot_instances = 50 leaking or near-leaking instances
budgets = [50, 100, 150, 200, 300, 500]
reference_budget = 800 or exact when n <= 10
```

Use the production value function, Leverage SHAP sampler, and projected regression. Compare each lower budget against the reference.

```python
async def convergence_study(pilot_inputs: list[EvalInput], generate_fn, n_pilot=50):
    """
    Determine the minimum reliable coalition budget.
    """
    budgets = [50, 100, 150, 200, 300, 500]
    reference_budget = 800
    
    results = []
    
    for inp in pilot_inputs[:n_pilot]:
        n = len(inp.evidence)

        phi_ref = await run_shapley(
            inp,
            budget=reference_budget,
            estimator="leverage_shap_projected_regression",
        )
        dclr_ref = compute_dclr(phi_ref, inp.leakage_labels)

        for budget in budgets:
            phi_M = await run_shapley(
                inp,
                budget=budget,
                estimator="leverage_shap_projected_regression",
            )
            dclr_M = compute_dclr(phi_M, inp.leakage_labels)

            mae = np.mean(np.abs(phi_ref - phi_M))
            rank_corr = spearmanr(np.abs(phi_ref), np.abs(phi_M)).correlation
            
            results.append({
                "instance_id": inp.instance_id,
                "n": n,
                "budget": budget,
                "normalized_budget": budget / (n * math.log(n)),
                "dclr_error": abs(dclr_ref - dclr_M),
                "mae": mae,
                "rank_correlation": rank_corr,
            })
    
    return results
```

Suggested acceptance thresholds:

```text
mean dclr_error <= 0.02
median rank_corr(abs_phi) >= 0.90
top_5_overlap >= 0.80
```

### Evaluator Sensitivity

Compare the default Shapley evaluator against candidate cheaper/stronger evaluators on the same sampled instances and same coalition masks.

Report:

```text
abs(DCLR_default - DCLR_candidate)
rank_corr(abs_phi_default, abs_phi_candidate)
closed_world_full_prediction_gap
```

Use this to decide whether a cheaper evaluator is acceptable or whether Qwen3-30B-A3B remains the default.

### Optional Batch Contamination Audit

Multi-coalition batching is disabled by default. If enabled later, compare batched outputs against single-coalition outputs on an audit subset.

Report:

```text
prediction_difference_by_task
value_difference = abs(v_single - v_batch)
DCLR_difference
duplicate_set_consistency
```

Enable batch mode only if the batch-vs-single differences are below the convergence tolerances.

---

## 12. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Players | Evidence items from structured output | Already atomic, factual, structured |
| Taxonomy/leakage | `C0-C8` taxonomy + effective source date | Uses the companion leakage spec and keeps leakage binary/auditable |
| Execution order | Leakage-first | If no leaked claims exist, DCLR is exactly zero and Shapley can be skipped |
| v(∅) | Normalized to 0 | All tasks use above-baseline value functions |
| v(full) | Closed-world evaluator with all evidence claims | Keeps one consistent value function for all coalitions |
| Original prediction | Metadata only | Used for reconstruction diagnostics, not Shapley efficiency |
| v(S) prompt | Compact closed-world prediction-only prompt | Forces use of selected facts and minimizes token cost |
| Temperature | 0 for all coalition evaluations | Deterministic output required for regression |
| Output length | Task-specific short JSON | Reduces sample tokens |
| Sampling method | Leverage SHAP, not KernelSHAP | Uses leverage-score row sampling and projected regression |
| Default budget | `C=15`, cap 1200 | Calibrated production setting from pilot convergence sweeps |
| High-rigor budget | `C=25`, cap 2000 on audit subset | Used for convergence validation and paper case studies |
| Coalition batching | Disabled by default | Avoids cross-coalition contamination |
| Shapley evaluator | Same model/checkpoint as generator by default | Minimizes evaluator mismatch and improves attribution faithfulness |

---

## 13. References

- Musco & Witter, "Provably Accurate Shapley Value Estimation via Leverage Score Sampling," ICLR 2025 (Spotlight)
- Covert & Lee, "Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression," AISTATS 2021
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017
- Zhang et al., "Efficient Sampling Approaches to Shapley Value Approximation," SIGMOD 2023
