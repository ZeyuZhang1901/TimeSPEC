# Two-Day Rebuttal Validation Results

These are targeted validation results, not replacements for the submitted 2,769-instance experiment.

## Executive Summary

| Validation | Data / scope | Headline result | Reviewer concern addressed | Main caveat |
|---|---|---|---|---|
| Judge/taxonomy robustness | 120 public-source claims; C0–C8 balanced; Gemini ×3, Claude, Kimi | Final leakage agreement 99.2–100%; leakage F1 0.992–1.000 | Automatic-label reliability | Labels are rule/source-derived, not independent human C0–C8 gold |
| Semantic prompt robustness | Two complete C0–C8 prompt variants on 120 claims | Final leakage agreement 100% | Prompt sensitivity | Exact category agreement only 65.8–78.3% |
| Human cutoff-status adjudication | 30 difficult retrievals; 9 disagreements decomposed into 174 atomic claims | 26/27 scorable passages correct: 96.3% accuracy, 90.0% precision, 100% recall, 94.7% F1 | Human-vs-verifier cutoff status | Targeted challenge subset, not population accuracy |
| Qwen four-method pilot | 12 public [Autocast](https://github.com/andyzoujm/autocast) questions; four methods | TimeSPEC reduced DCLR 86.4% vs. live RAG, with performance 0.675 vs. 0.670 | Method effectiveness; faithfulness; Shapley value | Small pilot |
| Shapley isolation | Same Qwen pilot plus exact/LOO coalitions | Same OLR 0.400 produced DCLR 0.661 vs. 0.291; live RAG DCLR 0.138 vs. LOO 0.083 | Value beyond unweighted/LOO leakage | Small sample |
| Label-error sensitivity | 200 simulations at 5%, 10%, 20% cross-tier noise | TimeSPEC remained below live RAG in 99.5%, 99.5%, and 86.5% | Taxonomy-error robustness | Synthetic perturbations |
| Snapshot-RAG comparison | 8 matched questions with usable [Wayback](https://web.archive.org/) snapshots | Snapshot reduced live-RAG DCLR 62.6%; TimeSPEC DCLR 0.000 and performance 0.853 vs. 0.822 | External frozen-retrieval baseline | 66.7% snapshot coverage |
| Runtime/cost | 12-question Qwen pilot | TimeSPEC: 116.9 s evaluation, 740 coalitions/728 calls, approx. \$0.034 Qwen coalition cost | Practical overhead | Judge/search monetary charges excluded |

**Overall assessment:** The results strongly support output-level leakage-label reliability, Shapley’s added diagnostic value, and TimeSPEC’s inference-time mitigation. They provide supportive—not definitive—evidence for exact C0–C8 categorization and historical-web provenance. They do not establish latent-reasoning faithfulness or population-level accuracy.

## 0. Implementation Alignment and Prompt A/B

| Stage | Pilot implementation | Role |
|---|---|---|
| Forecast generation | Qwen3.5-35B | Emits structured `id`–`fact`–`source_date` evidence |
| Generation supervision | Gemini 3.1 Flash Lite | Applies C0–C8 rules and deterministic leakage checks |
| Retrieval/search escalation | Perplexity Sonar | Initial date-filtered retrieval and generation-time re-verification |
| Post-hoc taxonomy/date judge | Gemini 3.1 Flash Lite | Rejudges final structured evidence |
| Shapley evaluator | Qwen3.5-35B | Predicts from controlled claim subsets |

The inspected post-hoc evaluator does not invoke search escalation; search escalation is generation-time TimeSPEC behavior. The operational date should be described as estimated earliest verified date \(\hat\tau(c)\), and closed-world claims should be limited to the explicit evidence context.

The released judge prompt referred to a taxonomy document without embedding C0–C8 definitions. On the same clear challenge set:

| Prompt condition | C0–C8 accuracy | Operational-tier accuracy | Leakage F1 |
|---|---:|---:|---:|
| Definitions not embedded | 22.2% | 43.3% | 1.000 |
| Appendix-B definitions embedded | 75.6% | 87.8% | 1.000 |

Exact categories are prompt-sensitive, while final leakage remained stable on clear cases. Any reported validation must release the exact embedded prompt.

## 1. Taxonomy, Date, and Judge Robustness

### Setup

- 120 clear public-source claims: 90 balanced C0--C8 cases and 30 authoritative date-boundary cases.
- Gemini 3.1 Flash Lite: three temperature-0 runs.
- Claude Sonnet 4 and Kimi K2.5: one run each.
- Full Appendix-B taxonomy and deterministic post-checks supplied to every judge.

### Results

- Gemini repeated-run agreement:
  - exact category: 92.5%;
  - final leakage label: 100%;
  - effective date: 83.3%.
- Exact C0--C8 accuracy:
  - Gemini: 75.6--82.2% across three runs;
  - Claude: 98.9%;
  - Kimi: 96.7%.
- Cross-judge agreement versus Gemini:
  - Claude: 60.8% exact category, 90.8% operational tier, 99.2% leakage;
  - Kimi: 63.3% exact category, 89.2% operational tier, 100% leakage.
- Clear-set leakage F1:
  - Gemini: 1.000 in all three runs;
  - Claude: 0.992;
  - Kimi: 1.000.
- Date-boundary mean absolute error:
  - Gemini/Claude: 0 days;
  - Kimi: 0.03 days.

### Interpretation

Exact fine-grained categories vary more across judges than the quantities that enter DCLR. Operational-tier agreement remains about 89--91%, and final leakage labels remain 99.2--100% consistent across judges. This supports robustness of the final leakage decision on clear cases while motivating explicit uncertainty for ambiguous open-web claims.

### Scope

The challenge labels are rule/source-derived, not independent human C0--C8 annotations. The result validates clear taxonomy/date cases and judge stability; it does not establish accuracy on every ambiguous claim.

### Representative Boundary Errors

- `c2_05`: all judge families often classified a federal-funds percentage as C3 rather than gold C2; this adjacent boundary did not change its effective date or leakage.
- `c1_08`: Kimi classified the Supreme Court's nine-justice attribute as time-dependent C2 rather than stable C1.
- One Gemini run classified several post-cutoff comparisons as C7 rather than C4; exact category changed, but leakage remained correct.

## 2. Compact Four-Method Qwen Pilot

### Setup

- 12 public Autocast Treatment questions across finance, politics, and science.
- Qwen3.5-35B.
- Temporal Hint, Self-Correction, Temporal RAG, and TimeSPEC.
- Same structured evidence protocol, one-regeneration cap, Gemini judge, and Perplexity retrieval used by the released implementation.
- Shapley computed for every instance, including zero-leakage outputs, to enable reconstruction-faithfulness analysis.

### Main Results

| Method | Performance | OLR | Shapley-DCLR | Mean reconstruction gap | High/medium faithfulness | LOO-weighted leakage |
|---|---:|---:|---:|---:|---:|---:|
| Temporal Hint | 0.738 | 0.219 | 0.210 | 0.165 | 75.0% | 0.250 |
| Self-Correction | 0.743 | 0.000 | 0.000 | 0.213 | 58.3% | 0.000 |
| Temporal RAG | 0.670 | 0.103 | 0.138 | 0.118 | 83.3% | 0.083 |
| TimeSPEC | 0.675 | 0.014 | 0.019 | 0.069 | 83.3% | 0.000 |

### Key Findings

- TimeSPEC reduced DCLR by 86.4% relative to Temporal RAG (0.138 to 0.019) while slightly improving performance (0.670 to 0.675).
- For Temporal RAG, DCLR was 34.6% higher than OLR (0.138 versus 0.103), showing that its leaked claims carried above-average decision weight.
- TimeSPEC's mean reconstruction gap was 0.069, with 83.3% of outputs classified as high/medium faithfulness.
- Shapley and leave-one-out were materially different for Temporal RAG (0.138 versus 0.083), consistent with Shapley's treatment of redundant/interacting evidence.

### Claim-Level Contrast

- Temporal RAG on `autocast_G6`: OLR was 0.400 but DCLR was 0.661. A leaked post-cutoff claim about China's representation at the November 2015 trilateral summit carried Shapley value 0.776.
- Temporal Hint on `autocast_G1477`: OLR was also 0.400, but DCLR was only 0.291. The two leaked claims were less dominant than the valid claims driving the prediction.

The equal OLR but sharply different DCLR illustrates the information lost by unweighted claim counting.

## 3. Robustness to Label Errors and Temporal Uncertainty

### Random Cross-Tier Label Perturbation

- At 5% global label flips:
  - TimeSPEC remained lower-DCLR than Temporal RAG in 99.5% of 200 simulations;
  - Self-Correction remained lower-DCLR than Temporal Hint in 100%.
- At 10% flips:
  - both supervision comparisons were preserved in 99.5% of simulations.
- At 20% flips:
  - TimeSPEC remained below Temporal RAG in 86.5%;
  - Self-Correction remained below Temporal Hint in 98.0%.

### DCLR Bounds for Flagged/Low-Confidence Claims

- Temporal Hint: 0.190--0.222.
- Self-Correction: 0.000--0.016.
- Temporal RAG: 0.138--0.184.
- TimeSPEC: 0.019--0.066.

The supervised methods remain below their unsupervised counterparts across these uncertainty bounds.

## 4. Runtime and Resource Profile

### Observed Runtime

- Shared four-method generation/retrieval: 184.9 seconds for 12 questions.
- Post-hoc evaluation:
  - Temporal Hint: 64.7 seconds;
  - Self-Correction: 52.2 seconds;
  - Temporal RAG: 129.1 seconds;
  - TimeSPEC: 116.9 seconds.

### Evaluation Calls

- Temporal Hint: 294 coalition-model calls for 48 claims.
- Self-Correction: 188 calls for 45 claims.
- Temporal RAG: 708 calls for 64 claims.
- TimeSPEC: 728 calls for 63 claims.

### TimeSPEC Generation Behavior

- 25% of outputs triggered one regeneration.
- Mean forecasting-model calls: 1.50 per instance.
- Mean supervisor/resupervisor calls before search re-verification: 1.25 per instance.

### Stage-Level Accounting

| Stage | Calls/requests | Tokens | Note |
|---|---:|---:|---|
| Shared no-RAG + RAG initial generation | 24 Qwen calls | Not logged | Included in 184.9 s shared generation |
| Retrieval query generation | 12 judge calls | Not logged | Shared by RAG and TimeSPEC |
| Initial Perplexity retrieval | 30 requests | Provider-specific | 2.5 requests/question |
| TimeSPEC supervisor/resupervisor | Mean 1.25 calls/question | Not logged | Before search re-verification |
| TimeSPEC search escalation | 14 search + 14 re-verification calls | Provider-specific | Observed across 12 questions |
| Post-hoc TimeSPEC taxonomy | 12 judge batches | 20,077 tokens | Included in 116.9 s evaluation |
| Post-hoc TimeSPEC Shapley | 740 coalitions; 728 Qwen calls | Approx. 170,311 input + 4,749 output | Approx. \$0.034 Qwen cost |

Generation token usage and Perplexity charges were not emitted by the released logger and should not be presented as exact.

### Approximate Qwen Coalition Cost

Using repository reference prices of \$0.1625/M input and \$1.30/M output:

- Temporal Hint: approximately \$0.012 for all 12 evaluations.
- Self-Correction: approximately \$0.008.
- Temporal RAG: approximately \$0.032.
- TimeSPEC: approximately \$0.034.

These are estimated Qwen coalition costs only. Taxonomy-judge and Perplexity charges should be reported separately because provider pricing and search-request charges vary.

## 5. Implementation Finding

The released judge prompt originally referred to an external C0--C8 decision tree without embedding the definitions. On the clear challenge set this yielded poor category accuracy despite correct leakage/date labels. Embedding the Appendix-B taxonomy restored coherent classification. The rebuttal and revision should describe and release the exact prompt used for any reported validation.

## 6. Operating-Point Comparison

| Approach | Strongest guarantee | Main limitation relative to TimeSPEC |
|---|---|---|
| [Frozen-snapshot retrieval](https://arxiv.org/abs/2602.00758) | Retrieved documents existed in the snapshot | Does not remove parametric leakage or provide claim-level attribution |
| [ChronoGPT](https://arxiv.org/abs/2502.21206) | Stronger parameter-level cutoff | Requires retraining/checkpoints; does not inherently secure retrieval |
| [ForecastBench](https://openreview.net/forum?id=lfPkGWXLLf) | Outcome unavailable at forecast time | Does not audit or mitigate arbitrary historical backtests |
| TimeSPEC | Only validated claims enter explicit final evidence | Does not remove latent parametric knowledge or guarantee snapshot provenance |

ForecastBench already supplies the submitted 827-question Control group. A targeted Snapshot-RAG comparison was subsequently run on eight matched questions; results are in Section 8.

## 7. Low-Faithfulness Example

Temporal RAG on `autocast_G1477` produced a reconstruction gap of 1.0: its prediction was not reproducible from its evidence list. The pipeline correctly flags such cases as low faithfulness, supporting conditional DCLR reporting rather than treating every rationale as equally trustworthy.

## 8. Snapshot-RAG Pilot

The Wayback feasibility gate found usable pre-cutoff snapshots for 8 of the 12 Autocast pilot questions (66.7% coverage). On those same eight questions:

| Method | Performance | OLR | DCLR | Reconstruction gap | High/medium faithfulness |
|---|---:|---:|---:|---:|---:|
| Live Temporal RAG | 0.880 | 0.113 | 0.145 | 0.040 | 87.5% |
| Snapshot RAG | 0.822 | 0.056 | 0.054 | 0.253 | 50.0% |
| TimeSPEC | 0.853 | 0.000 | 0.000 | 0.091 | 75.0% |

- Snapshot RAG reduced DCLR by **62.6%** relative to live date-filtered RAG, confirming the value of frozen retrieval provenance.
- TimeSPEC achieved **zero measured DCLR** on the matched subset while outperforming Snapshot RAG in performance (0.853 vs. 0.822).
- Snapshot coverage was incomplete and its reconstruction faithfulness was lower, so this remains a small operating-point comparison rather than a universal baseline ranking.

## 9. Semantic Prompt Robustness

Two semantically equivalent complete taxonomy prompts were compared with the current embedded Appendix-B prompt on the same 120 claims:

| Prompt variant | Exact category agreement | Tier agreement | Date agreement | Leakage agreement | Gold leakage accuracy |
|---|---:|---:|---:|---:|---:|
| Rules-first | 78.3% | 87.5% | 91.7% | 100% | 100% |
| Tier-first | 65.8% | 83.3% | 90.8% | 100% | 100% |

Exact fine-grained categories remain prompt-sensitive, but final leakage decisions were invariant across both complete prompt formulations. This supports the operational leakage metric while reinforcing the need to release the exact taxonomy prompt and report tier/category uncertainty separately.

## 10. Human-Adjudicated Cutoff-Status Validation

### Design

1. Collect 100 actual Perplexity retrieval outputs from 60 public Autocast questions spanning finance, politics, and science.
2. Human-review a stratified subset, including all high-confidence automatic A/B cases and a random sample of initially unverifiable cases.
3. Obtain an independent professor review on 30 difficult overlap items.
4. The initial passage-level labels disagreed because long Perplexity paragraphs mixed pre- and post-cutoff assertions. To align the unit with the paper, decompose the nine disputed paragraphs into 174 atomic claims.
5. Human-adjudicate all 43 atomic claims the verifier flagged as post-cutoff.

Labels:

- A: publicly knowable before cutoff;
- B: knowable only after cutoff;
- C: insufficient historical evidence;
- F: factually false/unsupported as stated.

### Results

| Measure | Result |
|---|---:|
| Temporally scorable passage-level cases | 27 |
| Accuracy | 96.3% (26/27) |
| Leakage precision | 90.0% |
| Leakage recall | 100% |
| Leakage F1 | 94.7% |
| False positives / false negatives | 1 / 0 |
| Original disputed paragraphs containing at least one confirmed leaked atomic claim | 8/9 |

### Atomic Disagreement Analysis

| Human label among 43 verifier-flagged atomic claims | Count |
|---|---:|
| Confirmed post-cutoff (B) | 16 |
| Pre-cutoff (A) | 6 |
| Factually false/unsupported (F) | 20 |
| Unresolved (C) | 1 |

The high F count is **not** a population factual-error rate: these 43 claims were selected from the most disputed, verifier-flagged paragraphs. It demonstrates that temporal verification and factual verification are separate tasks.

### Interpretation and Scope

Atomic decomposition resolved eight of nine original passage-level disagreements because valid pre-cutoff and leaked post-cutoff subclaims could be labeled separately. The 96.3% result is a targeted human-adjudicated challenge-set result, not a general accuracy estimate. It validates cutoff status on difficult cases; it does not establish exact earliest-date recovery, strict cited-page version compliance, or latent internal-reasoning faithfulness.

## 11. Exploratory 100-Retrieval Provenance Audit

A single-annotator stratified audit reviewed all 26 high-confidence automatic A/B items and a random 20-of-74 sample of initially unverifiable items. The weighted exploratory estimate was:

| Claim-level public availability | Weighted estimate |
|---|---:|
| Verified pre-cutoff | 70.6% |
| Post-cutoff leakage | 22.0% |
| Unverifiable | 7.4% |

These estimates concern whether the claim was publicly knowable somewhere before cutoff. They do not prove that the exact cited webpage version contained the claim then. Because this is a single-annotator stratified estimate, it should be treated as exploratory rather than a headline rebuttal result.
