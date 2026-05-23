# benchmark — Comprehensive Experiment Design

**Paper:** All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection in LLM Backtesting
**Deadline:** May 25, 2026 (ARR submission)
**Last updated:** 2026-05-04

---

## 1. Dataset Design

### 1.1 Design Principle: Treatment vs. Control

| Group | Definition | Purpose |
|---|---|---|
| **Treatment** | Questions whose ground-truth resolution date is BEFORE all models' knowledge cutoffs | Leakage IS possible — models may have seen the answer in training |
| **Control** | Questions whose ground-truth resolution date is AFTER all models' knowledge cutoffs | Leakage should NOT be possible — validates the metric doesn't produce false positives |

**Temporal boundary:**
- Treatment: resolution date ≤ **March 2024** (before GPT-OSS cutoff, the earliest model we use)
- Control: resolution date ≥ **July 2025** (after Nemotron cutoff, the latest model we use)

### 1.2 Unified Data Schema

Every instance across all sources shares this format:

```json
{
  "id": "autocast_G1855",
  "source": "autocast | forecastbench | sports_2025 | finance_2025 | politics_2025",
  "question": "Will X happen before Y date?",
  "background": "Context available before temporal_cutoff...",
  "temporal_cutoff": "2020-03-15",
  "answer": "yes",
  "answer_type": "binary | mc | numerical",
  "resolution_date": "2020-06-01",
  "group": "treatment | control",
  "domain": "politics | finance | science | sports | technology | legal",
  "choices": ["yes", "no"]
}
```

### 1.3 Data Sources

#### Treatment Group (~1,500 instances)

| Source | Domain | N | Notes |
|---|---|---|---|
| **Autocast** (resolved, closed ≤ Mar 2024) | Politics, Finance, Science, Technology, COVID | ~1,000 | Stratified sample from 3,748 resolved. Mix of T/F (500), numerical (300), MC (200). Prioritize high-leakage-potential topics: COVID, elections, geopolitics, AI predictions. |
| **Original legal task** (expanded) | Legal | ~200 | Expand from 98 → 200 using additional Supreme Court / appellate cases pre-2024. Keep format identical. |
| **Original NBA salary task** (expanded) | Sports | ~200 | Expand from 152 → 200 using 2018–2023 free agents. |
| **Original stock ranking** (multiple windows) | Finance | ~100 | Keep COVID window (100) as one sub-condition for continuity with KDD submission. |
| **Total Treatment** | | **~1,500** | |

#### Control Group (~500 instances)

| Source | Domain | N | How to Construct |
|---|---|---|---|
| **Sports results (2025-2026 season)** | Sports | ~150 | NBA 2025-26 season game outcomes, NFL 2025 playoff results, soccer Champions League 2025-26. Format: "Will [Team A] beat [Team B] on [date]?" + pre-game stats as background. All verifiable, unambiguous. |
| **Financial indicators (Q3 2025 – Q1 2026)** | Finance | ~150 | Quarterly GDP, CPI, unemployment rate, S&P 500 closing values on specific dates, corporate earnings (FAANG). Format: "What will be the [indicator] for [period]?" + prior quarter data as background. |
| **Political/world events (Jul 2025 – Apr 2026)** | Politics | ~100 | Election results, policy decisions, international agreements from this period. Format mirrors Autocast T/F questions. |
| **Science/technology milestones (2025-2026)** | Science/Tech | ~100 | AI model releases, space missions, scientific discoveries announced after Jul 2025. |
| **Total Control** | | **~500** | |

#### Summary

| | Treatment | Control | Total |
|---|---|---|---|
| **Instances** | ~1,500 | ~500 | **~2,000** |
| **Domains** | 6 | 4 | 6 |
| **Answer types** | T/F, MC, Numerical | T/F, MC, Numerical | All |
| **Scale vs. KDD** | 4.3× | New | **5.7×** |

### 1.4 Metric Computation Strategy

Running full Shapley (100 MC samples) on 2,000 instances × 4 models × 5 methods = 40,000 runs is prohibitively expensive. Strategy:

| Metric | Computed on | Instances | Purpose |
|---|---|---|---|
| **OLR** (Overall Leakage Rate) | Full dataset | All 2,000 × 4 models × 5 methods | Headline leakage numbers; cheap (no Shapley) |
| **Shapley-DCLR** | Representative subset | 500 stratified instances × 4 models × 5 methods | Deep analysis; expensive but core metric |
| **Task performance** | Full dataset | All 2,000 × 4 models × 5 methods | Accuracy/correlation numbers |

This gives both breadth (2,000 instances for OLR) and depth (500 for Shapley-DCLR).

---

## 2. Models

### 2.1 Selection Criteria

- Must be Instruction/Hybrid/Reasoning type (capable of structured evidence + rationale output)
- At least Medium size (need reliable complex reasoning)
- Well-separated knowledge cutoffs for the Treatment/Control design
- Available on Tinker platform

### 2.2 Selected Models

| # | Model | Tinker ID | Type | Size | Knowledge Cutoff | Sample Cost | Role |
|---|---|---|---|---|---|---|---|
| M1 | **GPT-OSS-120B** | openai/gpt-oss-120b | Reasoning | Medium (MoE) | **Apr 2024** | $0.44/M | Earliest cutoff; all Treatment questions are post-training |
| M2 | **Qwen3.5-35B-A3B** | Qwen/Qwen3.5-35B-A3B | Hybrid+Vision | Medium (MoE) | **~Oct 2024** | $0.89/M | Mid-range cutoff; cost-effective workhorse |
| M3 | **Nemotron-3-Super-120B** | nvidia/Nemotron-3-Super-120B-A12B-BF16 | Hybrid | Large (MoE) | **Jun 2025** | $0.96/M | Latest cutoff; defines Control boundary; cheap |
| M4 | **Kimi-K2.6** | moonshotai/Kimi-K2.6 | Reasoning+Vision | Large (MoE) | **~Apr 2025** | $3.66/M | Strong reasoning; validates Shapley on reasoning-heavy model |

### 2.3 Why These Four

1. **Three different organizations** (OpenAI, Alibaba/Qwen, NVIDIA, Moonshot) → no single-vendor bias
2. **Cutoff spread**: Apr 2024 → Oct 2024 → Apr 2025 → Jun 2025
3. **All MoE** → cost-effective at scale (important for 2,000 × 5 methods)
4. **Mix of Reasoning + Hybrid** → tests whether reasoning-type models (which produce longer CoT) exhibit different leakage patterns than hybrid models
5. **All capable of structured output** with evidence lists and cited rationales

### 2.4 Cross-Model Evaluation Pipeline

To address circularity (reviewer concern W2), the **evaluation pipeline** (claim extraction, categorization, leakage detection) uses a **fixed evaluator model** different from the prediction backbone:

| Prediction model | Evaluator model |
|---|---|
| GPT-OSS-120B | Qwen3.5-35B-A3B |
| Qwen3.5-35B-A3B | GPT-OSS-120B |
| Nemotron-3-Super-120B | Qwen3.5-35B-A3B |
| Kimi-K2.6 | GPT-OSS-120B |

Additionally, run one model (Qwen3.5) with **self-evaluation** vs. **cross-evaluation** and report agreement rate → quantifies circularity bias directly.

---

## 3. Methods / Baselines

### 3.1 Baseline Hierarchy

The baselines form a progressive ladder from "no mitigation" to "full pipeline," isolating what each component contributes:

| # | Method | Temporal Instruction | Search Access | Claim Extraction | Verification | Regeneration | What It Tests |
|---|---|---|---|---|---|---|---|
| B0 | **Unconstrained** | None | No | No | No | No | Raw leakage level (upper bound) |
| B1 | **Temporal Hint** | Prompt: "Only use info before [date]" | No | No | No | No | Can prompting alone reduce leakage? |
| B2 | **Multi-turn Self-Correction** | Prompt + "Review and remove post-cutoff info, regenerate" (2 rounds) | No | No | No | Yes (implicit) | Does iterative self-reflection help without external grounding? |
| B3 | **Temporal RAG** | System prompt with cutoff | Yes (date-filtered) | No | No | No | Does search grounding alone reduce leakage? (Isolates search value) |
| B4 | **TimeSPEC** | Full pipeline | Yes (date-filtered) | Yes | Yes | Yes | Full system (claim extraction + verification + regeneration) |

### 3.2 What Each Comparison Demonstrates

| Comparison | Research Question | Expected Finding |
|---|---|---|
| B0 vs. B1 | Does a simple prompt help? | Partial reduction, but far from elimination |
| B1 vs. B2 | Does self-reflection help beyond prompting? | Marginal improvement; model can't verify its own knowledge |
| B1 vs. B3 | Does grounded search reduce leakage? | Mixed — may reduce some leakage but can also *introduce* leakage via retrieved docs (addresses legal anomaly) |
| B3 vs. B4 | Does claim-level verification add value over raw RAG? | Yes — this is the core contribution. RAG alone may amplify leakage; verification filters it. |
| B0 vs. B4 | End-to-end leakage reduction | Headline number (75–99% reduction claim) |

### 3.3 Baseline Implementation Details

**B0: Unconstrained (Superforecasting)**
- Prompt: "You are an expert forecaster. Analyze the following question and provide your prediction with detailed reasoning."
- No temporal restriction mentioned.

**B1: Temporal Hint**
- Prompt: Same as B0 + "Important: You must only use information that was publicly available before [temporal_cutoff]. Do not use any knowledge of events after this date."
- Single-pass generation.

**B2: Multi-turn Self-Correction**
- Round 1: Same as B1.
- Round 2: "Review your previous answer. Identify any claims that rely on information from after [temporal_cutoff]. Remove them and regenerate your prediction using only pre-cutoff information."
- Round 3 (optional): Repeat Round 2.
- Final answer is from last round.

**B3: Temporal RAG**
- System prompt with temporal cutoff.
- The model is given access to date-filtered search (same Perplexity API / search infrastructure as TimeSPEC).
- Prompt: "Use the provided search results (filtered to before [date]) to ground your reasoning. Provide your prediction with evidence cited from the search results."
- No claim extraction, no verification of individual claims, no regeneration.
- This is the critical ablation: same search access as TimeSPEC, but without the claim-level verification architecture.

**B4: TimeSPEC (Full)**
- Five-phase pipeline: Generator → Supervisor (claim extraction + categorization + search verification) → Regenerator → Re-supervisor → Aggregator.
- Same search access as B3, but adds structured claim-level verification and regeneration.

### 3.4 Optional Additional Baselines (if time permits)

| Method | Description | Purpose |
|---|---|---|
| **B2.5: Structured Self-Correction** | Like B2 but explicitly asks model to list claims and mark temporal status before regenerating (no external search) | Tests whether structured reflection without verification helps |
| **B3.5: RAG + Post-hoc Filtering** | Like B3 but after generation, filter out sentences containing post-cutoff info (simple heuristic, not Shapley-weighted) | Tests whether simple post-hoc filtering matches claim-level verification |
| **B4-ablation: TimeSPEC-NoRegen** | Supervisor detects leakage but does not regenerate | Isolates detection value from mitigation value |

---

## 4. Experiment Matrix

### 4.1 Main Results Table

The core results table in the paper:

| | | **GPT-OSS-120B** | **Qwen3.5-35B-A3B** | **Nemotron-3-Super** | **Kimi-K2.6** |
|---|---|---|---|---|---|
| **Treatment** | B0: Unconstrained | OLR / DCLR / Perf | ... | ... | ... |
| | B1: Temporal Hint | OLR / DCLR / Perf | ... | ... | ... |
| | B2: Self-Correction | OLR / DCLR / Perf | ... | ... | ... |
| | B3: Temporal RAG | OLR / DCLR / Perf | ... | ... | ... |
| | B4: TimeSPEC | OLR / DCLR / Perf | ... | ... | ... |
| **Control** | B0: Unconstrained | OLR / DCLR / Perf | ... | ... | ... |
| | B4: TimeSPEC | OLR / DCLR / Perf | ... | ... | ... |

### 4.2 Key Hypotheses to Test

| # | Hypothesis | How to Test | Expected |
|---|---|---|---|
| H1 | Shapley-DCLR is near-zero in Control for all models | Compare Treatment vs. Control DCLR | Control DCLR ≈ 0; Treatment DCLR >> 0 |
| H2 | TimeSPEC reduces DCLR significantly vs. all baselines | Paired bootstrap test, B4 vs. B0/B1/B2/B3 | p < 0.01 for B4 vs. each |
| H3 | RAG alone does NOT reliably reduce leakage | Compare B3 DCLR vs. B1 DCLR | B3 may be similar or worse than B1 on some domains |
| H4 | Claim verification (B4) adds value over RAG (B3) | Compare B4 vs. B3 DCLR | B4 significantly lower |
| H5 | Framework produces consistent measurements across models | Compare OLR/DCLR rankings across M1-M4 | Same relative ordering of methods |
| H6 | Leakage correlates with temporal proximity to cutoff | Within Treatment, bin questions by distance from model cutoff | Closer to cutoff → higher leakage |

### 4.3 Statistical Analysis

For every reported number:
- **Bootstrap 95% CI** (10,000 resamples over instances)
- **Paired bootstrap significance test** for all pairwise method comparisons
- **Effect size** (Cohen's d)
- Report as: `metric [95% CI]` or `metric ± SE`

### 4.4 Additional Analyses

| Analysis | Section | Purpose |
|---|---|---|
| **Treatment vs. Control validation** | Main results | Validates metric (H1) |
| **Leakage by domain** | Breakdown table | Shows which domains are most leakage-prone |
| **Leakage by question type** | Breakdown | T/F vs. numerical vs. MC differences |
| **Temporal proximity analysis** | Figure | Leakage as function of (resolution_date − model_cutoff) |
| **Cross-evaluator agreement** | Appendix | Quantifies circularity bias |
| **Ablation study** | Subsection | TimeSPEC component contributions |
| **Cost analysis** | Table | Per-instance cost for each method |
| **Shapley convergence** | Appendix | MC samples vs. estimate stability |

---

## 5. Ablation Study Design

Using one model (Qwen3.5-35B-A3B) on the 500-instance Shapley subset:

| Configuration | Search | Claim Extract | Verify | Regen | Re-verify | Aggregate |
|---|---|---|---|---|---|---|
| Full TimeSPEC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| No Re-verification | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| No Regeneration | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| No Verification (extract only) | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| No Claim Extraction (RAG only) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| No Search (self-correction) | ✗ | ✗ | ✗ | ✓ (self) | ✗ | ✗ |

---

## 6. Cost Estimate

### 6.1 Per-Instance Token Estimates

| Method | Input tokens | Output tokens | Total tokens/instance |
|---|---|---|---|
| B0: Unconstrained | ~2,000 | ~1,500 | ~3,500 |
| B1: Temporal Hint | ~2,200 | ~1,500 | ~3,700 |
| B2: Self-Correction (2 rounds) | ~6,000 | ~3,000 | ~9,000 |
| B3: Temporal RAG | ~8,000 | ~2,000 | ~10,000 |
| B4: TimeSPEC (full pipeline) | ~20,000 | ~5,000 | ~25,000 |
| Shapley evaluation (100 MC) | ~200,000 | ~50,000 | ~250,000 |

### 6.2 Total Cost Estimate (using cheapest model Qwen3.5 at $0.89/M sample)

| Component | Instances × Models × Methods | Tokens | Est. Cost |
|---|---|---|---|
| Full dataset (OLR + perf) | 2,000 × 4 × 5 = 40,000 runs | ~400M tokens | ~$360 |
| Shapley-DCLR subset | 500 × 4 × 5 = 10,000 runs × 100 MC | ~2.5B tokens | ~$2,200 |
| Ablation (1 model) | 500 × 1 × 6 configs = 3,000 runs | ~75M tokens | ~$67 |
| Cross-evaluator | 500 × 4 (cross-eval pairs) | ~200M tokens | ~$180 |
| **Total estimate** | | | **~$2,800** |

*Note: This assumes average cost across models. Kimi-K2.6 ($3.66/M) will increase the total. Realistic budget: $3,000–5,000.*

---

## 7. Expected Results Narrative

The experiments should demonstrate:

1. **Validation (H1):** Shapley-DCLR ≈ 0 in Control, >> 0 in Treatment → the metric measures real leakage, not noise.
2. **Scale:** Consistent findings across 2,000 instances and 6 domains → generalizable, not cherry-picked.
3. **Progressive baseline ladder:** B0 > B1 > B2 ≈ B3 > B4 in leakage → each component adds value, with claim verification (B3→B4) being the critical step.
4. **Cross-model consistency:** All 4 models show the same pattern → framework is model-agnostic.
5. **The RAG insight:** B3 (Temporal RAG) may not reduce leakage reliably — it can even introduce new leakage via retrieved documents. This explains the legal anomaly from the KDD submission and motivates why claim-level verification is necessary.
6. **Performance-leakage tradeoff:** Visualize as 2D scatter (Figure 3 style) across all models and methods.

---

## 8. Differences from KDD Submission

| Aspect | KDD (rejected) | benchmark (planned) |
|---|---|---|
| Scale | 350 instances, 3 tasks | 2,000 instances, 6+ domains |
| Models | 1 (Claude Sonnet 4) | 4 (GPT-OSS, Qwen3.5, Nemotron, Kimi-K2.6) |
| Baselines | 2 (Unconstrained, Temporal Hint) | 5 (+ Self-Correction, Temporal RAG, TimeSPEC) |
| Control experiment | None | 500 instances post-cutoff |
| Statistical rigor | None | Bootstrap CIs, significance tests, effect sizes |
| Cross-model evaluation | None | Different predictor/evaluator pairs |
| Ablation | None | 6-configuration component analysis |
| Cost reporting | None | Full cost table |
| Data sources | Own collection only | Public benchmarks (Autocast) + own collection |
| Reproducibility | API-dependent | Public datasets, unified schema, code release |

---

## 9. Timeline

| Week | Dates | Deliverables |
|---|---|---|
| **1** | May 5–11 | Data: Curate Autocast subset (1,000 Treatment); begin Control collection (sports/finance events); implement B2, B3 baselines; set up Tinker pipeline for 4 models |
| **2** | May 12–18 | Run: Full OLR experiment (2,000 × 4 × 5); run Shapley on 500-instance subset; ablation study; cross-evaluator experiment |
| **3** | May 19–25 | Analyze + Write: Bootstrap CIs; all figures/tables; rewrite experiments section; Treatment vs. Control analysis; submit |

---

## 10. Open Decisions

- [ ] Final Autocast sampling strategy (stratified by domain/type/year — exact proportions)
- [ ] Confirm GPT-OSS-120B and Nemotron cutoff dates with empirical probing before committing
- [ ] Control group: determine exact sports/finance/politics sources and question count per domain
- [ ] Shapley subset selection: random 500, or stratified to cover all domains equally?
- [ ] Whether to keep all 3 original tasks (legal, salary, stock) as a "continuity" subset or replace entirely
- [ ] Maximum budget allocation across models (Kimi-K2.6 is 4× more expensive)
