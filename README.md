# TimeSPEC: Temporal Leakage Evaluation and Mitigation

Code, data, and results for the paper *"All Leaks Count, Some Count More:
Interpretable Temporal Contamination Detection and Mitigation in LLM
Backtesting"* (EMNLP 2026).

The repository implements two contributions:

1. **Shapley-DCLR** — a model-agnostic evaluation pipeline that decomposes an
   LLM's prediction rationale into atomic claims, classifies each claim's
   temporal provenance under a nine-class taxonomy (C0–C8), and computes the
   Shapley-weighted Decision-Critical Leakage Rate via Leverage SHAP.
2. **TimeSPEC** — an inference-time forecasting architecture that interleaves
   temporally-filtered retrieval with claim-level supervision, so that every
   piece of stated evidence in the final rationale is a verified pre-cutoff
   claim.

The paper's experiments compare four generation methods on a
(Search, Supervision) grid: **Temporal Hint** (−/−), **Self-Correction**
(−/+), **Temporal RAG** (+/−), and **TimeSPEC** (+/+), referred to as
B1–B4 throughout the code.

## Repository Layout

```
├── generation/          # B1–B4 generation pipeline
│   ├── methods/         # b1_temporal_hint ... b4_timespec
│   ├── runner.py        # Orchestrator: RAG retrieval, method dispatch, I/O
│   └── perplexity.py    # Date-filtered Perplexity retrieval client
├── leakage_eval/        # Shapley-DCLR evaluation pipeline
│   ├── taxonomy/        # C0–C8 classifier, date verifier, search escalation
│   ├── shapley/         # Leverage SHAP, exact Shapley, coalition sampling
│   ├── metrics/         # OLR, DCLR, task-specific performance
│   └── pipeline.py      # Main entry point
├── batch_eval/          # Anthropic Batch API path (optional)
├── configs/             # YAML experiment configurations
├── scripts/             # All experiment scripts — see scripts/README.md
├── data/                # Benchmark data — see data/README.md
├── camera_ready/        # Camera-ready experiments E1/E3/E5/E6: code inputs,
│                        #   logs, and result files — see camera_ready/README.md
├── rebuttal_validation/ # Rebuttal-phase pilots, judge-robustness runs,
│                        #   provenance audits, human annotation data
├── test_e2e/            # End-to-end tests with sample fixtures
├── usage_log.py         # Opt-in per-call cost logging (CR_USAGE_LOG env var)
├── taxonomy_and_leakage_detection.md  # C0–C8 taxonomy specification
└── shapley_evaluation_pipeline.md     # Shapley / Leverage SHAP design document
```

## Setup

Requires Python ≥ 3.11.

```bash
# The code imports itself as the package `LeakageEvaluationFramework`,
# so clone into a directory with that name and put its parent on PYTHONPATH.
git clone https://github.com/ZeyuZhang1901/TimeSPEC.git LeakageEvaluationFramework
cd LeakageEvaluationFramework
pip install -r requirements.txt
export PYTHONPATH="$(dirname "$PWD")"
```

Create a `.env` file in the repository root:

```
OPENROUTER_API_KEY=sk-or-...   # generation + evaluation (https://openrouter.ai)
PERPLEXITY_API_KEY=pplx-...    # retrieval + search-escalated verification
ANTHROPIC_API_KEY=sk-ant-...   # optional, only for the batch evaluation path
```

## Reproducing the Paper

### Main ablation (§4.1): B1–B4 generation + Shapley-DCLR evaluation

Generate completions for all four methods on the binary benchmark, then
evaluate each method's outputs:

```bash
# Generation (one model; the paper runs three)
python -m LeakageEvaluationFramework.generation.cli run \
    --dataset data/binary.jsonl \
    --output runs/main/<model_tag> \
    --model <openrouter_model_id> \
    --methods b1,b2,b3,b4 \
    --judge-model google/gemini-3.1-flash-lite \
    --concurrency 8

# Evaluation (per method)
python -m LeakageEvaluationFramework.leakage_eval.cli run \
    --config configs/leakage_eval_default.yaml \
    --input runs/main/<model_tag>/<method>/completions.json \
    --dataset data/binary.jsonl \
    --output runs/main/<model_tag>/eval/<method> \
    --model <openrouter_model_id> \
    --method <b1|b2|b3|b4>
```

`scripts/run_cross_eval_all_parallel.sh` and the other `run_*` wrappers in
`scripts/` reproduce the parallel evaluation grid used for the paper's main
table.

### Task-sensitivity probe (§4.2)

Evaluates B1–B4 on three tasks (Legal, NBA Salary, Stock ranking) whose
reliance on post-cutoff information differs:

```bash
bash scripts/run_task_sensitivity_eval.sh
```

The probe data ships in `data/task_sensitivity_full_metadata.json`.

### Camera-ready experiments (E1 / E3 / E5 / E6)

These back the human-validation, frozen-snapshot, metric-comparison, and cost
results added in the camera-ready revision. All result files are shipped in
`camera_ready/`; the commands below re-run them from scratch.

**E1 — Human validation and inter-annotator agreement (Appendix D):**

```bash
python scripts/cr_prepare_annotator2_sheet.py     # blinded sheet, seed 7
python scripts/cr_score_iaa.py                    # kappa, confusion, consistency
```

**E3 — Frozen-snapshot retrieval baseline (§4.3, Appendix F):**

```bash
python scripts/cr_sample_treatment.py --dataset data/binary.jsonl --n 350 --seed 42
python scripts/cr_snapshot_crawl.py               # resumable Wayback crawl
bash   scripts/cr_run_e3_full.sh                  # 3 arms: generation + evaluation
python scripts/cr_analyze_snapshot.py             # bootstrap comparison
```

**E5 — Shapley vs. uniform / Top-K / leave-one-out weighting (Appendix E):**

```bash
python scripts/cr_e5_metric_comparison.py         # offline, from cached coalitions
```

**E6 — Per-stage cost and latency (Appendix C):**

```bash
bash   scripts/cr_run_e6_full.sh                  # instrumented 100-question run
python scripts/cr_e6_cost_table.py \
    --usage-log camera_ready/e6_cost/full_usage_log.jsonl \
    --output camera_ready/e6_cost/cost_table.json --n-questions 100
```

Set `CR_USAGE_LOG=<path>` to record every LLM/search call (component, stage,
tokens, cost, latency) for any run. See `scripts/README.md` for the full
script index, including the rebuttal-phase pilots in `rebuttal_validation/`.

## Shipped Results

| Paper location | Result | Files |
|---|---|---|
| §4.1, Appendix D | Judge–human agreement 97.8%; inter-annotator κ = 0.97 | `camera_ready/e1_human/iaa_report.json`, `rebuttal_validation/human_taxonomy_120/` |
| §4.3, Appendix F | Snapshot-RAG vs. live retrieval vs. TimeSPEC (159 matched questions) | `camera_ready/e3_snapshot/comparison.json`, `eval/`, `gen/` |
| Appendix E | Uniform / Top-K / LOO vs. Shapley-DCLR comparison | `camera_ready/e5_shapley/*.json` |
| Appendix C | $0.051/question per-stage cost and latency table | `camera_ready/e6_cost/cost_table.json`, `full_usage_log.jsonl` |
| Rebuttal pilots | Judge robustness, provenance audits, faithfulness checks | `rebuttal_validation/` |

Camera-ready runs use Qwen 3.5-35B for generation and Gemini 3.1 Flash Lite
as the judge; the main ablation covers three LLMs. A fully instrumented run
costs about $0.051 per question end-to-end (generation, retrieval,
supervision, and the complete Shapley evaluation).

## Design Documents

- `taxonomy_and_leakage_detection.md` — the C0–C8 temporal provenance
  taxonomy, precedence rules, and three-tier verification routing.
- `shapley_evaluation_pipeline.md` — Shapley value computation, Leverage SHAP,
  coalition sampling, and the closed-world value function.

## Citation

```bibtex
@inproceedings{zhang2026timespec,
  title  = {All Leaks Count, Some Count More: Interpretable Temporal
            Contamination Detection and Mitigation in LLM Backtesting},
  author = {Zhang, Zeyu and Stadie, Bradly C.},
  year   = {2026},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026}
}
```

## License

Released for academic research purposes.
