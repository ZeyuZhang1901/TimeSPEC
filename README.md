# Temporal Leakage Evaluation Framework

Code for the paper *"All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection and Mitigation in LLM Backtesting"*.

This repository implements:
1. **Shapley-DCLR evaluation pipeline** — a model-agnostic framework that decomposes LLM prediction rationales into atomic claims, classifies their temporal provenance under a nine-class taxonomy (C0–C8), and computes Shapley-weighted Decision-Critical Leakage Rates via Leverage SHAP.
2. **Four generation methods** occupying the cells of a (Search, Supervision) grid:
   - **Temporal Hint** (−Search, −Supervision)
   - **Self-Correction** (−Search, +Supervision)
   - **Temporal RAG** (+Search, −Supervision)
   - **TimeSPEC** (+Search, +Supervision) — the full inference-time leakage prevention architecture

## Repository Structure

```
├── generation/          # B1–B4 generation pipeline
│   ├── methods/         # b1_temporal_hint, b2_self_correction, b3_rag, b4_timespec
│   ├── prompts.py       # Shared prompt templates
│   ├── runner.py        # Orchestrator: RAG retrieval, method dispatch, I/O
│   └── scripts/         # Shell scripts for running generation
├── leakage_eval/        # Evaluation pipeline
│   ├── taxonomy/        # C0–C8 classifier, date verifier, search escalation
│   ├── shapley/         # Leverage SHAP, exact Shapley, coalition sampling
│   ├── metrics/         # OLR, DCLR, task-specific performance
│   ├── evaluators/      # OpenRouter client, coalition prediction parsers
│   ├── io/              # Data loading, JSONL I/O, result writers
│   ├── audit/           # Convergence, contamination, sensitivity audits
│   └── pipeline.py      # Main evaluation pipeline entry point
├── batch_eval/          # Anthropic Batch API path (optional)
├── configs/             # YAML experiment configurations
├── scripts/             # Orchestration and analysis utilities
├── test_e2e/            # End-to-end tests with sample fixtures
├── taxonomy_and_leakage_detection.md   # C0–C8 taxonomy specification
└── shapley_evaluation_pipeline.md      # Shapley/Leverage SHAP design document
```

## Setup

### Prerequisites

- Python ≥ 3.11
- API keys for:
  - [OpenRouter](https://openrouter.ai) (`OPENROUTER_API_KEY`) — generation + evaluation
  - [Perplexity](https://www.perplexity.ai) (`PERPLEXITY_API_KEY`) — RAG retrieval and search-escalated verification
  - [Anthropic](https://www.anthropic.com) (`ANTHROPIC_API_KEY`) — optional, for batch evaluation path

### Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```
OPENROUTER_API_KEY=sk-or-...
PERPLEXITY_API_KEY=pplx-...
ANTHROPIC_API_KEY=sk-ant-...   # optional
```

### Dataset

The benchmark dataset is not included in this repository. Place the dataset under `dataset/` with the following structure:

```
dataset/
├── final/
│   └── binary.jsonl          # 2,769 binary forecasting questions (Treatment + Control)
└── evaluation/
    └── task_sensitivity_full_metadata.json  # Stock, NBA Salary, Legal task data
```

## Running Experiments

### 1. Generation (B1–B4)

Run all four methods across three models:

```bash
bash generation/scripts/run_all.sh
```

Or run a single model:

```bash
bash generation/scripts/run_one.sh <model_id> <output_dir>
```

### 2. Evaluation (Shapley-DCLR)

Run the leakage evaluation pipeline on generated completions:

```bash
uv run python -m LeakageEvaluationFramework.leakage_eval.cli run \
    --config configs/leakage_eval_default.yaml \
    --input <path/to/completions.json> \
    --dataset <path/to/dataset.jsonl> \
    --output <output_dir> \
    --model <model_id> \
    --method <b1|b2|b3|b4>
```

### 3. Task-Sensitivity Probe

```bash
bash scripts/run_task_sensitivity_eval.sh
```

See `configs/` for all available YAML configurations.

## Key Design Documents

- `taxonomy_and_leakage_detection.md` — Full specification of the nine-class temporal provenance taxonomy (C0–C8), precedence rules, and the three-tier verification routing.
- `shapley_evaluation_pipeline.md` — Detailed design of the Shapley value computation, including Leverage SHAP implementation, coalition sampling, and the closed-world value function.

## License

This code is released for academic research purposes.
