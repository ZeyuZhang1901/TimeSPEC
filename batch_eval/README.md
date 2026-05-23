# Batch Evaluation (Anthropic API)

Batch-mode leakage evaluation using the Anthropic Message Batches API.

This module mirrors `leakage_eval/` but submits taxonomy and Shapley coalition
prompts as batches to the Anthropic API, then assembles the results into the
same output schema (`instance_results.jsonl`, `summary.json`, etc.).

## Why batch?

The Anthropic Message Batches API gives a **50% discount** on all per-token
rates in exchange for accepting up to 24h of latency. For our offline
leakage evaluation, latency is irrelevant — what matters is cost.

## Layout

- `anthropic_batch.py` — low-level batch API client (submit, poll, fetch).
- `prompts.py`         — reuses `leakage_eval` taxonomy + coalition prompt builders.
- `pipeline.py`        — two-phase orchestration:
                         Phase 1 = taxonomy batch, Phase 2 = Shapley coalitions batch.
- `state.py`           — persistent on-disk state (batch IDs, request manifests) for resume.
- `cli.py`             — CLI entry point.

## Output

Results land in the same `runtime/<model>/eval_results_<judge_tag>/<baseline>/`
directory tree used by the sync pipeline (manifest, instance_results.jsonl,
claim_results.jsonl, shapley_results.jsonl, errors.jsonl, summary.json).
