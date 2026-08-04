# Scripts Index

Grouped by the experiment they back. All Python scripts run with the package
root on `PYTHONPATH` (see the repository README for environment setup). Paths
inside scripts are relative to the package root unless the script takes
explicit `--dataset` / `--results-root` / `--output` arguments.

## Main-paper ablation (cross-model B1-B4 generation + evaluation)

| Script | Purpose |
|---|---|
| `run_four_baselines_eval_parallel.sh` | Evaluate B1-B4 outputs for one model in parallel |
| `run_cross_eval_all_parallel.sh` | Cross-model evaluation grid used for the main ablation table |
| `run_claude_self_eval_batch.sh` / `run_claude_self_eval_parallel.sh` | Claude self-evaluation arms (Anthropic Batch API path) |
| `run_kimi_self_eval_parallel.sh` | Kimi self-evaluation arm |
| `reeval_b4_stock_rescue.sh` | Targeted re-evaluation of B4 stock instances after a provider outage |
| `retry_failed_eval.sh` | Re-run evaluation for instances that failed mid-run |
| `dedupe_batch_results.py` | De-duplicate Anthropic batch result files |

## Task-sensitivity probe (337 instances, legal/salary/stock)

| Script | Purpose |
|---|---|
| `run_task_sensitivity_eval.sh` | Full probe evaluation (results in `results/probe_task_sensitivity/` at repo top level) |
| `pilot_shapley_exact_threshold8.py` | Validates the exact-enumeration threshold (n <= 8) for Shapley |
| `compare_paper_vs_exact_shapley.py` | Leverage SHAP vs. exact Shapley agreement check |

## Rebuttal validation (12/20-question pilot, judge robustness, provenance, human audits)

| Script | Purpose |
|---|---|
| `rebuttal_prepare_autocast.py` | Build the 20-question Autocast pilot set |
| `rebuttal_analyze_mini.py` | Analyze the 12-question mini pilot (B3/B4) |
| `rebuttal_snapshot_feasibility.py` / `rebuttal_fetch_snapshots.py` | Wayback feasibility + fetch for the 12-question pilot (superseded by `cr_snapshot_crawl.py`) |
| `rebuttal_run_snapshot_rag.py` | Snapshot-RAG generation, pilot version (superseded by `cr_run_snapshot_rag.py`) |
| `rebuttal_analyze_snapshot.py` | Pilot snapshot comparison (superseded by `cr_analyze_snapshot.py`) |
| `rebuttal_run_judges.py` / `rebuttal_prompt_robustness.py` | Judge-robustness and prompt-robustness experiments |
| `rebuttal_build_challenge.py` | Build the 120-claim taxonomy/date challenge set |
| `rebuttal_prepare_human_taxonomy_120.py` | Blinded sheet for human annotator 1 |
| `rebuttal_score_human_taxonomy_120.py` | Score judges against annotator 1 |
| `rebuttal_prepare_human_audit.py`, `rebuttal_prepare_atomic_flagged_sheet.py`, `rebuttal_prepare_verifier_adjudication.py`, `rebuttal_prepare_professor_overlap.py`, `rebuttal_prepare_unverifiable_sample.py`, `rebuttal_prepare_dual_provenance_audit.py` | Sheets for the various human audit tracks |
| `rebuttal_score_atomic_adjudication.py`, `rebuttal_score_professor_overlap.py`, `rebuttal_score_verifier_vs_professor.py` | Scoring for those audit tracks |
| `rebuttal_run_provenance_audit.py`, `rebuttal_collect_provenance_samples.py`, `rebuttal_postprocess_provenance.py`, `rebuttal_apply_provenance_corrections.py`, `rebuttal_record_provenance_confirmation.py`, `rebuttal_summarize_human_provenance.py` | Provenance audit pipeline |
| `rebuttal_decompose_disagreements.py` | Error decomposition of judge-human disagreements |

## Camera-ready experiments (E1/E3/E5/E6; results in `camera_ready/`)

| Script | Purpose |
|---|---|
| `cr_prepare_annotator2_sheet.py` | E1: blinded, re-shuffled sheet for the second annotator (seed 7) |
| `cr_score_iaa.py` | E1: inter-annotator agreement (kappa, confusion, consistency checks) |
| `cr_sample_treatment.py` | E3/E6: seeded treatment-question sampling from `data/binary.jsonl` |
| `cr_snapshot_crawl.py` | E3: resumable Wayback crawl (CDX lookup + fetch, up to 3 URLs/question) |
| `cr_run_snapshot_rag.py` | E3: Snapshot-RAG generation from archived contexts |
| `cr_run_e3_full.sh` | E3: full detached chain (crawl-wait, generation x3 arms, eval x3, analyses) |
| `cr_analyze_snapshot.py` | E3: three-arm comparison with bootstrap CIs (`comparison.json`) |
| `cr_e5_metric_comparison.py` | E5: OLR / Top-K / LOO vs. Shapley-DCLR from cached coalition audits |
| `cr_loo_topup.py` | E5: top up missing leave-one-out coalitions (probe caches needed none) |
| `cr_run_e6_full.sh` | E6: instrumented 100-question x 4-method generation + evaluation |
| `cr_e6_cost_table.py` | E6: aggregate per-call usage log into the per-stage cost table |
| `cr_status.sh` | Status of detached camera-ready jobs (PIDs, logs, result counts) |

Exact invocations used for the camera-ready runs are recorded in
`camera_ready/logs/*.log` and in the shell wrappers (`cr_run_e3_full.sh`,
`cr_run_e6_full.sh`).
