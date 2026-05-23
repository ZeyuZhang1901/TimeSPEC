"""Command-line interface for leakage evaluation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.config import load_config
from LeakageEvaluationFramework.leakage_eval.io.loaders import load_completion_rows, load_dataset_metadata
from LeakageEvaluationFramework.leakage_eval.io.metadata import (
    export_quality_gated_eval_metadata,
    validate_metadata_for_completions,
    write_metadata,
)
from LeakageEvaluationFramework.leakage_eval.pipeline import run_evaluation_job


def main() -> None:
    """Run the CLI."""

    parser = argparse.ArgumentParser(prog="leakage_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run leakage evaluation")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--input", required=True, type=Path)
    run_parser.add_argument("--dataset", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--method", required=True)
    run_parser.add_argument("--max-instances", type=int, default=None)
    run_parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated instance_id values to evaluate (filters input rows; order preserved).",
    )
    run_parser.add_argument("--taxonomy-backend", choices=["llm", "local"], default="llm")
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--checkpoint-every", type=int, default=200,
        help="Write results every N instances for crash resilience (default 200).")

    audit_parser = subparsers.add_parser("audit-convergence", help="Run Shapley convergence audit")
    audit_parser.add_argument("--config", required=True, type=Path)
    audit_parser.add_argument("--results", required=True, type=Path)

    summary_parser = subparsers.add_parser("summarize", help="Summarize a completed evaluation")
    summary_parser.add_argument("--results", required=True, type=Path)

    metadata_parser = subparsers.add_parser("export-metadata", help="Export dataset metadata for leakage evaluation")
    metadata_parser.add_argument("--data-root", required=True, type=Path)
    metadata_parser.add_argument("--output", required=True, type=Path)
    metadata_parser.add_argument("--tasks", default="legal,salary,stock")

    validate_parser = subparsers.add_parser("validate-metadata", help="Validate metadata against completion rows")
    validate_parser.add_argument("--input", required=True, type=Path)
    validate_parser.add_argument("--dataset", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "run":
        config = load_config(
            args.config,
            input_path=args.input,
            dataset_path=args.dataset,
            output_dir=args.output,
        )
        instance_ids = None
        if args.instance_ids:
            instance_ids = tuple(id.strip() for id in args.instance_ids.split(",") if id.strip())

        summary = asyncio.run(
            run_evaluation_job(
                config,
                model=args.model,
                method=args.method,
                max_instances=args.max_instances,
                instance_ids=instance_ids,
                taxonomy_backend=args.taxonomy_backend,
                concurrency=args.concurrency,
                checkpoint_every=args.checkpoint_every,
            )
        )
        print(f"Wrote leakage evaluation to {config.output_dir}")
        print(summary)
        return
    if args.command == "audit-convergence":
        raise SystemExit("Convergence audit runner is scaffolded but not wired yet.")
    if args.command == "summarize":
        raise SystemExit("Summary CLI is scaffolded but not wired yet.")
    if args.command == "export-metadata":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        metadata = export_quality_gated_eval_metadata(args.data_root, tasks)
        write_metadata(args.output, metadata)
        print(f"Wrote {len(metadata)} metadata rows to {args.output}")
        return
    if args.command == "validate-metadata":
        rows = load_completion_rows(args.input)
        metadata = load_dataset_metadata(args.dataset)
        errors = validate_metadata_for_completions(rows, metadata)
        if errors:
            preview = "\n".join(errors[:20])
            raise SystemExit(f"Metadata validation failed with {len(errors)} errors:\n{preview}")
        print(f"Metadata validation passed for {len(rows)} completion rows.")
        return


if __name__ == "__main__":
    main()

