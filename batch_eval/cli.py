"""CLI for batch-mode (Anthropic) leakage evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from LeakageEvaluationFramework.batch_eval.pipeline import run_batch_job
from LeakageEvaluationFramework.leakage_eval.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="batch_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Submit/poll/fetch a batch evaluation job.")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--dataset", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--model", required=True, help="Generator model id (read from completions).")
    run.add_argument("--method", required=True)
    run.add_argument("--max-instances", type=int, default=None)
    run.add_argument("--poll-interval-sec", type=float, default=30.0)
    run.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    for noisy in ("httpcore", "httpx", "hpack", "h2", "urllib3", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "run":
        config = load_config(
            args.config,
            input_path=args.input,
            dataset_path=args.dataset,
            output_dir=args.output,
        )
        summary = run_batch_job(
            config,
            model=args.model,
            method=args.method,
            max_instances=args.max_instances,
            poll_interval_sec=args.poll_interval_sec,
        )
        print(f"Wrote batch evaluation to {config.output_dir}")
        print(summary)


if __name__ == "__main__":
    main()
