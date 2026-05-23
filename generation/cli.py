"""CLI entry point for the B1–B4 generation pipeline.

Usage:
    uv run python -m LeakageEvaluationFramework.generation.cli run \\
        --dataset  dataset/benchmark/final/binary.jsonl \\
        --output   runtime/benchmark/claude_sonnet4 \\
        --model    anthropic/claude-sonnet-4 \\
        --methods  b1,b2,b3,b4 \\
        --split    all \\
        --concurrency 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generation.cli",
        description="Run B1–B4 generation methods on the temporal leakage benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one or more generation methods.")
    run_p.add_argument("--dataset", required=True, type=Path, help="Path to benchmark JSONL file.")
    run_p.add_argument("--output", required=True, type=Path, help="Output root directory for this model run.")
    run_p.add_argument("--model", required=True, help="OpenRouter model ID (e.g. anthropic/claude-sonnet-4).")
    run_p.add_argument("--methods", default="b1,b2,b3,b4", help="Comma-separated methods to run.")
    run_p.add_argument("--split", default="all", choices=["all", "treatment", "control"])
    run_p.add_argument("--max-instances", type=int, default=None, help="Cap on number of instances.")
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--max-tokens", type=int, default=4096)
    run_p.add_argument("--judge-model", default="google/gemini-2.0-flash-lite-001")
    run_p.add_argument("--batch-size", type=int, default=8)
    run_p.add_argument("--max-regen-cycles", type=int, default=1)
    run_p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "8")),
        help=(
            "Number of parallel worker threads for RAG retrieval and per-instance "
            "supervisor calls in B2/B4. Overrides the CONCURRENCY env var (default 8)."
        ),
    )
    run_p.add_argument(
        "--shared-rag-cache",
        type=Path,
        default=None,
        help=(
            "Directory for a cross-model RAG search cache.  "
            "When multiple models are run on the same dataset, pointing them all at "
            "the same directory lets later runs reuse Perplexity results fetched by "
            "earlier runs, cutting Perplexity costs by up to 67%% (3 models × 1 fetch). "
            "Defaults to <output>/../shared_rag_cache relative to --output."
        ),
    )
    run_p.add_argument("--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # Silence extremely noisy third-party libraries regardless of --verbose
    for noisy in ("httpcore", "httpx", "hpack", "h2", "urllib3", "asyncio", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "run":
        _run(args)


def _run(args: argparse.Namespace) -> None:
    from LeakageEvaluationFramework.generation.runner import build_config, run_all_methods

    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid = {"b1", "b2", "b3", "b4"}
    unknown = set(methods) - valid
    if unknown:
        print(f"Error: unknown methods: {unknown}. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    # Resolve shared RAG cache directory (default: sibling of --output)
    shared_rag_cache: Path | None = args.shared_rag_cache
    if shared_rag_cache is None and any(m in {"b3", "b4"} for m in methods):
        shared_rag_cache = args.output.parent / "shared_rag_cache"
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Using default shared RAG cache: %s", shared_rag_cache
        )

    cfg = build_config(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        max_regen_cycles=args.max_regen_cycles,
    )

    run_all_methods(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        cfg=cfg,
        methods=methods,
        split=args.split,
        max_instances=args.max_instances,
        concurrency=args.concurrency,
        shared_rag_cache_dir=shared_rag_cache,
    )


if __name__ == "__main__":
    main()
