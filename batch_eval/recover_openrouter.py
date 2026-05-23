"""Recover failed Shapley coalitions for a finished batch_eval job via OpenRouter.

Use case: an Anthropic batch finished with some `errored` requests (e.g. credit
exhaustion mid-batch). The successful coalition values are fine to keep; we just
need to re-evaluate the failed coalitions through OpenRouter and re-solve the
Shapley regression for every affected instance.

CRASH SAFETY (this matters because each OpenRouter call costs money):
  * Every successful OpenRouter prediction is appended to
    ``state/or_recoveries.jsonl`` IMMEDIATELY (line-buffered + flush). On a
    crash/restart, those custom_ids are loaded back and skipped.
  * Each per-instance finalize result is appended to
    ``state/recovered_instances.jsonl`` IMMEDIATELY. On a crash/restart, those
    instance_ids are loaded back and skipped.
  * The canonical ``instance_results.jsonl`` / ``shapley_results.jsonl`` are
    written ATOMICALLY only at the very end, once all affected instances are
    re-finalized. Originals are copied to ``dedupe_backup/pre_recovery_*``.

DATA INTEGRITY:
  * When the on-disk claim decisions are incomplete for some instance (some
    taxonomy results were dropped by the original batch), missing claim_ids are
    padded with ``local_claim_decision`` before computing Shapley.

Workflow:
  1. Read state/shapley_batches.json + state/shapley_requests.jsonl.
  2. Stream Anthropic batch results: classify each custom_id as succeeded (raw
     text) or failed.
  3. Re-run failed coalitions through OpenRouter (resumable, streamed to disk).
  4. For each affected instance: pad decisions if needed, build plan, call
     ``_finalize_instance``, append the result to recovered_instances.jsonl.
  5. Merge recovered + unaffected rows into the canonical output files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from LeakageEvaluationFramework.batch_eval.anthropic_batch import BatchResult
from LeakageEvaluationFramework.batch_eval.pipeline import (
    _claim_records_for_instance,
    _finalize_instance,
    _instance_row,
    _load_claim_decisions_from_disk,
    _plan_instance,
    _skipped_shapley,
)
from LeakageEvaluationFramework.leakage_eval.config import load_config
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterCoalitionPredictor,
)
from LeakageEvaluationFramework.leakage_eval.io.loaders import (
    build_eval_inputs,
    load_completion_rows,
    load_dataset_metadata,
)
from LeakageEvaluationFramework.leakage_eval.metrics.aggregation import aggregate_instance_rows
from LeakageEvaluationFramework.leakage_eval.metrics.dclr import leakage_rate
from LeakageEvaluationFramework.leakage_eval.metrics.leakage import decision_labels
from LeakageEvaluationFramework.leakage_eval.metrics.performance import performance_score
from LeakageEvaluationFramework.leakage_eval.schemas import EvalOutput
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import local_claim_decision

log = logging.getLogger("recover_openrouter")


# ============================================================
# Incremental persistence stores (the thing I should have built first)
# ============================================================


class _AppendOnlyJsonl:
    """JSONL file that flushes after every line. Safe for resume after crash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()

    def write_sync(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    async def write_async(self, record: dict) -> None:
        async with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()


def _load_jsonl_dict(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in rec:
                out[rec[key]] = rec
    return out


# ============================================================
# Step 1 + 2: pull every coalition's outcome from Anthropic
# ============================================================


def _classify_batch_results(state_dir: Path, anthropic_client) -> dict[str, BatchResult | None]:
    """Return {custom_id: BatchResult or None}. None means the request failed."""
    batches = json.loads((state_dir / "shapley_batches.json").read_text())
    out: dict[str, BatchResult | None] = {}
    for h in batches:
        log.info("Fetching batch %s ...", h["batch_id"])
        n_ok = n_fail = 0
        for r in anthropic_client.messages.batches.results(h["batch_id"]):
            if r.result.type == "succeeded":
                msg = r.result.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                if text:
                    out[r.custom_id] = BatchResult(custom_id=r.custom_id, text=text, error=None)
                    n_ok += 1
                else:
                    out[r.custom_id] = None
                    n_fail += 1
            else:
                out[r.custom_id] = None
                n_fail += 1
        log.info("  succeeded=%d, failed=%d", n_ok, n_fail)
    return out


# ============================================================
# Step 3: re-run failed coalitions through OpenRouter (incremental)
# ============================================================


async def _recover_failed(
    failed: list[tuple[str, str, tuple[int, ...]]],
    inst_by_id: dict[str, "EvalInput"],
    predictor: OpenRouterCoalitionPredictor,
    concurrency: int,
    or_store: _AppendOnlyJsonl,
    already_done: set[str],
) -> dict[str, str]:
    """Return {custom_id: prediction_json_text}. Streams every result to or_store."""
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, str] = {}
    progress = {"done": 0, "ok": 0, "err": 0, "skip": 0}

    pending = [(cid, iid, mk) for cid, iid, mk in failed if cid not in already_done]
    progress["skip"] = len(failed) - len(pending)
    if progress["skip"]:
        log.info("Skipping %d already-recovered coalitions (resumed from disk)", progress["skip"])

    async def _one(cid: str, iid: str, mask: tuple[int, ...]) -> None:
        async with sem:
            inst = inst_by_id[iid]
            subset = [item for item, bit in zip(inst.evidence, mask) if bit]
            try:
                pred = await predictor.predict(inst, subset)
                text = json.dumps(pred)
                await or_store.write_async({"custom_id": cid, "text": text})
                out[cid] = text
                progress["ok"] += 1
            except Exception as exc:
                progress["err"] += 1
                if progress["err"] <= 20:
                    log.warning("OpenRouter failed for %s: %s", cid, exc)
            progress["done"] += 1
            if progress["done"] % 500 == 0:
                log.info(
                    "  OpenRouter recovery: %d/%d  (ok=%d, err=%d)",
                    progress["done"], len(pending), progress["ok"], progress["err"],
                )

    await asyncio.gather(*[_one(cid, iid, mk) for cid, iid, mk in pending])
    log.info(
        "OpenRouter recovery done: %d ok, %d err, %d total (this run); %d already done from prior runs",
        progress["ok"], progress["err"], len(pending), progress["skip"],
    )
    return out


# ============================================================
# Step 4: pad decisions, finalize per-instance, stream to disk
# ============================================================


def _padded_decisions(inst, raw_decisions):
    """Return one decision per evidence position, aligned with evidence order.

    The on-disk claim_results.jsonl is deduped by (instance_id, claim_id), but
    the Shapley value vector has one entry per evidence POSITION. When an
    instance's evidence contains duplicate item.id values (an upstream
    extraction artifact), or when the LLM verifier omitted some claim_ids, the
    on-disk row count is smaller than the evidence count. We replicate the
    in-memory behavior of the original pipeline: walk evidence positions, look
    up each position's decision by item.id, and fall back to
    ``local_claim_decision`` if no decision exists for that id.
    """
    by_id = {d.claim_id: d for d in raw_decisions}
    records = _claim_records_for_instance(inst)
    out = []
    n_filled_from_disk = 0
    n_filled_from_local = 0
    for record in records:
        if record.claim_id in by_id:
            out.append(by_id[record.claim_id])
            n_filled_from_disk += 1
        else:
            out.append(local_claim_decision(record))
            n_filled_from_local += 1
    if n_filled_from_local > 0:
        log.debug(
            "Instance %s: padded %d positions with local_claim_decision (n_evidence=%d, n_unique_disk_ids=%d)",
            inst.instance_id, n_filled_from_local, len(records), len(by_id),
        )
    return out


def _make_skipped_output(inst, decisions, reason) -> EvalOutput:
    labels = decision_labels(decisions) if decisions else []
    leaked_claims = sum(labels) if labels else 0
    shapley_result = _skipped_shapley(
        inst,
        (
            "No leaked claims; Shapley-DCLR is exactly zero."
            if reason == "no_leakage"
            else f"Skipped: {reason}"
        ),
    )
    return EvalOutput(
        instance_id=inst.instance_id,
        model=inst.model,
        method=inst.method,
        task=inst.task,
        n_claims=len(inst.evidence),
        leaked_claims=leaked_claims,
        olr=leakage_rate(labels) if labels else 0.0,
        shapley_dclr=0.0,
        performance=performance_score(inst.task, inst.prediction, inst.ground_truth),
        claim_decisions=decisions,
        shapley_result=shapley_result,
    )


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _backup_outputs(job_output_dir: Path) -> None:
    backup_dir = job_output_dir / "dedupe_backup"
    backup_dir.mkdir(exist_ok=True)
    for fname in ("instance_results.jsonl", "shapley_results.jsonl", "summary.json"):
        src = job_output_dir / fname
        dst = backup_dir / f"pre_recovery_{fname}"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


# ============================================================
# Main
# ============================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Existing batch_eval output dir.")
    p.add_argument("--model", required=True, help="OpenRouter model id (e.g. anthropic/claude-sonnet-4).")
    p.add_argument("--method", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit-instances", type=int, default=None,
                   help="Smoke test: only process this many affected instances.")
    p.add_argument("--only-instance-ids", default=None,
                   help="Comma-separated instance_ids to recover (overrides --limit-instances).")
    p.add_argument("--finalize-only", action="store_true",
                   help="Skip OpenRouter calls; just finalize using existing or_recoveries.jsonl.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    for noisy in ("httpcore", "httpx", "anthropic", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv(Path(".env"))
    from anthropic import Anthropic
    anth = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    config = load_config(args.config, input_path=args.input, dataset_path=args.dataset, output_dir=args.output)

    rows = load_completion_rows(config.input_path)
    meta = load_dataset_metadata(config.dataset_path)
    instances = build_eval_inputs(rows, meta, model=args.model, method=args.method)
    inst_by_id = {i.instance_id: i for i in instances}
    log.info("Loaded %d instances", len(instances))

    requests_index = {
        json.loads(line)["custom_id"]: json.loads(line)
        for line in (args.output / "state/shapley_requests.jsonl").read_text().splitlines()
        if line.strip()
    }
    log.info("Indexed %d coalition requests from state", len(requests_index))

    # Step 1-2: pull batch outcomes
    text_by_id = _classify_batch_results(args.output / "state", anth)
    failed_ids = sorted(cid for cid, br in text_by_id.items() if br is None)
    log.info("Failed coalitions to recover: %d / %d total", len(failed_ids), len(text_by_id))

    affected_instances = {requests_index[cid]["instance_id"] for cid in failed_ids}
    log.info("Affected instances: %d", len(affected_instances))
    if args.only_instance_ids:
        keep = set(s.strip() for s in args.only_instance_ids.split(",") if s.strip())
        missing = keep - affected_instances
        if missing:
            log.warning("Requested IDs not in affected set (will be ignored): %s", sorted(missing))
        keep = keep & affected_instances
        log.info("RESTRICTED to %d explicit instance_ids: %s", len(keep), sorted(keep))
        affected_instances = keep
        failed_ids = [cid for cid in failed_ids if requests_index[cid]["instance_id"] in keep]
        log.info("Failed coalitions in restricted set: %d", len(failed_ids))
    elif args.limit_instances is not None:
        keep = set(sorted(affected_instances)[: args.limit_instances])
        log.info("LIMITED to %d instances for testing: %s", len(keep), sorted(keep))
        affected_instances = keep
        failed_ids = [cid for cid in failed_ids if requests_index[cid]["instance_id"] in keep]
        log.info("Failed coalitions in limited set: %d", len(failed_ids))

    failed_triples = [
        (cid, requests_index[cid]["instance_id"], tuple(requests_index[cid]["mask"]))
        for cid in failed_ids
    ]

    # Step 3: incremental OpenRouter recovery (resume-safe)
    state_dir = args.output / "state"
    or_store_path = state_dir / "or_recoveries.jsonl"
    already_done_map = _load_jsonl_dict(or_store_path, "custom_id")
    log.info("Loaded %d prior OpenRouter recoveries from %s", len(already_done_map), or_store_path)

    or_store = _AppendOnlyJsonl(or_store_path)
    try:
        if args.finalize_only:
            log.info("--finalize-only set: skipping new OpenRouter calls.")
            new_or_recoveries: dict[str, str] = {}
        else:
            predictor = OpenRouterCoalitionPredictor(model=args.model, temperature=0.0, max_tokens=128)
            already_done_ids = set(already_done_map.keys())
            new_or_recoveries = asyncio.run(_recover_failed(
                failed_triples, inst_by_id, predictor, args.concurrency,
                or_store, already_done_ids,
            ))
    finally:
        or_store.close()

    # Merge prior + new OR recoveries
    or_text_by_id: dict[str, str] = {cid: rec["text"] for cid, rec in already_done_map.items()}
    or_text_by_id.update(new_or_recoveries)
    log.info("Total OR recoveries available: %d", len(or_text_by_id))

    # Step 4: per-instance results_by_mask
    per_inst_mask_results: dict[str, dict[tuple[int, ...], BatchResult]] = defaultdict(dict)
    for cid, rec in requests_index.items():
        iid = rec["instance_id"]
        if iid not in affected_instances:
            continue
        mask = tuple(rec["mask"])
        if cid in or_text_by_id:
            per_inst_mask_results[iid][mask] = BatchResult(
                custom_id=cid, text=or_text_by_id[cid], error=None,
            )
        else:
            br = text_by_id.get(cid)
            if br is not None:
                per_inst_mask_results[iid][mask] = br

    # Step 5: finalize each instance, stream result to recovered_instances.jsonl
    decisions_by_inst = _load_claim_decisions_from_disk(args.output)
    finalized_path = state_dir / "recovered_instances.jsonl"
    already_finalized = _load_jsonl_dict(finalized_path, "instance_id")
    log.info("Loaded %d prior finalized instances from %s", len(already_finalized), finalized_path)

    finalized_store = _AppendOnlyJsonl(finalized_path)
    new_outputs: dict[str, EvalOutput] = {}
    n_partial = 0
    n_padded = 0
    try:
        for iid in sorted(affected_instances):
            if iid in already_finalized:
                continue
            inst = inst_by_id[iid]
            raw_decisions = decisions_by_inst.get(iid, [])
            if not raw_decisions:
                output = _make_skipped_output(inst, raw_decisions, "no_claims")
                new_outputs[iid] = output
                finalized_store.write_sync({
                    "instance_id": iid,
                    "instance_row": _instance_row(output),
                    "shapley_row": asdict(output.shapley_result),
                })
                continue
            decisions = _padded_decisions(inst, raw_decisions)
            if len(decisions) != len({d.claim_id for d in raw_decisions}):
                n_padded += 1
            labels = decision_labels(decisions)
            leaked_claims = sum(labels)
            if config.shapley.skip_if_no_leakage and leaked_claims == 0:
                output = _make_skipped_output(inst, decisions, "no_leakage")
            else:
                plan = _plan_instance(inst, config, decisions, labels)
                per_inst = per_inst_mask_results.get(iid, {})
                n_missing = sum(1 for m in plan["masks"] if sum(m) > 0 and m not in per_inst)
                if n_missing > 0:
                    n_partial += 1
                try:
                    output = _finalize_instance(plan, per_inst, config)
                except Exception as exc:
                    log.error("Finalize FAILED for %s: %s — falling back to skipped", iid, exc)
                    output = _make_skipped_output(inst, decisions, f"finalize_error_{type(exc).__name__}")
            new_outputs[iid] = output
            finalized_store.write_sync({
                "instance_id": iid,
                "instance_row": _instance_row(output),
                "shapley_row": asdict(output.shapley_result),
            })
        log.info(
            "Re-finalized %d affected instances this run (%d new, %d already done last run); "
            "%d had padded decisions, %d still partial",
            len(affected_instances), len(new_outputs), len(already_finalized), n_padded, n_partial,
        )
    finally:
        finalized_store.close()

    # Step 6: merge into canonical outputs (only at end, atomically)
    _backup_outputs(args.output)

    inst_path = args.output / "instance_results.jsonl"
    sha_path = args.output / "shapley_results.jsonl"

    existing_instance_rows = [
        json.loads(l) for l in inst_path.read_text().splitlines() if l.strip()
    ]
    existing_shapley_rows = [
        json.loads(l) for l in sha_path.read_text().splitlines() if l.strip()
    ]
    log.info("Existing rows: instance=%d shapley=%d",
             len(existing_instance_rows), len(existing_shapley_rows))

    # Combine: prior-finalized (from previous runs of THIS recovery script) + new-finalized
    all_finalized = _load_jsonl_dict(finalized_path, "instance_id")
    log.info("All finalized instances on disk: %d", len(all_finalized))

    new_iids = set(all_finalized.keys())
    new_instance_rows = [rec["instance_row"] for rec in all_finalized.values()]
    new_shapley_rows = [rec["shapley_row"] for rec in all_finalized.values()]

    merged_inst = (
        [r for r in existing_instance_rows if r["instance_id"] not in new_iids] + new_instance_rows
    )
    merged_sha = (
        [r for r in existing_shapley_rows if r["instance_id"] not in new_iids] + new_shapley_rows
    )

    _atomic_write_jsonl(inst_path, merged_inst)
    _atomic_write_jsonl(sha_path, merged_sha)

    summary = aggregate_instance_rows(merged_inst)
    summary["recovery_n_affected"] = len(new_iids)
    summary["recovery_n_partial_after_or"] = n_partial
    summary["recovery_n_padded_decisions"] = n_padded
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps({
        "n_instances": summary["n_instances"],
        "olr_mean": summary.get("olr", {}).get("mean"),
        "shapley_dclr_mean": summary.get("shapley_dclr", {}).get("mean"),
        "performance_mean": summary.get("performance", {}).get("mean"),
        "recovery_n_affected": len(new_iids),
        "recovery_n_partial_after_or": n_partial,
        "recovery_n_padded_decisions": n_padded,
    }, indent=2))


if __name__ == "__main__":
    main()
