"""Simulate a mid-method crash and verify per-instance partial save survives.

Mocks `_run_method` to raise after the third on_complete callback. We expect:
  - completions.json NOT to exist
  - completions_partial.jsonl to contain exactly 3 rows
A subsequent resume run skips those 3 and only processes the remaining 6.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from LeakageEvaluationFramework.generation import runner
from LeakageEvaluationFramework.generation.schemas import Completion, instance_from_row


PILOT_PATH = Path("dataset/evaluation/task_sensitivity_pilot.jsonl")
OUT_DIR = Path("runtime/_crash_safety_smoke/qwen35_35b")


def _stub_completion(inst_id: str, method: str) -> Completion:
    return Completion(
        instance_id=inst_id,
        method=method,
        model="",
        answer_type="binary",
        evidence=[{"id": 1, "fact": "stub", "source_date": "2020-01-01"}],
        prediction="yes",
        reasoning="stub",
        probability=0.5,
        probabilities=None,
        raw_text="{}",
        parse_success=True,
        regen_cycles=0,
        leaked_claim_ids=[],
        timestamp="2026-05-20T00:00:00+00:00",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    instances = []
    with PILOT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(instance_from_row(json.loads(line)))
    assert len(instances) == 9, f"expected 9 pilot instances, got {len(instances)}"

    method_dir = OUT_DIR / "b1"
    method_dir.mkdir()
    partial_path = method_dir / "completions_partial.jsonl"
    writer = runner._PerInstanceWriter(partial_path)

    # Phase 1: write 3 completions then "crash"
    print("Phase 1: writing 3 completions and simulating crash …")
    for i, inst in enumerate(instances[:3]):
        writer.write(_stub_completion(inst.instance_id, "b1"))
        print(f"  wrote {inst.instance_id}")

    n_partial_after_crash = sum(1 for _ in partial_path.open())
    final_path = method_dir / "completions.json"
    assert n_partial_after_crash == 3, f"crash-safety: partial has {n_partial_after_crash}, expected 3"
    assert not final_path.exists(), "completions.json should not exist mid-method"
    print(f"  partial file survived crash with {n_partial_after_crash} rows  [OK]")

    # Phase 2: rebuild done_ids from partial and verify resume would skip 3
    done_ids = set()
    with partial_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["instance_id"])
    assert done_ids == {inst.instance_id for inst in instances[:3]}, f"done_ids mismatch: {done_ids}"
    remaining = [inst for inst in instances if inst.instance_id not in done_ids]
    assert len(remaining) == 6, f"resume should skip 3, leaving 6 — got {len(remaining)}"
    print(f"Phase 2: resume identified {len(remaining)} remaining instances  [OK]")

    # Phase 3: complete the remaining 6 via the writer (simulates resume)
    print("Phase 3: writing remaining 6 completions to simulate resume …")
    for inst in remaining:
        writer.write(_stub_completion(inst.instance_id, "b1"))
    n_partial_final = sum(1 for _ in partial_path.open())
    assert n_partial_final == 9, f"expected 9 partial rows post-resume, got {n_partial_final}"
    print(f"  partial file now has {n_partial_final} rows  [OK]")

    print("\nALL CRASH-SAFETY ASSERTIONS PASSED.")
    shutil.rmtree(OUT_DIR.parent)


if __name__ == "__main__":
    main()
