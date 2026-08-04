"""Prepare a deterministic 20-item human check of auto-unverifiable retrievals."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
INPUT = AUDIT / "conservative_auto_results.jsonl"
OUTPUT = AUDIT / "verification_unverifiable_sample.md"


def main() -> None:
    rows = [
        json.loads(line)
        for line in INPUT.open(encoding="utf-8")
        if line.strip()
    ]
    candidates = [row for row in rows if row["auto_label"] == "C"]
    random.Random(42).shuffle(candidates)
    selected = candidates[:20]
    lines = [
        "# Human Check: Unverifiable Sample",
        "",
        "Verify whether each C label is appropriate. Respond `CONFIRM ALL C` or list corrections as "
        "`R### | A/B/C | H/M/L | note`.",
        "",
    ]
    for row in selected:
        lines.extend(
            [
                f"## {row['sample_id']} — Auto C ({row['auto_confidence']})",
                "",
                f"- Cutoff: {row['cutoff_date']}",
                f"- Claim: {row['retrieved_text'][:500].replace(chr(10), ' ')}",
                f"- Source: {row['source_url']}",
                f"- Pre-cutoff snapshot: {row.get('pre_snapshot_url') or 'none'}",
                f"- Post-cutoff snapshot: {row.get('post_snapshot_url') or 'none'}",
                f"- Auto reason: {row['auto_reason']}",
                "",
            ]
        )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    (AUDIT / "unverifiable_sample_ids.json").write_text(
        json.dumps([row["sample_id"] for row in selected], indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"n": len(selected), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
