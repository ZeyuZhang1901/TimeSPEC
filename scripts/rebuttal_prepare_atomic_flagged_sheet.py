"""Prepare a concise human check of atomic claims automatically labeled leaked."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"


def main() -> None:
    rows = [
        json.loads(line)
        for line in (AUDIT / "atomic_disagreement_claims.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    ]
    flagged = [row for row in rows if row["automatic_is_leaked"]]
    lines = [
        "# Human Check: 43 Atomic Claims Flagged as Post-Cutoff",
        "",
        "These claims came from the nine mixed retrieval paragraphs. Confirm whether "
        "each atomic claim is post-cutoff. Respond `CONFIRM ALL FLAGGED` or list "
        "corrections as `R###.N | A/B/C/F | note`.",
        "",
    ]
    for row in flagged:
        claim_id = f"{row['parent_id']}.{row['claim_id']}"
        lines.extend(
            [
                f"## {claim_id}",
                "",
                f"- Cutoff: {row['cutoff_date']}",
                f"- Claim: {row['claim_text']}",
                f"- Source: {row['source_url']}",
                f"- Verifier date/category: {row['automatic_effective_date']} / {row['automatic_category']}",
                f"- Verifier reason: {row['automatic_reason']}",
                "",
            ]
        )
    output = AUDIT / "atomic_flagged_43.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(flagged), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
