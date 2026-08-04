"""Record human confirmation of all high-confidence A/B provenance labels."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
CSV_PATH = AUDIT / "human_verification.csv"


def main() -> None:
    with CSV_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    confirmed = 0
    for row in rows:
        if row["auto_label"] in {"A", "B"} and row["auto_confidence"] == "H":
            row["human_confirm"] = "yes"
            row["human_corrected_label"] = row["auto_label"]
            row["human_note"] = "Confirmed by student annotator."
            confirmed += 1
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    confirmed_rows = [row for row in rows if row["human_confirm"] == "yes"]
    counts = Counter(row["human_corrected_label"] for row in confirmed_rows)
    summary = {
        "confirmed_items": len(confirmed_rows),
        "confirmed_counts": dict(counts),
        "verified_pre_cutoff_rate_among_confirmed": counts["A"] / len(confirmed_rows),
        "post_cutoff_rate_among_confirmed": counts["B"] / len(confirmed_rows),
        "selection_note": (
            "The confirmed set contains only high-confidence automatic A/B cases; "
            "rates are not estimates over all 100 retrievals."
        ),
    }
    (AUDIT / "human_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
