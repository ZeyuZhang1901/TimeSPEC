"""Prepare the nine verifier/professor cutoff-status disagreements."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"


def main() -> None:
    report = json.loads((AUDIT / "verifier_vs_professor.json").read_text(encoding="utf-8"))
    professor = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "professor_labels.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    }
    sources = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (AUDIT / "conservative_auto_results.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    }
    lines = [
        "# Final Adjudication: Nine Cutoff-Status Disagreements",
        "",
        "For each item, read the entire retrieved claim. Choose:",
        "",
        "- `P`: professor label A is correct (all asserted information was pre-cutoff);",
        "- `V`: verifier label B is correct (the text contains a post-cutoff target/outcome);",
        "- `C`: unresolved.",
        "",
        "Respond as `R### | P/V/C | brief note`.",
        "",
    ]
    for item in report["disagreements"]:
        sample_id = item["sample_id"]
        source = sources[sample_id]
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"- Cutoff: {source['cutoff_date']}",
                f"- Full retrieved text: {source['retrieved_text']}",
                f"- Source: {source['source_url']}",
                f"- Professor rationale: {professor[sample_id]['note']}",
                f"- Verifier category/date: {item['predicted_category']} / {item['predicted_effective_date']}",
                f"- Verifier rationale: {item['reason']}",
                "",
            ]
        )
    output = AUDIT / "final_adjudication_9.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(report["disagreements"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
