"""Prepare a small, deterministic Autocast Treatment subset for rebuttal runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "rebuttal_validation" / "data" / "autocast" / "autocast_questions.json"
DEFAULT_OUTPUT = ROOT / "rebuttal_validation" / "data" / "autocast_rebuttal_20.jsonl"


DOMAIN_TERMS = {
    "finance": {"Finance", "Economic Indicators", "Economic Policy", "Business", "Finance – markets"},
    "politics": {
        "US Politics",
        "Politics – US",
        "Foreign Policy",
        "Politics and Elections",
        "Elections and Referenda",
        "Non-US Politics",
        "Geopolitics",
    },
    "science": {
        "Technology",
        "Technological Advances",
        "Health",
        "Environment",
        "Scientific Discoveries",
        "Physical Sciences – Astrophysics and Cosmology",
        "Biological Sciences – Medicine",
        "Computer Science – AI and Machine Learning",
    },
}


def parse_day(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def classify(tags: list[str]) -> str | None:
    tag_set = set(tags)
    for domain, terms in DOMAIN_TERMS.items():
        if tag_set & terms:
            return domain
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    questions = json.loads(args.input.read_text(encoding="utf-8"))
    eligible = []
    for row in questions:
        domain = classify(row.get("tags") or [])
        if (
            row.get("qtype") != "t/f"
            or row.get("status") != "Resolved"
            or str(row.get("answer", "")).lower() not in {"yes", "no"}
            or domain is None
        ):
            continue
        publish = parse_day(row["publish_time"])
        close = parse_day(row["close_time"])
        if close > "2024-03-31":
            continue
        background = (row.get("background") or "").strip()
        if not background or len(background) > 2500:
            continue
        eligible.append((domain, publish, row))

    quotas = {domain: args.count // len(DOMAIN_TERMS) for domain in DOMAIN_TERMS}
    for domain in list(quotas)[: args.count % len(DOMAIN_TERMS)]:
        quotas[domain] += 1

    selected: list[dict] = []
    seen_questions: set[str] = set()
    for domain in DOMAIN_TERMS:
        candidates = sorted(
            (item for item in eligible if item[0] == domain),
            key=lambda item: (item[1], item[2]["id"]),
        )
        # Spread selection across the available timeline.
        if not candidates:
            continue
        step = max(1, len(candidates) // quotas[domain])
        for _, _, row in candidates[::step]:
            if len([x for x in selected if x["domain"] == domain]) >= quotas[domain]:
                break
            question = row["question"].strip()
            if question in seen_questions:
                continue
            seen_questions.add(question)
            selected.append(
                {
                    "id": f"autocast_{row['id']}",
                    "source": "autocast",
                    "question": question,
                    "background": (row.get("background") or "").strip(),
                    "temporal_cutoff": parse_day(row["publish_time"]),
                    "answer": str(row["answer"]).lower(),
                    "answer_type": "binary",
                    "resolution_date": parse_day(row["close_time"]),
                    "group": "treatment",
                    "domain": domain,
                    "choices": ["yes", "no"],
                    "source_url": (row.get("source_links") or [""])[0],
                    "tags": row.get("tags") or [],
                }
            )

    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} rows, selected {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {domain: len([row for row in selected if row["domain"] == domain]) for domain in DOMAIN_TERMS}
    print(json.dumps({"output": str(args.output), "n": len(selected), "domains": counts}, indent=2))


if __name__ == "__main__":
    main()
