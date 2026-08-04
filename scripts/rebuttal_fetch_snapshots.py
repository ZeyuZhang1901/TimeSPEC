"""Fetch and extract text from Wayback snapshots passing the feasibility gate."""

from __future__ import annotations

import asyncio
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal_validation" / "results" / "snapshot_feasibility.json"
OUTPUT = ROOT / "rebuttal_validation" / "results" / "snapshot_contexts.json"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            text = re.sub(r"\s+", " ", data).strip()
            if len(text) > 1:
                self.parts.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


async def fetch_one(client: httpx.AsyncClient, row: dict, semaphore: asyncio.Semaphore) -> tuple[str, dict]:
    result = {
        "snapshot_url": row.get("snapshot_url"),
        "snapshot_timestamp": row.get("snapshot_timestamp"),
        "source_url": row.get("source_url"),
        "usable": False,
        "text": "",
        "text_chars": 0,
        "error": None,
    }
    if not row.get("available") or not row.get("snapshot_url"):
        result["error"] = "snapshot_unavailable"
        return row["instance_id"], result
    async with semaphore:
        try:
            response = await client.get(row["snapshot_url"])
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            text = parser.text()
            # Favor the leading content but retain enough context for retrieval.
            text = text[:6000]
            result["text"] = text
            result["text_chars"] = len(text)
            result["usable"] = len(text) >= 500
        except Exception as exc:
            result["error"] = str(exc)
    return row["instance_id"], result


async def run(rows: list[dict]) -> dict[str, dict]:
    timeout = httpx.Timeout(90.0)
    headers = {"User-Agent": "Mozilla/5.0 (TimeSPEC academic validation)"}
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        pairs = await asyncio.gather(*(fetch_one(client, row, semaphore) for row in rows))
    return dict(pairs)


def main() -> None:
    feasibility = json.loads(INPUT.read_text(encoding="utf-8"))
    contexts = asyncio.run(run(feasibility["results"]))
    usable = sum(bool(row["usable"]) for row in contexts.values())
    report = {
        "n_questions": len(contexts),
        "available_snapshots": feasibility["available_questions"],
        "usable_contexts": usable,
        "coverage": usable / len(contexts),
        "gate_passed": usable >= 8,
        "contexts": contexts,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_questions", "available_snapshots", "usable_contexts", "coverage", "gate_passed")}, indent=2))


if __name__ == "__main__":
    main()
