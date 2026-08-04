"""Check pre-cutoff Wayback snapshot coverage for the 12-question pilot."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import quote

import httpx


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "rebuttal_validation" / "data" / "autocast_rebuttal_20.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "results" / "snapshot_feasibility.json"


async def lookup(client: httpx.AsyncClient, row: dict, semaphore: asyncio.Semaphore) -> dict:
    source_url = row.get("source_url") or ""
    result = {
        "instance_id": row["id"],
        "cutoff": row["temporal_cutoff"],
        "source_url": source_url,
        "snapshot_timestamp": None,
        "snapshot_url": None,
        "available": False,
        "error": None,
    }
    if not source_url:
        result["error"] = "missing_source_url"
        return result
    endpoint = (
        "https://web.archive.org/cdx/search/cdx?"
        f"url={quote(source_url, safe='')}&output=json&filter=statuscode%3A200"
        f"&to={row['temporal_cutoff'].replace('-', '')}&limit=-1"
        "&fl=timestamp%2Coriginal%2Cstatuscode"
    )
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list) and len(payload) > 1:
                    timestamp, original, _ = payload[-1]
                    result["snapshot_timestamp"] = timestamp
                    result["snapshot_url"] = f"https://web.archive.org/web/{timestamp}id_/{original}"
                    result["available"] = True
                return result
            except Exception as exc:
                result["error"] = str(exc)
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        return result


async def run(rows: list[dict]) -> list[dict]:
    timeout = httpx.Timeout(70.0)
    headers = {"User-Agent": "Mozilla/5.0 (TimeSPEC academic validation)"}
    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        return await asyncio.gather(*(lookup(client, row, semaphore) for row in rows))


def main() -> None:
    rows = [json.loads(line) for line in DATASET.open(encoding="utf-8") if line.strip()][:12]
    results = asyncio.run(run(rows))
    available = sum(bool(row["available"]) for row in results)
    report = {
        "n_questions": len(results),
        "available_questions": available,
        "coverage": available / len(results),
        "gate_passed": available >= 8,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_questions", "available_questions", "coverage", "gate_passed")}, indent=2))


if __name__ == "__main__":
    main()
