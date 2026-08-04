"""Automate archival enrichment and preliminary labels for 100 retrieval items."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from collections import Counter
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import httpx

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal_validation" / "provenance_audit" / "retrieval_samples.jsonl"
OUTPUT = ROOT / "rebuttal_validation" / "provenance_audit"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
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


def cdx_url(url: str, *, to: str | None = None, from_date: str | None = None, limit: int = 1) -> str:
    params = [
        f"url={quote(url, safe='')}",
        "output=json",
        "filter=statuscode%3A200",
        "fl=timestamp%2Coriginal%2Cstatuscode",
        f"limit={limit}",
    ]
    if to:
        params.append(f"to={to.replace('-', '')}")
    if from_date:
        params.append(f"from={from_date.replace('-', '')}")
    return "https://web.archive.org/cdx/search/cdx?" + "&".join(params)


async def get_json(client: httpx.AsyncClient, url: str) -> list:
    for attempt in range(3):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == 2:
                return []
            await asyncio.sleep(1 + attempt)
    return []


async def archive_lookup(client: httpx.AsyncClient, row: dict, semaphore: asyncio.Semaphore) -> dict:
    cutoff = date.fromisoformat(row["cutoff_date"])
    async with semaphore:
        pre_payload, post_payload = await asyncio.gather(
            get_json(client, cdx_url(row["source_url"], to=row["cutoff_date"], limit=-1)),
            get_json(
                client,
                cdx_url(
                    row["source_url"],
                    from_date=(cutoff + timedelta(days=1)).isoformat(),
                    limit=1,
                ),
            ),
        )
    out = dict(row)
    for prefix, payload in (("pre", pre_payload), ("post", post_payload)):
        if isinstance(payload, list) and len(payload) > 1:
            timestamp, original, _ = payload[-1] if prefix == "pre" else payload[1]
            out[f"{prefix}_snapshot_timestamp"] = timestamp
            out[f"{prefix}_snapshot_url"] = f"https://web.archive.org/web/{timestamp}id_/{original}"
        else:
            out[f"{prefix}_snapshot_timestamp"] = None
            out[f"{prefix}_snapshot_url"] = None
    return out


async def fetch_text(client: httpx.AsyncClient, url: str | None, semaphore: asyncio.Semaphore) -> str:
    if not url:
        return ""
    async with semaphore:
        try:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return ""
            parser = TextExtractor()
            parser.feed(response.text)
            return parser.text()[:5000]
        except Exception:
            return ""


async def enrich_archives(rows: list[dict]) -> list[dict]:
    timeout = httpx.Timeout(90.0)
    headers = {"User-Agent": "Mozilla/5.0 (TimeSPEC academic provenance audit)"}
    lookup_sem = asyncio.Semaphore(6)
    fetch_sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        enriched = await asyncio.gather(*(archive_lookup(client, row, lookup_sem) for row in rows))
        tasks = []
        for row in enriched:
            tasks.append(fetch_text(client, row["pre_snapshot_url"], fetch_sem))
            tasks.append(fetch_text(client, row["post_snapshot_url"], fetch_sem))
        texts = await asyncio.gather(*tasks)
    for index, row in enumerate(enriched):
        row["pre_snapshot_text"] = texts[index * 2]
        row["post_snapshot_text"] = texts[index * 2 + 1]
    return enriched


SYSTEM = """You are auditing whether an exact piece of retrieved information respected a historical cutoff.
Return JSON only. Labels:
A = VERIFIED_PRE_CUTOFF: the exact information or a clear semantic equivalent is present in the pre-cutoff snapshot.
B = POST_CUTOFF_LEAKAGE: evidence establishes that the information appeared only after cutoff or explicitly describes a post-cutoff event.
C = UNVERIFIABLE: archives are absent/inadequate or do not establish either conclusion.
Do not infer leakage merely because one snapshot lacks a fact. Be conservative."""


async def auto_label(client: OpenRouterTextClient, row: dict, semaphore: asyncio.Semaphore) -> dict:
    prompt = f"""ITEM: {row['sample_id']}
CUTOFF: {row['cutoff_date']}
RETRIEVED TEXT:
{row['retrieved_text'][:1800]}

SOURCE URL: {row['source_url']}
PRE-CUTOFF SNAPSHOT: {row.get('pre_snapshot_timestamp')}
PRE-CUTOFF TEXT:
{row.get('pre_snapshot_text', '')[:2500]}

FIRST POST-CUTOFF SNAPSHOT: {row.get('post_snapshot_timestamp')}
POST-CUTOFF TEXT:
{row.get('post_snapshot_text', '')[:2500]}

Return {{"label":"A|B|C","confidence":"H|M|L","reason":"one sentence","evidence_date":"YYYY-MM-DD or null"}}."""
    async with semaphore:
        text = await client.call_text(prompt, system=SYSTEM)
    match = re.search(r"\{[\s\S]*\}", text)
    try:
        result = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        result = {}
    label = str(result.get("label", "C")).upper()
    if label not in {"A", "B", "C"}:
        label = "C"
    confidence = str(result.get("confidence", "L")).upper()
    if confidence not in {"H", "M", "L"}:
        confidence = "L"
    # The audit's A label is evidentiary, not inferential: without a usable
    # pre-cutoff archive, explicit pre-cutoff availability is not verified.
    if label == "A" and not row.get("pre_snapshot_text"):
        label = "C"
        confidence = "M"
        result["reason"] = (
            "No usable pre-cutoff archive was available; the retrieved text may "
            "describe an earlier event, but content-level availability is unverified."
        )
    return {
        **row,
        "auto_label": label,
        "auto_confidence": confidence,
        "auto_reason": str(result.get("reason", "Automatic audit could not establish provenance.")),
        "auto_evidence_date": result.get("evidence_date"),
    }


async def label_all(rows: list[dict]) -> list[dict]:
    client = OpenRouterTextClient(
        OpenRouterConfig(model="google/gemini-3.1-flash-lite", temperature=0.0, max_tokens=512)
    )
    semaphore = asyncio.Semaphore(6)
    return await asyncio.gather(*(auto_label(client, row, semaphore) for row in rows))


def main() -> None:
    rows = [json.loads(line) for line in INPUT.open(encoding="utf-8") if line.strip()]
    enriched = asyncio.run(enrich_archives(rows))
    labeled = asyncio.run(label_all(enriched))
    output_jsonl = OUTPUT / "auto_audit_results.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "sample_id", "cutoff_date", "retrieved_text", "source_url",
        "pre_snapshot_url", "post_snapshot_url",
        "auto_label", "auto_confidence", "auto_reason",
        "human_confirm", "human_corrected_label", "human_note",
    ]
    with (OUTPUT / "human_verification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in labeled:
            writer.writerow({key: row.get(key, "") for key in fields})

    counts = Counter(row["auto_label"] for row in labeled)
    high = Counter(row["auto_label"] for row in labeled if row["auto_confidence"] == "H")
    summary = {
        "n": len(labeled),
        "auto_counts": dict(counts),
        "high_confidence_counts": dict(high),
        "pre_snapshot_available": sum(bool(row["pre_snapshot_url"]) for row in labeled),
        "pre_snapshot_text_usable": sum(bool(row["pre_snapshot_text"]) for row in labeled),
        "human_verification_file": str(OUTPUT / "human_verification.csv"),
    }
    (OUTPUT / "auto_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
