"""Resumable Wayback crawl: feasibility + fetch for the camera-ready Snapshot-RAG run.

For each question, try up to --max-urls candidate URLs from background_source_urls.
For each candidate, find the latest pre-cutoff Wayback snapshot via the CDX API,
fetch it, and extract text. A question is usable once one candidate yields >= 500
chars of extracted text. Progress is checkpointed per question to crawl_progress.jsonl
so the job resumes after interruption. The final snapshot_contexts.json matches the
format consumed by cr_run_snapshot_rag.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import httpx


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


async def cdx_lookup(client: httpx.AsyncClient, url: str, cutoff: str) -> tuple[str, str] | None:
    """Return (timestamp, original_url) of the latest pre-cutoff snapshot, or None."""
    endpoint = (
        "https://web.archive.org/cdx/search/cdx?"
        f"url={quote(url, safe='')}&output=json&filter=statuscode%3A200"
        f"&to={cutoff.replace('-', '')}&limit=-1"
        "&fl=timestamp%2Coriginal%2Cstatuscode"
    )
    for attempt in range(3):
        try:
            response = await client.get(endpoint)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and len(payload) > 1:
                timestamp, original, _ = payload[-1]
                return timestamp, original
            return None
        except Exception:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    return None


async def fetch_snapshot(client: httpx.AsyncClient, timestamp: str, original: str) -> str:
    snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{original}"
    for attempt in range(3):
        try:
            response = await client.get(snapshot_url)
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            return parser.text()[:6000]
        except Exception:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    return ""


async def crawl_question(
    client: httpx.AsyncClient,
    row: dict,
    semaphore: asyncio.Semaphore,
    max_urls: int,
) -> dict:
    result = {
        "instance_id": row["id"],
        "cutoff": row["temporal_cutoff"],
        "usable": False,
        "text": "",
        "text_chars": 0,
        "snapshot_timestamp": None,
        "snapshot_url": None,
        "source_url": None,
        "candidates_tried": 0,
        "error": None,
    }
    candidates = [u for u in (row.get("background_source_urls") or []) if u.startswith("http")]
    if not candidates:
        result["error"] = "no_candidate_urls"
        return result
    async with semaphore:
        for url in candidates[:max_urls]:
            result["candidates_tried"] += 1
            hit = await cdx_lookup(client, url, row["temporal_cutoff"])
            if hit is None:
                continue
            timestamp, original = hit
            text = await fetch_snapshot(client, timestamp, original)
            if len(text) >= 500:
                result.update(
                    usable=True,
                    text=text,
                    text_chars=len(text),
                    snapshot_timestamp=timestamp,
                    snapshot_url=f"https://web.archive.org/web/{timestamp}id_/{original}",
                    source_url=original,
                )
                return result
        result["error"] = "no_usable_snapshot"
    return result


async def run(rows: list[dict], progress_path: Path, concurrency: int, max_urls: int) -> None:
    done: set[str] = set()
    if progress_path.exists():
        for line in progress_path.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    todo = [row for row in rows if row["id"] not in done]
    print(f"[crawl] {len(done)} done, {len(todo)} to go", flush=True)

    timeout = httpx.Timeout(90.0)
    headers = {"User-Agent": "Mozilla/5.0 (TimeSPEC academic validation)"}
    semaphore = asyncio.Semaphore(concurrency)
    completed = len(done)
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:

        async def worker(row: dict) -> None:
            nonlocal completed
            result = await crawl_question(client, row, semaphore, max_urls)
            async with lock:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 10 == 0:
                    print(f"[crawl] {completed}/{len(rows)} at {time.strftime('%H:%M:%S')}", flush=True)

        await asyncio.gather(*(worker(row) for row in todo))


def finalize(progress_path: Path, output_path: Path) -> None:
    contexts: dict[str, dict] = {}
    for line in progress_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            contexts[row["instance_id"]] = row
    usable = sum(bool(row["usable"]) for row in contexts.values())
    report = {
        "n_questions": len(contexts),
        "usable_contexts": usable,
        "coverage": usable / len(contexts) if contexts else 0.0,
        "contexts": contexts,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_questions", "usable_contexts", "coverage")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-urls", type=int, default=3)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "crawl_progress.jsonl"
    contexts_path = output_dir / "snapshot_contexts.json"

    rows = [json.loads(line) for line in Path(args.dataset).open(encoding="utf-8") if line.strip()]
    if not args.finalize_only:
        asyncio.run(run(rows, progress_path, args.concurrency, args.max_urls))
    finalize(progress_path, contexts_path)


if __name__ == "__main__":
    main()
