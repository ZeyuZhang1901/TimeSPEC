"""Minimal Perplexity retrieval adapter used by the rebuttal pilot.

The original experiment code imported this functionality from a sibling project
that is not included in the released repository. This module provides the same
small interface so the public release is runnable in isolation.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from openai import OpenAI

from LeakageEvaluationFramework import usage_log


class PerplexitySearchClient:
    def __init__(self, api_key: str, model: str = "sonar") -> None:
        if not api_key:
            raise ValueError("Missing PERPLEXITY_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        self.model = model
        self.call_count = 0

    def search(
        self,
        queries: str | list[str],
        *,
        cutoff_date: str | None = None,
        max_results: int = 5,
        stage: str = "search",
    ) -> list[dict[str, Any]]:
        query_list = [queries] if isinstance(queries, str) else list(queries)
        results: list[dict[str, Any]] = []
        for query in query_list:
            prompt = query
            if cutoff_date:
                prompt += (
                    f"\nOnly use information publicly available before {cutoff_date}. "
                    "Include source dates and avoid later summaries."
                )
            extra_body: dict[str, Any] = {}
            if cutoff_date:
                parsed = date.fromisoformat(cutoff_date)
                extra_body["search_before_date_filter"] = parsed.strftime("%m/%d/%Y")
            started = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
                extra_body=extra_body or None,
            )
            usage_log.record_response(
                component="perplexity",
                stage=stage,
                model=self.model,
                response=response,
                latency_s=time.perf_counter() - started,
            )
            self.call_count += 1
            content = response.choices[0].message.content or ""
            citations = getattr(response, "citations", None)
            if citations is None:
                citations = getattr(response, "model_extra", {}).get("citations", [])
            citations = list(citations or [])
            results.append(
                {
                    "title": query,
                    "date": cutoff_date or "unknown",
                    "snippet": content,
                    "url": citations[0] if citations else "",
                    "citations": citations,
                }
            )
            if len(results) >= max_results:
                break
        return results


def retrieve_and_build_context(
    search_client: PerplexitySearchClient,
    queries: list[str],
    *,
    cutoff_date: str,
    budget: int,
    max_results_per_query: int,
) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    all_results: list[dict[str, Any]] = []
    citations: list[str] = []
    for query in queries:
        query_results = search_client.search(
            query,
            cutoff_date=cutoff_date,
            max_results=max_results_per_query,
            stage="rag_retrieval",
        )
        all_results.extend(query_results)
        for result in query_results:
            citations.extend(result.get("citations") or [])
    blocks = []
    for index, result in enumerate(all_results, 1):
        blocks.append(
            f"[{index}] Query: {result['title']}\n"
            f"Cutoff-filtered result: {result['snippet']}\n"
            f"Source: {result.get('url', '')}"
        )
    context = "\n\n".join(blocks)
    if len(context) > budget:
        context = context[:budget]
    return context, all_results, queries, citations
