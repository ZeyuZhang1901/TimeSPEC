"""Low-level Anthropic Message Batches API client.

Wraps the SDK's Message Batches API with:
  - submit:     send a list of (custom_id, messages, model, max_tokens, system) requests.
  - poll:       block until processing_status == "ended".
  - fetch:      stream results back as (custom_id, text, error) tuples.

A "batch" here = up to 100,000 message requests in one HTTP submission.
Anthropic processes the batch asynchronously (usually <1h, max 24h) and
gives a 50% discount on all per-token costs.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Iterator

from anthropic import Anthropic, APIConnectionError, InternalServerError, RateLimitError
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from dotenv import load_dotenv

log = logging.getLogger(__name__)

MAX_BATCH_REQUESTS = 80_000

# Retry budget for batch submission.
# Anthropic enforces a workspace-level cap on total in-flight Message Batch
# requests. When we hit it, the only remedy is to wait for previously-submitted
# batches to drain. Exponential backoff with a 30-min cap lets us wait several
# hours total without giving up.
_SUBMIT_MAX_RETRIES = 24
_SUBMIT_BASE_DELAY_SEC = 60.0
_SUBMIT_MAX_DELAY_SEC = 1800.0
_TRANSIENT_MAX_RETRIES = 6
_TRANSIENT_BASE_DELAY_SEC = 5.0


@dataclass(frozen=True)
class BatchRequest:
    """One message request inside a batch."""

    custom_id: str
    system: str
    user: str
    model: str
    max_tokens: int
    temperature: float = 0.0


@dataclass(frozen=True)
class BatchResult:
    """One result returned from a batch."""

    custom_id: str
    text: str | None
    error: str | None
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicBatchClient:
    """Submit/poll/fetch wrapper around Anthropic Message Batches API."""

    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv(override=False)
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("Missing ANTHROPIC_API_KEY for AnthropicBatchClient.")
        self._client = Anthropic(api_key=key)

    def submit(self, requests: list[BatchRequest]) -> str:
        """Submit a batch; return its batch_id.

        Splits internally if > MAX_BATCH_REQUESTS — caller should chunk if needed.
        """
        if len(requests) == 0:
            raise ValueError("Cannot submit an empty batch.")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise ValueError(
                f"Batch size {len(requests)} exceeds max {MAX_BATCH_REQUESTS}. "
                "Caller must split into multiple batches."
            )

        api_requests: list[Request] = []
        for req in requests:
            params = MessageCreateParamsNonStreaming(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system,
                messages=[{"role": "user", "content": req.user}],
                temperature=req.temperature,
            )
            api_requests.append(Request(custom_id=req.custom_id, params=params))

        batch_id = self._create_with_retry(api_requests, request_count=len(requests))
        log.info("Submitted batch %s with %d requests", batch_id, len(requests))
        return batch_id

    def _create_with_retry(self, api_requests: list[Request], *, request_count: int) -> str:
        """Create a batch, retrying on workspace rate-limit and transient errors."""
        for attempt in range(1, _SUBMIT_MAX_RETRIES + 1):
            try:
                batch = self._client.messages.batches.create(requests=api_requests)
                return batch.id
            except RateLimitError as exc:
                if attempt == _SUBMIT_MAX_RETRIES:
                    log.error(
                        "Batch submit gave up after %d attempts (still rate-limited): %s",
                        attempt, exc,
                    )
                    raise
                delay = min(
                    _SUBMIT_BASE_DELAY_SEC * (2 ** (attempt - 1)),
                    _SUBMIT_MAX_DELAY_SEC,
                ) + random.uniform(0.0, 15.0)
                log.warning(
                    "Batch submit (%d requests) hit workspace rate limit "
                    "(attempt %d/%d); sleeping %.0fs before retry: %s",
                    request_count, attempt, _SUBMIT_MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            except (APIConnectionError, InternalServerError) as exc:
                if attempt > _TRANSIENT_MAX_RETRIES:
                    raise
                delay = _TRANSIENT_BASE_DELAY_SEC * (2 ** (attempt - 1)) + random.uniform(0.0, 2.0)
                log.warning(
                    "Batch submit transient error (attempt %d): sleeping %.1fs: %s",
                    attempt, delay, exc,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable: _create_with_retry loop exited without returning")

    def poll(self, batch_id: str, *, interval_sec: float = 30.0, max_wait_sec: float = 26 * 3600) -> str:
        """Poll until the batch is ``ended``; return final processing_status.

        Raises TimeoutError after ``max_wait_sec`` (default 26h, slightly above
        Anthropic's 24h expiry to allow detection).
        """
        start = time.time()
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts
            log.info(
                "Batch %s status=%s processing=%d succeeded=%d errored=%d canceled=%d expired=%d",
                batch_id, status,
                counts.processing, counts.succeeded, counts.errored, counts.canceled, counts.expired,
            )
            if status == "ended":
                return status
            if time.time() - start > max_wait_sec:
                raise TimeoutError(f"Batch {batch_id} did not finish in {max_wait_sec}s (last status: {status}).")
            time.sleep(interval_sec)

    def fetch_results(self, batch_id: str) -> Iterator[BatchResult]:
        """Stream results from a finished batch."""
        for result in self._client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            inner = result.result
            if inner.type == "succeeded":
                msg = inner.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                in_tok = msg.usage.input_tokens if msg.usage else 0
                out_tok = msg.usage.output_tokens if msg.usage else 0
                yield BatchResult(custom_id=custom_id, text=text, error=None,
                                   input_tokens=in_tok, output_tokens=out_tok)
            elif inner.type == "errored":
                err = inner.error
                err_msg = f"{err.type}: {getattr(err, 'message', '')}" if hasattr(err, 'type') else str(err)
                yield BatchResult(custom_id=custom_id, text=None, error=err_msg)
            elif inner.type == "canceled":
                yield BatchResult(custom_id=custom_id, text=None, error="canceled")
            elif inner.type == "expired":
                yield BatchResult(custom_id=custom_id, text=None, error="expired")
            else:
                yield BatchResult(custom_id=custom_id, text=None, error=f"unknown result type: {inner.type}")
