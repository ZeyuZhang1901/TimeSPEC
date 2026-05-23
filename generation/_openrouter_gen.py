"""OpenRouter-backed text generator for B1–B4 generation methods.

Thread-safe: the underlying OpenAI client is thread-safe by design.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0


class OpenRouterGenerator:
    """Thin wrapper around OpenRouter's OpenAI-compatible API for generation.

    Accepts pre-built (system, user) message tuples and returns raw text.
    Drop-in replacement for TinkerGenerator.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        concurrency: int = 1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._concurrency = concurrency
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        load_dotenv(override=False)
        key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("Missing OPENROUTER_API_KEY for OpenRouterGenerator.")
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def generate_one(self, system: str, user: str) -> str:
        """Generate a single completion."""
        return self._call_sync(system, user)

    def generate_batch(
        self,
        inputs: list[tuple[str, str]],
        batch_size: int = 8,
    ) -> list[str]:
        """Generate completions for a list of (system, user) pairs.

        Uses ThreadPoolExecutor with ``self._concurrency`` workers.
        Falls back to sequential processing when concurrency is 1.
        """
        n = len(inputs)
        if self._concurrency <= 1:
            return self._generate_batch_sequential(inputs, batch_size)

        completions: list[str | None] = [None] * n
        done = 0
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            future_to_idx = {
                pool.submit(self._call_sync, sys_msg, usr_msg): i
                for i, (sys_msg, usr_msg) in enumerate(inputs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completions[idx] = future.result()
                done += 1
                if done % batch_size == 0 or done == n:
                    logger.info("Generated %d / %d completions", done, n)
        return completions  # type: ignore[return-value]

    def _generate_batch_sequential(
        self,
        inputs: list[tuple[str, str]],
        batch_size: int,
    ) -> list[str]:
        completions: list[str] = []
        for i, (sys_msg, usr_msg) in enumerate(inputs):
            completions.append(self._call_sync(sys_msg, usr_msg))
            if (i + 1) % batch_size == 0:
                logger.info("Generated %d / %d completions", i + 1, len(inputs))
        return completions

    def _call_sync(self, system: str, user: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "extra_body": {"reasoning": {"effort": "none"}},
        }
        if self._frequency_penalty:
            params["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty:
            params["presence_penalty"] = self._presence_penalty

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(**params)
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SEC * attempt)
        raise RuntimeError(
            f"OpenRouter generation failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error
