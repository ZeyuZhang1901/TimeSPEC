"""OpenRouter-backed text client for judge calls and coalition prediction."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from openai import OpenAI

from LeakageEvaluationFramework import usage_log

if TYPE_CHECKING:
    from LeakageEvaluationFramework.leakage_eval.schemas import EvalInput, EvidenceItem

_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0
_MAX_RETRIES_RATE_LIMIT = 8
_RATE_LIMIT_BASE_DELAY = 5.0


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate limit" in str(exc).lower()


def _retry_delay(attempt: int, exc: Exception) -> float:
    if _is_rate_limit(exc):
        import random
        return _RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
    return _RETRY_DELAY_SEC * attempt


def _max_attempts(exc: Exception) -> int:
    return _MAX_RETRIES_RATE_LIMIT if _is_rate_limit(exc) else _MAX_RETRIES


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter model settings."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 64


@dataclass(frozen=True)
class UsageStat:
    """Token usage returned by OpenRouter."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenRouterTextClient:
    """Thin async wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(self, config: OpenRouterConfig, api_key: str | None = None) -> None:
        self.config = config
        load_dotenv(override=False)
        key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("Missing OPENROUTER_API_KEY for OpenRouterTextClient.")
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self.last_usage: UsageStat | None = None

    async def call_text(
        self, prompt: str, system: str | None = None, stage: str = "judge"
    ) -> str:
        """Call a text model."""

        text, usage = await asyncio.to_thread(self._call_text_sync, prompt, system, stage)
        self.last_usage = usage
        return text

    def _call_text_sync(
        self, prompt: str, system: str | None, stage: str = "judge"
    ) -> tuple[str, UsageStat | None]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # Most reasoning models should suppress hidden reasoning for short JSON
        # judgments. Gemini 3.1 Pro, however, requires reasoning and rejects
        # effort="none", so allow its provider default.
        if "gemini-3.1-pro" not in self.config.model:
            params["extra_body"] = {"reasoning": {"effort": "none"}}

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES_RATE_LIMIT + 1):
            try:
                started = time.perf_counter()
                resp = self._client.chat.completions.create(**params)
                usage_log.record_response(
                    component="judge_client",
                    stage=stage,
                    model=self.config.model,
                    response=resp,
                    latency_s=time.perf_counter() - started,
                )
                content = resp.choices[0].message.content or ""
                usage = None
                if resp.usage is not None:
                    usage = UsageStat(
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens,
                        total_tokens=resp.usage.total_tokens,
                    )
                return content, usage
            except Exception as exc:  # provider/network failures are retried.
                last_error = exc
                if attempt < _max_attempts(exc):
                    time.sleep(_retry_delay(attempt, exc))
                elif attempt >= _max_attempts(exc):
                    break
        raise RuntimeError(f"OpenRouter request failed after {attempt} attempts: {last_error}") from last_error


class OpenRouterCoalitionPredictor:
    """Closed-world coalition predictor using OpenRouter.

    Satisfies the CoalitionPredictor protocol from value_function.py.
    Uses split system/user messages so that OpenRouter can cache the
    system message (stable instruction + question) across coalition calls
    for the same instance.
    """

    _PREDICT_RETRIES = 2

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        api_key: str | None = None,
    ) -> None:
        load_dotenv(override=False)
        key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("Missing OPENROUTER_API_KEY for OpenRouterCoalitionPredictor.")
        self._openai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_raw_text: str | None = None
        self.last_usage: UsageStat | None = None

    async def predict(
        self,
        instance: "EvalInput",
        evidence_subset: list["EvidenceItem"],
    ) -> dict:
        from LeakageEvaluationFramework.leakage_eval.evaluators.parsers import parse_prediction
        from LeakageEvaluationFramework.leakage_eval.shapley.prompts import build_coalition_messages

        system_msg, user_msg = build_coalition_messages(instance, evidence_subset)
        last_exc: Exception | None = None
        for attempt in range(1, self._PREDICT_RETRIES + 1):
            text = await asyncio.to_thread(
                self._call_sync, system_msg, user_msg
            )
            self.last_raw_text = text
            try:
                return parse_prediction(instance.task, text)
            except ValueError as exc:
                last_exc = exc
                if attempt < self._PREDICT_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY_SEC)
        raise last_exc  # type: ignore[misc]

    def _call_sync(self, system: str, user: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user},
        ]
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "extra_body": {"reasoning": {"effort": "none"}},
        }
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES_RATE_LIMIT + 1):
            try:
                started = time.perf_counter()
                resp = self._openai.chat.completions.create(**params)
                usage_log.record_response(
                    component="coalition_predictor",
                    stage="shapley_coalition",
                    model=self._model,
                    response=resp,
                    latency_s=time.perf_counter() - started,
                )
                content = resp.choices[0].message.content or ""
                if resp.usage is not None:
                    self.last_usage = UsageStat(
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens,
                        total_tokens=resp.usage.total_tokens,
                    )
                return content
            except Exception as exc:
                last_error = exc
                if attempt < _max_attempts(exc):
                    time.sleep(_retry_delay(attempt, exc))
                elif attempt >= _max_attempts(exc):
                    break
        raise RuntimeError(
            f"Coalition prediction failed after {attempt} attempts: {last_error}"
        ) from last_error

