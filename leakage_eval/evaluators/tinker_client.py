"""Tinker-backed coalition predictor."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from LeakageEvaluationFramework.leakage_eval.evaluators.parsers import parse_prediction
from LeakageEvaluationFramework.leakage_eval.schemas import EvalInput, EvidenceItem
from LeakageEvaluationFramework.leakage_eval.shapley.prompts import build_single_coalition_prompt


@dataclass(frozen=True)
class TinkerSamplingConfig:
    """Tinker sampling settings."""

    model_name: str
    renderer_name: str | None = None
    temperature: float = 0.0
    max_tokens: int = 64
    model_path: str | None = None


class TinkerCoalitionPredictor:
    """Closed-world coalition predictor using Tinker.

    The Tinker import and client creation are lazy so local tests can run
    without Tinker credentials.
    """

    def __init__(self, config: TinkerSamplingConfig) -> None:
        self.config = config
        self._sampling_client: Any | None = None
        self._renderer: Any | None = None
        self._sampling_params: Any | None = None
        self.last_raw_text: str | None = None

    async def predict(self, instance: EvalInput, evidence_subset: list[EvidenceItem]) -> dict:
        """Predict from one closed-world coalition."""

        prompt = build_single_coalition_prompt(instance, evidence_subset)
        text = await self._sample_text(prompt)
        self.last_raw_text = text
        try:
            return parse_prediction(instance.task, text)
        except ValueError:
            if instance.task == "stock":
                return _repair_stock_prediction(instance, text)
            raise

    async def _sample_text(self, prompt: str) -> str:
        return await asyncio.to_thread(self._sample_text_sync, prompt)

    def _sample_text_sync(self, prompt: str) -> str:
        self._ensure_client()
        assert self._sampling_client is not None
        assert self._renderer is not None
        assert self._sampling_params is not None

        messages = [{"role": "user", "content": prompt}]
        model_input = self._renderer.build_generation_prompt(messages)
        result = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._sampling_params,
        ).result()
        parsed_msg, _ = self._renderer.parse_response(result.sequences[0].tokens)

        from tinker_cookbook import renderers as renderer_utils

        return renderer_utils.get_text_content(parsed_msg)

    def _ensure_client(self) -> None:
        if self._sampling_client is not None:
            return

        load_dotenv(override=False)
        tinker_key = os.getenv("TINKER_API_KEY", "").strip()
        if not tinker_key:
            raise ValueError("Missing TINKER_API_KEY for TinkerCoalitionPredictor.")
        os.environ["TINKER_API_KEY"] = tinker_key

        import tinker
        from tinker import types as tinker_types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        service_client = tinker.ServiceClient()
        if self.config.model_path:
            self._sampling_client = service_client.create_sampling_client(model_path=self.config.model_path)
        else:
            self._sampling_client = service_client.create_sampling_client(base_model=self.config.model_name)

        tokenizer = get_tokenizer(self.config.model_name)
        renderer_name = self.config.renderer_name or model_info.get_recommended_renderer_name(self.config.model_name)
        self._renderer = renderers.get_renderer(renderer_name, tokenizer)
        self._sampling_params = tinker_types.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self._renderer.get_stop_sequences(),
        )


def _repair_stock_prediction(instance: EvalInput, text: str) -> dict:
    allowed = _allowed_stock_tickers(instance)
    if len(allowed) < 2:
        raise ValueError("Cannot repair stock prediction without at least two allowed tickers")

    ranking = _extract_ranking_like_list(text)
    repaired: list[str] = []
    for item in ranking:
        ticker = _to_allowed_ticker(str(item), allowed)
        if ticker is not None and ticker not in repaired:
            repaired.append(ticker)
    for ticker in allowed:
        if ticker not in repaired:
            repaired.append(ticker)
    if len(repaired) < 2:
        raise ValueError("Stock prediction repair failed to recover ranking")
    return {"ranking": repaired}


def _allowed_stock_tickers(instance: EvalInput) -> list[str]:
    meta = instance.raw.get("dataset_metadata")
    if not isinstance(meta, dict):
        return []
    source = meta.get("prompt_safe") if isinstance(meta.get("prompt_safe"), dict) else meta
    companies = source.get("companies") if isinstance(source, dict) else []
    return [str(ticker) for ticker in companies] if isinstance(companies, list) else []


def _extract_ranking_like_list(text: str) -> list[Any]:
    payload = _extract_json_like(text)
    if isinstance(payload, dict):
        for key in ("ranking", "rankings", "ranked_tickers", "tickers", "stocks"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for value in payload.values():
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return re.findall(r"\b[A-Z]{1,5}\b", text)


def _extract_json_like(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", stripped)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _to_allowed_ticker(value: str, allowed: list[str]) -> str | None:
    normalized = _normalize_token(value)
    allowed_by_norm = {_normalize_token(ticker): ticker for ticker in allowed}
    if normalized in allowed_by_norm:
        return allowed_by_norm[normalized]

    meta = value.upper()
    for ticker in allowed:
        if ticker.upper() == meta:
            return ticker
    return None


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())

