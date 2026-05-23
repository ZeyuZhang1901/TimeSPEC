"""Tinker-backed text generator for B0–B4 generation methods.

Lazy-initialises the Tinker client so module import works without credentials.
Thread-safe: multiple worker threads can call generate_one/generate_batch
concurrently once the client is initialized.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class TinkerGenerator:
    """Thin wrapper around the Tinker sampling client for generation.

    Accepts pre-built (system, user) message tuples and returns raw text.
    Thread-safe after the first call to _ensure_client().
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        renderer_name: str | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._renderer_name = renderer_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._sampling_client: Any | None = None
        self._renderer: Any | None = None
        self._sampling_params: Any | None = None
        self._init_lock = threading.Lock()

    def generate_one(self, system: str, user: str) -> str:
        """Generate a single completion."""

        self._ensure_client()
        model_input = self._build_input(system, user)
        result = self._sampling_client.sample(  # type: ignore[union-attr]
            prompt=model_input, num_samples=1, sampling_params=self._sampling_params,
        ).result()
        return self._decode(result)

    def generate_batch(self, inputs: list[tuple[str, str]], batch_size: int = 8) -> list[str]:
        """Generate completions for a list of (system, user) pairs.

        Submits futures in groups of batch_size to avoid overwhelming Tinker.
        """

        self._ensure_client()
        completions: list[str] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            futures = [
                self._sampling_client.sample(  # type: ignore[union-attr]
                    prompt=self._build_input(sys, usr),
                    num_samples=1,
                    sampling_params=self._sampling_params,
                )
                for sys, usr in batch
            ]
            for future in futures:
                completions.append(self._decode(future.result()))
            logger.info("Generated %d / %d completions", len(completions), len(inputs))
        return completions

    def _build_input(self, system: str, user: str) -> Any:
        assert self._renderer is not None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._renderer.build_generation_prompt(messages)

    def _decode(self, result: Any) -> str:
        assert self._renderer is not None
        from tinker_cookbook import renderers
        parsed_msg, _ = self._renderer.parse_response(result.sequences[0].tokens)
        return renderers.get_text_content(parsed_msg)

    def _ensure_client(self) -> None:
        if self._sampling_client is not None:
            return
        with self._init_lock:
            # Double-checked locking: another thread may have initialized while we waited.
            if self._sampling_client is not None:
                return

        import tinker
        from tinker import types as tinker_types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        service_client = tinker.ServiceClient()
        if self._checkpoint_path:
            self._sampling_client = service_client.create_sampling_client(
                model_path=self._checkpoint_path
            )
        else:
            self._sampling_client = service_client.create_sampling_client(
                base_model=self._model_name
            )

        tokenizer = get_tokenizer(self._model_name)
        renderer_name = self._renderer_name or model_info.get_recommended_renderer_name(
            self._model_name
        )
        self._renderer = renderers.get_renderer(renderer_name, tokenizer)
        self._sampling_params = tinker_types.SamplingParams(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._renderer.get_stop_sequences(),
        )
