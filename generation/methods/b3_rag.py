"""B3: Temporal hint + RAG context (no supervisor)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from LeakageEvaluationFramework.generation._openrouter_gen import OpenRouterGenerator
from LeakageEvaluationFramework.generation.parsers import parse_completion
from LeakageEvaluationFramework.generation.prompts import build_generation_prompt
from LeakageEvaluationFramework.generation.schemas import Completion, ForecastingTaskInstance, GenerationConfig

logger = logging.getLogger(__name__)


def generate(
    instances: list[ForecastingTaskInstance],
    cfg: GenerationConfig,
    context_map: dict[str, str],
    phase1_texts: list[str] | None = None,
    on_complete: Callable[[Completion], None] | None = None,
) -> list[Completion]:
    """Generate B3 completions: temporal hint + RAG context, no supervisor.

    Args:
        instances: Task instances to process.
        cfg: Generation config.
        context_map: Mapping from instance_id to pre-built RAG context block.
        phase1_texts: Optional pre-generated Phase 1B texts (shared with B4).
        on_complete: Optional callback invoked per completion for crash-safe save.

    Returns:
        One Completion per instance (order-preserving).
    """

    if phase1_texts is None:
        generator = OpenRouterGenerator(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=cfg.openrouter_api_key,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
        )
        prompt_inputs = [
            build_generation_prompt(inst, context_block=context_map.get(inst.instance_id, ""))
            for inst in instances
        ]
        phase1_texts = generator.generate_batch(prompt_inputs, batch_size=cfg.batch_size)

    ts = datetime.now(timezone.utc).isoformat()
    completions: list[Completion] = []

    for inst, raw_text in zip(instances, phase1_texts):
        parsed = parse_completion(inst.answer_type, raw_text)
        if parsed is None:
            logger.warning("B3 parse failure for %s", inst.instance_id)
            c = _failure_completion(inst, raw_text, ts)
            completions.append(c)
            if on_complete is not None:
                on_complete(c)
            continue
        c = Completion(
            instance_id=inst.instance_id,
            method="b3",
            model="",
            answer_type=inst.answer_type,
            evidence=parsed.get("evidence", []),
            prediction=parsed.get("prediction", ""),
            reasoning=parsed.get("reasoning", ""),
            probability=parsed.get("probability_yes"),
            probabilities=parsed.get("probabilities"),
            raw_text=raw_text,
            parse_success=True,
            regen_cycles=0,
            leaked_claim_ids=[],
            timestamp=ts,
            ranking=parsed.get("ranking"),
            probability_petitioner=parsed.get("probability_petitioner"),
            predicted_salary=parsed.get("predicted_salary"),
        )
        completions.append(c)
        if on_complete is not None:
            on_complete(c)

    return completions


def _failure_completion(
    inst: ForecastingTaskInstance,
    raw_text: str,
    ts: str,
) -> Completion:
    return Completion(
        instance_id=inst.instance_id,
        method="b3",
        model="",
        answer_type=inst.answer_type,
        evidence=[],
        prediction="",
        reasoning="",
        probability=None,
        probabilities=None,
        raw_text=raw_text,
        parse_success=False,
        regen_cycles=0,
        leaked_claim_ids=[],
        timestamp=ts,
    )
