"""B1: Temporal hint only — single generation call, no supervisor."""

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
    phase1_texts: list[str] | None = None,
    on_complete: Callable[[Completion], None] | None = None,
) -> list[Completion]:
    """Generate B1 completions.

    Args:
        instances: Task instances to process.
        cfg: Generation config.
        phase1_texts: Optional pre-generated raw texts (from shared Phase 1A).
                      If provided, skips generation.
        on_complete: Optional callback invoked with each completion as soon as
                     it is produced. Used by the runner for per-instance
                     crash-safe persistence.

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
        prompt_inputs = [build_generation_prompt(inst) for inst in instances]
        phase1_texts = generator.generate_batch(prompt_inputs, batch_size=cfg.batch_size, stage="phase1a_generation")

    completions: list[Completion] = []
    ts = datetime.now(timezone.utc).isoformat()

    for inst, raw_text in zip(instances, phase1_texts):
        parsed = parse_completion(inst.answer_type, raw_text)
        if parsed is None:
            logger.warning("B1 parse failure for %s", inst.instance_id)
            c = _failure_completion(inst, raw_text, ts)
        else:
            c = _build_completion(inst, parsed, raw_text, ts)
        completions.append(c)
        if on_complete is not None:
            on_complete(c)

    return completions


def _build_completion(
    inst: ForecastingTaskInstance,
    parsed: dict,
    raw_text: str,
    ts: str,
) -> Completion:
    return Completion(
        instance_id=inst.instance_id,
        method="b1",
        model="",  # filled by runner from cfg.model_name
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


def _failure_completion(
    inst: ForecastingTaskInstance,
    raw_text: str,
    ts: str,
) -> Completion:
    return Completion(
        instance_id=inst.instance_id,
        method="b1",
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
