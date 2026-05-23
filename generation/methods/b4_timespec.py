"""B4: Full TimeSPEC — temporal hint + RAG + C0-C8 supervisor with search escalation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable

from LeakageEvaluationFramework.generation._openrouter_gen import OpenRouterGenerator
from LeakageEvaluationFramework.generation.parsers import parse_completion
from LeakageEvaluationFramework.generation.prompts import (
    build_aggregator_prompt,
    build_generation_prompt,
    build_regeneration_prompt,
)
from LeakageEvaluationFramework.generation.schemas import Completion, ForecastingTaskInstance, GenerationConfig
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)

from ._supervisor import leaked_claim_ids, run_search_supervisor

logger = logging.getLogger(__name__)


def generate(
    instances: list[ForecastingTaskInstance],
    cfg: GenerationConfig,
    context_map: dict[str, str],
    search_client: Any,
    phase1_texts: list[str] | None = None,
    concurrency: int = 1,
    on_complete: Callable[[Completion], None] | None = None,
) -> list[Completion]:
    """Generate B4 completions: temporal hint + RAG + C0-C8 supervisor with search.

    Args:
        instances: Task instances to process.
        cfg: Generation config.
        context_map: Mapping from instance_id to pre-built RAG context block.
        search_client: PerplexitySearchClient for supervisor search escalation.
        phase1_texts: Optional pre-generated Phase 1B texts (shared with B3).
        concurrency: Number of instances to process in parallel (supervisor calls).
        on_complete: Optional callback invoked per completion for crash-safe save.

    Returns:
        One Completion per instance (order-preserving).
    """

    generator = OpenRouterGenerator(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=cfg.openrouter_api_key,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
    )
    or_client = OpenRouterTextClient(
        OpenRouterConfig(model=cfg.judge_model, max_tokens=cfg.judge_max_tokens),
        api_key=cfg.openrouter_api_key,
    )

    if phase1_texts is None:
        prompt_inputs = [
            build_generation_prompt(inst, context_block=context_map.get(inst.instance_id, ""))
            for inst in instances
        ]
        phase1_texts = generator.generate_batch(prompt_inputs, batch_size=cfg.batch_size)

    ts = datetime.now(timezone.utc).isoformat()

    if concurrency <= 1:
        out: list[Completion] = []
        for inst, p1_text in zip(instances, phase1_texts):
            c = _run_instance(
                inst, p1_text, context_map.get(inst.instance_id, ""),
                cfg, generator, or_client, search_client, ts,
            )
            out.append(c)
            if on_complete is not None:
                on_complete(c)
        return out

    results: list[Completion | None] = [None] * len(instances)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(
                _run_instance,
                inst, p1_text, context_map.get(inst.instance_id, ""),
                cfg, generator, or_client, search_client, ts,
            ): i
            for i, (inst, p1_text) in enumerate(zip(instances, phase1_texts))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                c = future.result()
            except Exception as exc:
                inst = instances[idx]
                logger.error("B4 instance %s failed: %s", inst.instance_id, exc)
                c = _failure_completion(inst, "", 0, [], ts)
            results[idx] = c
            if on_complete is not None:
                on_complete(c)

    return results  # type: ignore[return-value]


def _run_instance(
    inst: ForecastingTaskInstance,
    phase1_text: str,
    context_block: str,
    cfg: GenerationConfig,
    generator: OpenRouterGenerator,
    or_client: OpenRouterTextClient,
    search_client: Any,
    ts: str,
) -> Completion:
    current_text = phase1_text
    total_leaked_ids: list[int] = []
    all_validated_evidence: list[dict] = []
    cycles_used = 0
    last_leaked_ids: list[int] = []  # leakage flags for the LAST attempted evidence list

    for cycle in range(cfg.max_regen_cycles + 1):
        parsed = parse_completion(inst.answer_type, current_text)
        if parsed is None:
            logger.warning("B4 parse failure (cycle %d) for %s", cycle, inst.instance_id)
            return _failure_completion(inst, current_text, cycles_used, total_leaked_ids, ts)

        evidence = parsed.get("evidence", [])
        decisions = run_search_supervisor(
            instance_id=inst.instance_id,
            evidence=evidence,
            temporal_cutoff=inst.temporal_cutoff,
            answer_type=inst.answer_type,
            question=inst.question,
            model=cfg.model_name,
            method="b4",
            or_client=or_client,
            search_client=search_client,
            judge_model=cfg.judge_model,
        )
        this_leaked = leaked_claim_ids(decisions)
        total_leaked_ids = list(dict.fromkeys(total_leaked_ids + this_leaked))
        last_leaked_ids = this_leaked  # remember the LAST cycle's flags for the final filter

        # Collect validated (non-leaked) evidence from this trial
        leaked_set = set(this_leaked)
        validated = [ev for ev in evidence if ev.get("id") not in leaked_set]
        all_validated_evidence.extend(validated)

        if not this_leaked or cycle >= cfg.max_regen_cycles:
            break

        cycles_used += 1
        regen_sys, regen_usr = build_regeneration_prompt(
            inst,
            context_block=context_block,
            leaked_claim_ids=this_leaked,
            previous_evidence=evidence,
        )
        current_text = generator.generate_one(regen_sys, regen_usr)
        logger.info(
            "B4 regen cycle %d for %s: %d leaked claims",
            cycle + 1, inst.instance_id, len(this_leaked),
        )

    parsed_final = parse_completion(inst.answer_type, current_text)
    if parsed_final is None:
        return _failure_completion(inst, current_text, cycles_used, total_leaked_ids, ts)

    # Aggregation: closed-world generation from validated evidence across trials.
    # Skip only when the first trial was already clean (no leakage detected).
    if total_leaked_ids:
        seen_facts: set[str] = set()
        deduped_evidence: list[dict] = []
        for ev in all_validated_evidence:
            fact = (ev.get("fact") or "").strip().lower()
            if fact and fact not in seen_facts:
                seen_facts.add(fact)
                deduped_evidence.append(ev)

        # ALWAYS run the aggregator when leakage was detected, even when
        # ``deduped_evidence`` is empty. Without this, a fully-leaked final
        # regen would be emitted verbatim because the previous ``if deduped_evidence``
        # guard silently skipped the aggregation step.
        agg_sys, agg_usr = build_aggregator_prompt(inst, deduped_evidence)
        try:
            agg_text = generator.generate_one(agg_sys, agg_usr)
            parsed_agg = parse_completion(inst.answer_type, agg_text)
            if parsed_agg is not None:
                parsed_final = parsed_agg
                current_text = agg_text
            else:
                logger.warning("B4 aggregation parse failure for %s", inst.instance_id)
        except Exception as exc:
            logger.warning("B4 aggregation failed for %s: %s", inst.instance_id, exc)

    # Defence-in-depth: even after aggregation, scrub the final evidence list
    # so no supervisor-flagged or date-leaked claim escapes. This protects
    # against (a) the aggregator skipping/failing and (b) the aggregator
    # hallucinating fresh post-cutoff facts (e.g. memorised stock prices).
    final_evidence = _scrub_evidence(
        parsed_final.get("evidence", []),
        cutoff=inst.temporal_cutoff,
        flagged_ids=set(last_leaked_ids),
    )

    return Completion(
        instance_id=inst.instance_id,
        method="b4",
        model="",
        answer_type=inst.answer_type,
        evidence=final_evidence,
        prediction=parsed_final.get("prediction", ""),
        reasoning=parsed_final.get("reasoning", ""),
        probability=parsed_final.get("probability_yes"),
        probabilities=parsed_final.get("probabilities"),
        raw_text=current_text,
        parse_success=True,
        regen_cycles=cycles_used,
        leaked_claim_ids=total_leaked_ids,
        timestamp=ts,
        ranking=parsed_final.get("ranking"),
        probability_petitioner=parsed_final.get("probability_petitioner"),
        predicted_salary=parsed_final.get("predicted_salary"),
    )


def _scrub_evidence(
    evidence: list[dict],
    *,
    cutoff: str,
    flagged_ids: set,
) -> list[dict]:
    """Strip post-cutoff or supervisor-flagged claims from the final evidence list.

    Drops a claim when either:
      - its ``id`` is in ``flagged_ids`` (the last supervisor pass marked it leaked); or
      - its ``source_date`` is strictly later than ``cutoff`` (defence against
        the aggregator hallucinating fresh post-cutoff facts).
    """
    cutoff = (cutoff or "").strip()
    cleaned: list[dict] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if ev.get("id") in flagged_ids:
            continue
        sd = (ev.get("source_date") or "").strip()
        if cutoff and sd and sd > cutoff:
            continue
        cleaned.append(ev)
    return cleaned


def _failure_completion(
    inst: ForecastingTaskInstance,
    raw_text: str,
    cycles_used: int,
    leaked_ids: list[int],
    ts: str,
) -> Completion:
    return Completion(
        instance_id=inst.instance_id,
        method="b4",
        model="",
        answer_type=inst.answer_type,
        evidence=[],
        prediction="",
        reasoning="",
        probability=None,
        probabilities=None,
        raw_text=raw_text,
        parse_success=False,
        regen_cycles=cycles_used,
        leaked_claim_ids=leaked_ids,
        timestamp=ts,
    )
