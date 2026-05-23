"""Answer-type-dispatched prompt builders for generation methods B0–B4."""

from __future__ import annotations

from .schemas import ForecastingTaskInstance

_SYSTEM_UNCONSTRAINED = """You are an expert forecaster. Your task is to answer a prediction question using the best evidence available to you.

EVIDENCE INTEGRITY: Evidence must be factually genuine. Do not fabricate numbers, dates, or events. Each evidence item must reflect a real, publicly known fact. If you are uncertain about a specific value, state the fact qualitatively rather than inventing a precise figure.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific — not an opinion or prediction.
- For EACH fact, include a source_date field: the YYYY-MM-DD date when this information was first publicly known.
- Write each fact as a plain statement. Do NOT include citation markers like [1] in the evidence text.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
- You MUST cite every evidence item at least once.
- Do NOT introduce any new facts not listed in your evidence.

STEP 3 — PREDICTION:
Give your final prediction based solely on the reasoning above.
CRITICAL: You MUST always output the JSON object below, even under uncertainty. If you cannot be confident, provide your best-estimate value and reflect the uncertainty in the reasoning field. Never end your response without the JSON.
{output_spec}"""


_SYSTEM_PREAMBLE = """You are an expert forecaster. Your task is to answer a prediction question using only information that was publicly available before the specified knowledge cutoff date.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}. Do NOT use knowledge of events, announcements, or developments from on or after {cutoff_date}. If uncertain whether a piece of information predates the cutoff, omit it.

EVIDENCE INTEGRITY:
Evidence must be factually genuine. Do not fabricate numbers, dates, or events. Each evidence item must reflect a real, publicly known fact. If you are uncertain about a specific value, state the fact qualitatively rather than inventing a precise figure.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific — not an opinion or prediction.
- Each fact must have been publicly known before {cutoff_date}.
- For EACH fact, include a source_date field: the YYYY-MM-DD date when this information was FIRST PUBLICLY ANNOUNCED. This must be the announcement date, not the period-end date.
- Write each fact as a plain statement. Do NOT include citation markers like [1] in the evidence text.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
- You MUST cite every evidence item at least once.
- Do NOT introduce any new facts not listed in your evidence.

STEP 3 — PREDICTION:
Give your final prediction based solely on the reasoning above.
CRITICAL: You MUST always output the JSON object below, even under uncertainty. If you cannot be confident, provide your best-estimate value and reflect the uncertainty in the reasoning field. Never end your response without the JSON.
{output_spec}"""


_BINARY_OUTPUT_SPEC = """
OUTPUT FORMAT — a single JSON object, no markdown fences:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "probability_yes": <float 0.0–1.0>,
  "prediction": "<yes or no>"
}}"""


_MC_OUTPUT_SPEC = """
The choices for this question are: {choices_str}

OUTPUT FORMAT — a single JSON object, no markdown fences:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "probabilities": {{{{"<choice>": <float>, ...}}}},
  "prediction": "<one of the choices above>"
}}
The probabilities must sum to 1.0 and each key must exactly match one of the provided choices."""


_NUMERIC_OUTPUT_SPEC = """
OUTPUT FORMAT — a single JSON object, no markdown fences. You MUST output this even if uncertain; use your best estimate and explain uncertainty in the reasoning field:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts; state uncertainty explicitly if needed>",
  "prediction": <numeric value as a float>{unit_hint}
}}"""


_LEGAL_OUTPUT_SPEC = """
OUTPUT FORMAT — a single JSON object, no markdown fences. You MUST output this even if uncertain:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "probability_petitioner": <float in [0.0, 1.0]; probability that the petitioner wins>
}}"""


_SALARY_OUTPUT_SPEC = """
OUTPUT FORMAT — a single JSON object, no markdown fences. You MUST output this even if uncertain:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts; state uncertainty explicitly if needed>",
  "predicted_salary": <positive integer; Average Annual Value in USD, e.g. 12500000 for $12.5M>
}}"""


_STOCK_OUTPUT_SPEC = """
The tickers to rank are: {tickers_str}

OUTPUT FORMAT — a single JSON object, no markdown fences. You MUST output this even if uncertain:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "ranking": ["TICKER_BEST", "...", "TICKER_WORST"]
}}
The ranking must contain EXACTLY the tickers listed above (no additions, no omissions, no duplicates), ordered from highest expected return to lowest."""


_QUERY_GEN_PROMPT = """You are generating web search queries to gather relevant evidence for a forecasting question.
All evidence must predate the cutoff date. Generate search queries that would help find useful information to answer the question.

CUTOFF DATE: {cutoff_date}

QUESTION:
{question}

BACKGROUND:
{background}

Generate between {min_queries} and {max_queries} search queries.
- Each query should target a DISTINCT aspect of the information you need.
- Include the year in each query to anchor temporal relevance.
- Do not include queries about events after {cutoff_date}.

Output as JSON array: [{{"query": "..."}}]"""


_REGEN_SYSTEM_PREFIX = """The following evidence items from your previous response were flagged as potentially containing information that was not publicly available before the cutoff date: {leaked_ids_str}

You must generate a new response that does NOT rely on these items. Build a fresh evidence list and reasoning that avoids the leaked information. Your evidence list may include other facts, but not facts that depend on or refer to the flagged information."""


_AGGREGATOR_SYSTEM = """You are an expert forecaster. You have been given a list of verified evidence items that have been confirmed to predate the knowledge cutoff. Using ONLY this evidence, produce a final structured prediction. Do not introduce any new facts not in the provided evidence list."""

_AGGREGATOR_USER = """Evidence items (all verified as pre-cutoff):
{evidence_json}

{question_block}

{output_spec}"""


def build_unconstrained_prompt(inst: ForecastingTaskInstance) -> tuple[str, str]:
    """Build the (system, user) prompt for B0 — no temporal constraint.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """

    output_spec = _build_output_spec(inst)
    system = _SYSTEM_UNCONSTRAINED.format(output_spec=output_spec)
    user = _build_user_message(inst, context_block="")
    return system, user


def build_generation_prompt(
    inst: ForecastingTaskInstance,
    context_block: str = "",
) -> tuple[str, str]:
    """Build the (system, user) generation prompt for B1–B4.

    Args:
        inst: Task instance containing question, background, and cutoff.
        context_block: Pre-fetched RAG context (empty for B1/B2).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """

    output_spec = _build_output_spec(inst)
    system = _SYSTEM_PREAMBLE.format(cutoff_date=inst.temporal_cutoff, output_spec=output_spec)
    user = _build_user_message(inst, context_block)
    return system, user


def build_regeneration_prompt(
    inst: ForecastingTaskInstance,
    context_block: str,
    leaked_claim_ids: list[int],
    previous_evidence: list[dict],
) -> tuple[str, str]:
    """Build the (system, user) regeneration prompt when leaked claims are detected."""

    ids_str = ", ".join(f"[{i}]" for i in sorted(leaked_claim_ids))
    output_spec = _build_output_spec(inst)
    leaked_block = _REGEN_SYSTEM_PREFIX.format(leaked_ids_str=ids_str)
    system = _SYSTEM_PREAMBLE.format(cutoff_date=inst.temporal_cutoff, output_spec=output_spec)
    system = leaked_block + "\n\n" + system

    parts: list[str] = _base_question_parts(inst, context_block)
    if previous_evidence:
        parts.append("")
        parts.append("Your previous evidence list (for reference — do not reuse leaked items):")
        for ev in previous_evidence:
            parts.append(f"  [{ev.get('id', '?')}] {ev.get('fact', '')} (source_date: {ev.get('source_date', 'unknown')})")

    return system, "\n".join(parts)


def build_aggregator_prompt(
    inst: ForecastingTaskInstance,
    clean_evidence: list[dict],
) -> tuple[str, str]:
    """Build the (system, user) aggregation prompt from verified clean evidence."""

    import json

    output_spec = _build_output_spec(inst)
    question_block = f"QUESTION: {inst.question}\n\nBACKGROUND: {inst.background}"
    if inst.answer_type == "multiple_choice" and inst.choices:
        question_block += "\n\nCHOICES: " + "; ".join(inst.choices)
    if inst.answer_type == "stock" and inst.choices:
        question_block += "\n\nTICKERS TO RANK: " + ", ".join(inst.choices)

    user = _AGGREGATOR_USER.format(
        evidence_json=json.dumps(clean_evidence, ensure_ascii=False, indent=2),
        question_block=question_block,
        output_spec=output_spec.strip(),
    )
    return _AGGREGATOR_SYSTEM, user


def build_query_generation_prompt(inst: ForecastingTaskInstance) -> str:
    """Build the RAG query generation prompt for one instance."""

    min_q, max_q = _query_range(inst.answer_type, inst.domain)
    return _QUERY_GEN_PROMPT.format(
        cutoff_date=inst.temporal_cutoff,
        question=inst.question,
        background=inst.background,
        min_queries=min_q,
        max_queries=max_q,
    )


def _build_user_message(inst: ForecastingTaskInstance, context_block: str) -> str:
    return "\n".join(_base_question_parts(inst, context_block))


def _base_question_parts(inst: ForecastingTaskInstance, context_block: str) -> list[str]:
    parts: list[str] = []
    if context_block:
        parts.append("RETRIEVED CONTEXT (all pre-cutoff):")
        parts.append(context_block)
        parts.append("")
    parts.append(f"QUESTION: {inst.question}")
    parts.append("")
    parts.append(f"BACKGROUND: {inst.background}")
    if inst.answer_type == "multiple_choice" and inst.choices:
        parts.append("")
        parts.append("CHOICES: " + "; ".join(inst.choices))
    if inst.answer_type == "stock" and inst.choices:
        parts.append("")
        parts.append("TICKERS TO RANK: " + ", ".join(inst.choices))
    return parts


def _build_output_spec(inst: ForecastingTaskInstance) -> str:
    if inst.answer_type == "binary":
        return _BINARY_OUTPUT_SPEC
    if inst.answer_type == "multiple_choice":
        choices_str = "; ".join(inst.choices or [])
        return _MC_OUTPUT_SPEC.format(choices_str=choices_str)
    if inst.answer_type == "numeric":
        unit_hint = f', "unit": "<{inst.unit or "appropriate unit"}>"' if inst.unit else ""
        return _NUMERIC_OUTPUT_SPEC.format(unit_hint=unit_hint)
    if inst.answer_type == "legal":
        return _LEGAL_OUTPUT_SPEC
    if inst.answer_type == "salary":
        return _SALARY_OUTPUT_SPEC
    if inst.answer_type == "stock":
        tickers_str = ", ".join(inst.choices or [])
        return _STOCK_OUTPUT_SPEC.format(tickers_str=tickers_str)
    raise ValueError(f"Unknown answer_type: {inst.answer_type!r}")


def _query_range(answer_type: str, domain: str) -> tuple[int, int]:
    if answer_type == "binary":
        return (3, 6)
    if answer_type == "multiple_choice":
        return (4, 8)
    if answer_type == "legal":
        return (4, 8)
    if answer_type == "salary":
        return (5, 10)
    if answer_type == "stock":
        return (5, 10)
    return (5, 10)
