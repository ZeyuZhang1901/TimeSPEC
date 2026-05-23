"""Prompt templates for taxonomy and date verification."""

TAXONOMY_DATE_SYSTEM_PROMPT = """You are a temporal provenance auditor.

Your task is public-availability date verification, not factual truth checking and not outcome prediction. For each claim, decide the earliest reliable date on which the claim could be publicly verified. Use this date to determine leakage relative to the cutoff. Return only valid JSON matching the requested schema.
"""


def build_taxonomy_date_prompt(items_json: str, *, cutoff_date: str, task: str, target_description: str) -> str:
    """Build the taxonomy/date prompt body."""

    return f"""Cutoff date: {cutoff_date}
Task: {task}
Target outcome description: {target_description}

Apply the C0-C8 decision tree from taxonomy_and_leakage_detection.md independently to each item. Stop at the first matching rule.

For each item:
1. Treat declared_source_date as a candidate public-availability date, not ground truth.
2. Decide whether declared_source_date is plausible for when this exact claim could first be publicly verified.
3. If declared_source_date is implausible, provide the earliest reliable corrected date.
4. If the exact earliest date is uncertain, use the conservative latest plausible date and lower confidence.
5. Do not label a stable/background fact leaked merely because it appears in later litigation. For party attributes, use when the attribute was publicly knowable, not the later case date.
6. For claims about a specific case order, holding, filing, grant, denial, oral argument, or opinion, use the public release date of that document/event.
7. Leakage is determined from effective_source_date > cutoff_date, except direct outcome/consequential claims C7/C8 which are leaked by definition.

Items:
{items_json}

Output JSON:
{{"results":[{{"id":1,"category":"C0","matched_rule":1,"declared_date_plausible":true,"corrected_date":null,"effective_source_date":"YYYY-MM-DD","date_precision":"declared|corrected_exact|conservative_latest|missing","is_leaked":false,"confidence":"high|medium|low","reason":"brief date-verification explanation"}}]}}
"""

