"""Prompt templates for taxonomy and date verification."""

TAXONOMY_DATE_SYSTEM_PROMPT = """You are a temporal provenance auditor.

Your task is public-availability date verification, not factual truth checking and not outcome prediction. For each claim, decide the earliest reliable date on which the claim could be publicly verified. Use this date to determine leakage relative to the cutoff. Return only valid JSON matching the requested schema.
"""

_TAXONOMY_DECISION_TREE = """
C0-C8 TAXONOMY (apply the first matching category in the precedence order below):

Precedence: C7 > C8 > C6 > C3 > C4 > C5 > C2 > C1 > C0.

C7 TARGET OUTCOME: The claim states or directly reveals the event/outcome being predicted.
Always leaked because the target resolves after the cutoff.

C8 POST-OUTCOME CONSEQUENCE: The claim describes an effect or consequence that exists only
after the target outcome occurs. Always leaked.

C6 PUBLICATION / SOURCE: The claim explicitly attributes information to a dated report,
article, filing, opinion, release, database, or other publication. Use that source's public
release date.

C3 QUANTITATIVE MEASUREMENT: The claim states a measured numeric value (revenue, price,
rate, count, score, salary, etc.). Use the date the measurement was publicly released, not
merely the end of the measured period.

C4 COMPARATIVE / RELATIONAL: The claim compares two or more entities, values, periods, or
states. Use the latest public-availability date among the facts needed for the comparison.

C5 DISCRETE EVENT: The claim says a specific event occurred (signed, announced, filed,
decided, launched, enacted, appointed, etc.). Use the event's public announcement/release
date.

C2 TIME-INDEXED STATE: The claim states a condition, status, role, trend, or other
time-dependent fact and does not match C3-C6. This is the residual category for temporal
claims. Use the earliest source date corroborating that state.

C1 STABLE ENTITY ATTRIBUTE: The claim states a stable factual attribute that was already
true and publicly knowable through the cutoff. If the attribute changes over time or is
explicitly time-indexed, use C2 or a more specific temporal category instead.

C0 TIMELESS DEFINITION / LOGICAL TRUTH: The claim is definitional, mathematical, or
time-invariant by meaning (for example, a metric definition). It has no public-availability
date requirement.

If a claim matches multiple descriptions, always use the highest-precedence category.
"""


def build_taxonomy_date_prompt(items_json: str, *, cutoff_date: str, task: str, target_description: str) -> str:
    """Build the taxonomy/date prompt body."""

    return f"""Cutoff date: {cutoff_date}
Task: {task}
Target outcome description: {target_description}

{_TAXONOMY_DECISION_TREE}

Apply the C0-C8 decision tree independently to each item. Stop at the first matching rule.

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

