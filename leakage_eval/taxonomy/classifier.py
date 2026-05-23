"""Deterministic helpers for C0-C8 taxonomy decisions."""

from __future__ import annotations

import re

from LeakageEvaluationFramework.leakage_eval.schemas import ClaimCategory, ClaimRecord


_NUMBER_RE = re.compile(r"(\$?\d[\d,]*(?:\.\d+)?%?)")
_COMPARATIVE_TERMS = ("than", "compared", "more", "less", "higher", "lower", "better", "worse")
_SOURCE_TERMS = ("report", "filing", "opinion", "article", "database", "press release", "stated")
_EVENT_TERMS = ("signed", "announced", "filed", "decided", "granted", "enacted", "released", "appointed")
_DEFINITION_TERMS = ("means", "is defined as", "refers to")


def heuristic_category(record: ClaimRecord) -> ClaimCategory:
    """Assign a conservative heuristic category.

    This is not a replacement for the LLM decision tree. It provides a local
    fallback and testable baseline when no judge output is available.
    """

    text = record.claim_text.lower()
    target = record.target_description.lower()
    if target and _contains_target_outcome(text, target):
        return "C7"
    if any(term in text for term in ("became a landmark", "after the decision", "after the ruling")):
        return "C8"
    if any(term in text for term in _DEFINITION_TERMS):
        return "C0"
    if any(term in text for term in _SOURCE_TERMS):
        return "C6"
    if _NUMBER_RE.search(text):
        return "C3"
    if any(term in text for term in _COMPARATIVE_TERMS):
        return "C4"
    if any(term in text for term in _EVENT_TERMS):
        return "C5"
    if _looks_stable_attribute(text):
        return "C1"
    return "C2"


def _contains_target_outcome(text: str, target: str) -> bool:
    if len(target) < 8:
        return False
    return target in text


def _looks_stable_attribute(text: str) -> bool:
    stable_terms = (" is a ", " is an ", " operates in ", " consists of ", " established under ")
    return any(term in text for term in stable_terms)

