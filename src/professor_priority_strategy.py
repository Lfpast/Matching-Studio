from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .professor_preprocessing import ProfessorRecord


TITLE_WEIGHT_DEFAULTS = {
    "assistant_professor": 1.0,
    "associate_professor": 0.85,
    "professor": 0.70,
    "chair": 0.55,
    "associate_head": 0.40,
    "head": 0.25,
    "other": 0.60,
}

TITLE_PATTERNS = (
    ("associate head", "associate_head"),
    ("associate department head", "associate_head"),
    ("head of department", "head"),
    ("department head", "head"),
    ("head", "head"),
    ("chair", "chair"),
    ("assistant professor", "assistant_professor"),
    ("assistant prof", "assistant_professor"),
    ("asst professor", "assistant_professor"),
    ("associate professor", "associate_professor"),
    ("assoc professor", "associate_professor"),
    ("full professor", "professor"),
    ("professor", "professor"),
)


def normalize_title(title: str) -> str:
    title_lower = " ".join(str(title or "").lower().split())
    if not title_lower:
        return "other"

    for pattern, normalized in TITLE_PATTERNS:
        if pattern in title_lower:
            return normalized

    return "other"


def compute_priority_score(
    title: str,
    title_weights: Optional[Mapping[str, float]] = None,
    default_title_score: float = 0.60,
) -> float:
    weights = dict(TITLE_WEIGHT_DEFAULTS)
    if isinstance(title_weights, Mapping):
        for key, value in title_weights.items():
            try:
                weights[str(key)] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

    title_key = normalize_title(title)
    return float(weights.get(title_key, default_title_score))


def assign_priority_scores(
    records: Iterable[ProfessorRecord],
    title_weights: Optional[Mapping[str, float]] = None,
    default_title_score: float = 0.60,
) -> None:
    for record in records:
        record.priority_score = compute_priority_score(
            title=record.title,
            title_weights=title_weights,
            default_title_score=default_title_score,
        )
