from __future__ import annotations

from typing import Iterable, Optional

from .professor_preprocessing import ProfessorRecord


TITLE_WEIGHTS = {
    "assistant": 1.0,
    "lecturer": 0.8,
    "associate": 0.5,
    "professor": 0.3,
}


def normalize_title(title: str) -> str:
    title_lower = title.lower()
    if "assistant" in title_lower:
        return "assistant"
    if "lecturer" in title_lower:
        return "lecturer"
    if "associate" in title_lower:
        return "associate"
    if "professor" in title_lower:
        return "professor"
    return "professor"


def compute_priority_score(
    title: str,
    is_engineering: bool,
    years_since_phd: Optional[int] = None,
    w_years: float = 1.0,
    w_title: float = 1.0,
    w_engineering: float = 1.0,
    default_years_since_phd: int = 10,
    engineering_bonus: float = 1.2,
) -> float:
    years_value = years_since_phd if years_since_phd is not None else default_years_since_phd
    years_score = 1.0 / (1.0 + max(0, years_value))

    title_key = normalize_title(title)
    title_score = TITLE_WEIGHTS.get(title_key, 0.3)

    engineering_score = engineering_bonus if is_engineering else 1.0

    return (w_years * years_score) + (w_title * title_score) + (w_engineering * engineering_score)


def assign_priority_scores(
    records: Iterable[ProfessorRecord],
    w_years: float = 1.0,
    w_title: float = 1.0,
    w_engineering: float = 1.0,
    default_years_since_phd: int = 10,
    engineering_bonus: float = 1.2,
) -> None:
    for record in records:
        record.priority_score = compute_priority_score(
            title=record.title,
            is_engineering=record.is_engineering,
            years_since_phd=record.years_since_phd,
            w_years=w_years,
            w_title=w_title,
            w_engineering=w_engineering,
            default_years_since_phd=default_years_since_phd,
            engineering_bonus=engineering_bonus,
        )
