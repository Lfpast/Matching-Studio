from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def top_k_accuracy(ranked_lists: Iterable[Sequence[str]], ground_truth: Iterable[str], k: int) -> float:
    hits = 0
    total = 0
    for ranked, truth in zip(ranked_lists, ground_truth):
        total += 1
        if truth in ranked[:k]:
            hits += 1
    return hits / max(1, total)


def mean_reciprocal_rank(ranked_lists: Iterable[Sequence[str]], ground_truth: Iterable[str]) -> float:
    scores: List[float] = []
    for ranked, truth in zip(ranked_lists, ground_truth):
        try:
            rank = ranked.index(truth) + 1
            scores.append(1.0 / rank)
        except ValueError:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(ranked_lists: Iterable[Sequence[str]], ground_truth: Iterable[str], k: int) -> float:
    scores: List[float] = []
    for ranked, truth in zip(ranked_lists, ground_truth):
        dcg = 0.0
        for i, item in enumerate(ranked[:k]):
            if item == truth:
                dcg = 1.0 / np.log2(i + 2)
                break
        idcg = 1.0
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0
