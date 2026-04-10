from __future__ import annotations

from typing import Iterable, List, Set
import re

import networkx as nx

from .startup_preprocessing import StartupRecord


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_categories(categories: List[str]) -> Set[str]:
    return {str(category).strip().lower() for category in categories if str(category).strip()}


def tokenize_description(description: str) -> Set[str]:
    return set(_TOKEN_RE.findall(str(description).lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def combined_startup_similarity(
    a: StartupRecord,
    b: StartupRecord,
    w_cat: float,
    w_desc: float,
) -> float:
    cat_sim = jaccard(tokenize_categories(a.categories), tokenize_categories(b.categories))
    desc_sim = jaccard(tokenize_description(a.description), tokenize_description(b.description))
    return (w_cat * cat_sim) + (w_desc * desc_sim)


def build_startup_graph(
    records: Iterable[StartupRecord],
    similarity_threshold: float,
    category_weight: float = 0.571,
    description_weight: float = 0.429,
) -> nx.Graph:
    record_list = list(records)
    graph = nx.Graph()

    total_weight = float(category_weight) + float(description_weight)
    if total_weight <= 0:
        w_cat, w_desc = 0.571, 0.429
    else:
        w_cat = float(category_weight) / total_weight
        w_desc = float(description_weight) / total_weight

    category_tokens = {
        record.startup_id: tokenize_categories(record.categories)
        for record in record_list
    }
    description_tokens = {
        record.startup_id: tokenize_description(record.description)
        for record in record_list
    }

    for record in record_list:
        graph.add_node(
            record.startup_id,
            company_name=record.company_name,
            source_year=record.source_year,
        )

    for i, left in enumerate(record_list):
        for right in record_list[i + 1 :]:
            cat_sim = jaccard(category_tokens[left.startup_id], category_tokens[right.startup_id])
            desc_sim = jaccard(description_tokens[left.startup_id], description_tokens[right.startup_id])
            combined = (w_cat * cat_sim) + (w_desc * desc_sim)

            if combined >= similarity_threshold:
                graph.add_edge(
                    left.startup_id,
                    right.startup_id,
                    combined=float(combined),
                    cat_sim=float(cat_sim),
                    desc_sim=float(desc_sim),
                )

    return graph
