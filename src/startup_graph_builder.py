from __future__ import annotations

from typing import Iterable

import networkx as nx

from .embedding_model import TextEmbedder
from .hybrid_evidence_scorer import HybridEvidenceScorer, build_startup_record_chunks, normalize_weights
from .startup_preprocessing import StartupRecord


def build_startup_graph(
    records: Iterable[StartupRecord],
    embedder: TextEmbedder,
    similarity_threshold: float,
    category_weight: float = 0.6,
    description_weight: float = 0.4,
) -> nx.Graph:
    record_list = list(records)
    graph = nx.Graph()
    if not record_list:
        return graph

    weights = normalize_weights(
        {
            "company": 0.05,
            "category": category_weight,
            "description": description_weight,
            "meta": 0.05,
        },
        {
            "company": 0.05,
            "category": 0.35,
            "description": 0.55,
            "meta": 0.05,
        },
    )
    scorer = HybridEvidenceScorer(
        record_chunks=build_startup_record_chunks(record_list),
        embedder=embedder,
        field_weights=weights,
        calibration_enabled=False,
        coverage_bonus=0.0,
    )
    similarities = scorer.profile_similarity_matrix()

    for record in record_list:
        graph.add_node(
            record.startup_id,
            company_name=record.company_name,
            source_year=record.source_year,
        )

    for i, left in enumerate(record_list):
        for j in range(i + 1, len(record_list)):
            combined = float(similarities[i, j]) if similarities.size else 0.0
            if combined >= similarity_threshold:
                graph.add_edge(left.startup_id, record_list[j].startup_id, combined=combined)

    return graph
