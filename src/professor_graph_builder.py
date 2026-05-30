from __future__ import annotations

from typing import Iterable

import networkx as nx

from .embedding_model import TextEmbedder
from .hybrid_evidence_scorer import HybridEvidenceScorer, build_professor_record_chunks, normalize_weights
from .professor_preprocessing import ProfessorRecord


def build_graph(
    records: Iterable[ProfessorRecord],
    embedder: TextEmbedder,
    similarity_threshold: float = 0.2,
    interests_weight: float = 0.25,
    project_weight: float = 0.15,
    paper_weight: float = 0.20,
    deeptech_weight: float = 0.40,
) -> nx.Graph:
    record_list = list(records)
    graph = nx.Graph()
    if not record_list:
        return graph

    weights = normalize_weights(
        {
            "interests": interests_weight,
            "project": project_weight,
            "paper": paper_weight,
            "deeptech": deeptech_weight,
        },
        {
            "interests": 0.25,
            "project": 0.15,
            "paper": 0.20,
            "deeptech": 0.40,
        },
    )
    scorer = HybridEvidenceScorer(
        record_chunks=build_professor_record_chunks(record_list),
        embedder=embedder,
        field_weights=weights,
        calibration_enabled=False,
        coverage_bonus=0.0,
    )
    similarities = scorer.profile_similarity_matrix()

    for record in record_list:
        graph.add_node(record.name, department=record.department, title=record.title)

    for i, left in enumerate(record_list):
        for j in range(i + 1, len(record_list)):
            combined = float(similarities[i, j]) if similarities.size else 0.0
            if combined >= similarity_threshold:
                graph.add_edge(left.name, record_list[j].name, combined=combined)

    return graph
