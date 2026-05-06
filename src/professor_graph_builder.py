from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_model import TextEmbedder
from .professor_preprocessing import ProfessorRecord


def _deeptech_text(record: ProfessorRecord) -> str:
    parts: List[str] = []
    for project in record.deeptech_projects:
        parts.extend(project.applications)
        parts.extend(project.industries)
        parts.append(project.overview)
        parts.append(project.tech_edges)
    return " ".join(part for part in parts if part)


def _build_field_texts(record_list: List[ProfessorRecord]) -> Tuple[List[str], List[str], List[str], List[str]]:
    interests_texts = [str(record.research_interests).strip() or "professor research interests" for record in record_list]
    project_texts = [str(record.attributes.get("leading_project", "")).strip() or "professor leading project" for record in record_list]
    paper_texts = [str(record.attributes.get("paper", "")).strip() or "professor paper" for record in record_list]
    deeptech_texts = [_deeptech_text(record).strip() or "professor deeptech" for record in record_list]
    return interests_texts, project_texts, paper_texts, deeptech_texts


def _normalize_weights(
    interests_weight: float,
    project_weight: float,
    paper_weight: float,
    deeptech_weight: float,
) -> Tuple[float, float, float, float]:
    total_weight = float(interests_weight) + float(project_weight) + float(paper_weight) + float(deeptech_weight)
    if total_weight <= 0:
        return 0.25, 0.15, 0.20, 0.40

    return (
        float(interests_weight) / total_weight,
        float(project_weight) / total_weight,
        float(paper_weight) / total_weight,
        float(deeptech_weight) / total_weight,
    )


def build_graph(
    records: Iterable[ProfessorRecord],
    embedder: TextEmbedder,
    similarity_threshold: float = 0.2,
    interests_weight: float = 0.25,
    project_weight: float = 0.15,
    paper_weight: float = 0.20,
    deeptech_weight: float = 0.40,
) -> nx.Graph:
    graph = nx.Graph()

    record_list = list(records)
    if not record_list:
        return graph

    w_interests, w_project, w_paper, w_deeptech = _normalize_weights(
        interests_weight=interests_weight,
        project_weight=project_weight,
        paper_weight=paper_weight,
        deeptech_weight=deeptech_weight,
    )

    interest_texts, project_texts, paper_texts, deeptech_texts = _build_field_texts(record_list)

    interest_embeddings = embedder.encode(interest_texts)
    project_embeddings = embedder.encode(project_texts)
    paper_embeddings = embedder.encode(paper_texts)
    deeptech_embeddings = embedder.encode(deeptech_texts)

    interest_sims = cosine_similarity(interest_embeddings)
    project_sims = cosine_similarity(project_embeddings)
    paper_sims = cosine_similarity(paper_embeddings)
    deeptech_sims = cosine_similarity(deeptech_embeddings)

    for record in record_list:
        graph.add_node(record.name, department=record.department, title=record.title)

    for i, left in enumerate(record_list):
        for j in range(i + 1, len(record_list)):
            right = record_list[j]
            sim_interests = float(interest_sims[i, j])
            sim_projects = float(project_sims[i, j])
            sim_papers = float(paper_sims[i, j])
            sim_deeptech = float(deeptech_sims[i, j])

            combined_sim = (
                (w_interests * sim_interests)
                + (w_project * sim_projects)
                + (w_paper * sim_papers)
                + (w_deeptech * sim_deeptech)
            )

            if combined_sim >= similarity_threshold:
                graph.add_edge(
                    left.name,
                    right.name,
                    combined=combined_sim,
                    interests_sim=sim_interests,
                    project_sim=sim_projects,
                    paper_sim=sim_papers,
                    deeptech_sim=sim_deeptech,
                )

    return graph
