from __future__ import annotations

from typing import Dict, Iterable, List, Set
import re

import networkx as nx

from .professor_preprocessing import ProfessorRecord


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_interests(text: str) -> Set[str]:
    text_lower = text.lower()
    return set(TOKEN_RE.findall(text_lower))


def _deeptech_text(record: ProfessorRecord) -> str:
    parts: List[str] = []
    for project in record.deeptech_projects:
        parts.extend(project.applications)
        parts.extend(project.industries)
        parts.append(project.overview)
        parts.append(project.tech_edges)
    return " ".join(part for part in parts if part)


def build_graph(
    records: Iterable[ProfessorRecord],
    similarity_threshold: float = 0.2,
    department_edge_weight: float = 0.1,
) -> nx.Graph:
    graph = nx.Graph()

    record_list = list(records)
    interest_tokens: Dict[str, Set[str]] = {
        record.name: _tokenize_interests(record.research_interests) for record in record_list
    }
    project_tokens: Dict[str, Set[str]] = {
        record.name: _tokenize_interests(record.attributes.get("leading_project", "")) for record in record_list
    }
    paper_tokens: Dict[str, Set[str]] = {
        record.name: _tokenize_interests(record.attributes.get("paper", "")) for record in record_list
    }
    deeptech_tokens: Dict[str, Set[str]] = {
        record.name: _tokenize_interests(_deeptech_text(record)) for record in record_list
    }

    for record in record_list:
        graph.add_node(record.name, department=record.department, title=record.title)

    for i, left in enumerate(record_list):
        for right in record_list[i + 1 :]:
            weight = 0.0
            if left.department and left.department == right.department:
                weight = max(weight, department_edge_weight)

            def calc_sim(tokens_l: Set[str], tokens_r: Set[str]) -> float:
                if not tokens_l or not tokens_r:
                    return 0.0
                intersection = tokens_l & tokens_r
                union = tokens_l | tokens_r
                return len(intersection) / max(1, len(union))

            sim_interests = calc_sim(interest_tokens.get(left.name, set()), interest_tokens.get(right.name, set()))
            sim_projects = calc_sim(project_tokens.get(left.name, set()), project_tokens.get(right.name, set()))
            sim_papers = calc_sim(paper_tokens.get(left.name, set()), paper_tokens.get(right.name, set()))
            sim_deeptech = calc_sim(deeptech_tokens.get(left.name, set()), deeptech_tokens.get(right.name, set()))

            # Combine similarities
            combined_sim = (0.25 * sim_interests) + (0.15 * sim_projects) + (0.2 * sim_papers) + (0.4 * sim_deeptech)

            if combined_sim >= similarity_threshold:
                weight = max(weight, combined_sim)

            if weight > 0:
                graph.add_edge(left.name, right.name, weight=weight)

    return graph
