from __future__ import annotations

from typing import Iterable

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_model import TextEmbedder
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

    total_weight = float(category_weight) + float(description_weight)
    if total_weight <= 0:
        w_cat, w_desc = 0.6, 0.4
    else:
        w_cat = float(category_weight) / total_weight
        w_desc = float(description_weight) / total_weight

    category_texts = [", ".join(record.categories).strip() or "startup category" for record in record_list]
    description_texts = [str(record.description).strip() or "startup brief description" for record in record_list]

    cat_embeddings = embedder.encode(category_texts)
    desc_embeddings = embedder.encode(description_texts)

    cat_sims = cosine_similarity(cat_embeddings)
    desc_sims = cosine_similarity(desc_embeddings)

    for record in record_list:
        graph.add_node(
            record.startup_id,
            company_name=record.company_name,
            source_year=record.source_year,
        )

    for i, left in enumerate(record_list):
        for j in range(i + 1, len(record_list)):
            right = record_list[j]
            cat_sim = float(cat_sims[i, j])
            desc_sim = float(desc_sims[i, j])
            combined = (w_cat * cat_sim) + (w_desc * desc_sim)

            if combined >= similarity_threshold:
                graph.add_edge(
                    left.startup_id,
                    right.startup_id,
                    combined=combined,
                    cat_sim=cat_sim,
                    desc_sim=desc_sim,
                )

    return graph
