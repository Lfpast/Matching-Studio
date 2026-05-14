from __future__ import annotations

import argparse
import os

import yaml

from src.professor_preprocessing import build_records, clean_dataframe, load_and_merge_data
from src.embedding_model import TextEmbedder
from src.env_loader import load_project_env
from src.professor_graph_builder import build_graph
from src.professor_matching_engine import MatchingEngine
from src.professor_priority_strategy import assign_priority_scores


load_project_env()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Professor matching system")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--query", required=True, help="Enterprise query text")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})

    df = load_and_merge_data(
        info_path=data_cfg.get("raw_csv", "data/raw/professor_information.csv"),
        projects_path=data_cfg.get("projects_csv", "data/raw/professor_projects.csv"),
        publications_path=data_cfg.get("publications_csv", "data/raw/professor_publications.csv")
    )
    df = clean_dataframe(df)
    records = build_records(df)

    professor_cfg = config.get("professor", {})

    priority_cfg = professor_cfg.get("priority", {})
    assign_priority_scores(
        records,
        title_weights=priority_cfg.get("title_weights", {}),
        default_title_score=priority_cfg.get("default_title_score", 0.60),
    )

    embedding_cfg = config.get("embedding", {})
    embedder = TextEmbedder(
        model_name=embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=embedding_cfg.get("batch_size", 4),
    )

    graph_cfg = professor_cfg.get("graph", {})
    graph = build_graph(
        records,
        embedder=embedder,
        similarity_threshold=graph_cfg.get("similarity_threshold", 0.2),
        interests_weight=graph_cfg.get("interests_weight", 0.25),
        project_weight=graph_cfg.get("project_weight", 0.15),
        paper_weight=graph_cfg.get("paper_weight", 0.20),
        deeptech_weight=graph_cfg.get("deeptech_weight", 0.40),
    )
    
    query_cfg = config.get("query", {})
    matching_cfg = professor_cfg.get("matching", {})
    semantic_cfg = professor_cfg.get("semantic_matching", {})

    engine = MatchingEngine(
        records=records, 
        embedder=embedder, 
        graph=graph, 
        query_config=query_cfg,
        semantic_config=semantic_cfg,
    )

    top_k = args.top_k if args.top_k is not None else int(matching_cfg.get("default_top_k", 5))
    result = engine.match(
        query=args.query,
        top_k=top_k,
        alpha=matching_cfg.get("default_alpha", 0.8),
        beta=matching_cfg.get("default_beta", 0.2),
        graph_neighbor_weight=matching_cfg.get("default_graph_neighbor_weight", 0.1),
        validate_query=query_cfg.get("enable_validation", True),
        use_keyword_extraction=query_cfg.get("enable_keyword_extraction", True),
    )
    
    # Print query processing results
    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['suggestions']:
        print(f"Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    
    if result['keywords']:
        print(f"Extracted Keywords: {[kw for kw, _ in result['keywords'][:5]]}")
    
    if result['enhanced_query'] != args.query:
        print(f"Enhanced Query: {result['enhanced_query']}")
    
    print(f"{'='*60}\n")
    
    # Print matching results
    if result['results']:
        print("Top Matches:")
        for rank, item in enumerate(result['results'], start=1):
            print(f"{rank}. {item['name']} | score={item['score']:.4f} | dept={item['department']}")
            print(f"   Research: {item['research_interests'][:80]}...")
    else:
        print("No matching professors found for this query.")


if __name__ == "__main__":
    main()
