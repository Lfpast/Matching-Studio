from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_model import TextEmbedder
from .professor_preprocessing import ProfessorRecord
from .query_processor import (
    EnhancedQueryProcessor,
    QueryStatus,
    QueryValidationResult,
)


class MatchingEngine:
    def __init__(
        self,
        records: Iterable[ProfessorRecord],
        embedder: TextEmbedder,
        graph: Optional[nx.Graph] = None,
        query_config: Optional[Dict] = None,
        semantic_config: Optional[Dict] = None,
    ) -> None:
        self.records = list(records)
        self.embedder = embedder
        self.graph = graph
        self.query_config = query_config or {}
        self.semantic_config = semantic_config or {}
        self.semantic_weights = self._load_semantic_weights()
        self.min_field_similarity = self._safe_float(self.semantic_config.get("min_field_similarity", 0.08), 0.08)
        self.lower_bound = self._safe_float(self.semantic_config.get("lower_bound", 0.50), 0.50)
        self.name_to_index = {record.name: idx for idx, record in enumerate(self.records)}
        (
            self.interests_embeddings,
            self.project_embeddings,
            self.paper_embeddings,
            self.deeptech_embeddings,
        ) = self._build_field_embeddings()
        
        # Build domain embeddings for query validation
        self._domain_texts = self._build_domain_texts()
        self._domain_embeddings = self.embedder.encode(self._domain_texts) if self._domain_texts else None
        
        # Initialize query processor
        self.query_processor = EnhancedQueryProcessor(
            embedder=self.embedder,
            domain_embeddings=self._domain_embeddings,
            domain_texts=self._domain_texts,
            similarity_threshold=self.query_config.get("similarity_threshold", 0.25),
            weak_threshold=self.query_config.get("weak_threshold", 0.35),
        )
    
    def _build_domain_texts(self) -> List[str]:
        """Build domain text corpus from professor research interests."""
        domain_texts = []
        for record in self.records:
            if record.research_interests:
                domain_texts.append(record.research_interests)
        return domain_texts

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_semantic_weights(self) -> Dict[str, float]:
        defaults = {
            "interests": 0.20,
            "project": 0.10,
            "paper": 0.30,
            "deeptech": 0.40,
        }

        configured_weights = self.semantic_config.get("field_weights", {}) if isinstance(self.semantic_config, dict) else {}
        if not isinstance(configured_weights, dict):
            configured_weights = {}

        raw_weights: Dict[str, float] = {}
        for key, default_value in defaults.items():
            raw_value = configured_weights.get(key, default_value)
            try:
                raw_weights[key] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                raw_weights[key] = default_value

        total = sum(raw_weights.values())
        if total <= 0:
            return defaults

        return {key: (value / total) for key, value in raw_weights.items()}

    def _deeptech_record_text(self, record: ProfessorRecord) -> str:
        return " ".join(
            part
            for part in (self._deeptech_project_text(project) for project in record.deeptech_projects)
            if part
        )

    def _build_field_texts(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        interests_texts = [str(record.research_interests).strip() or "professor research interests" for record in self.records]
        project_texts = [str(record.attributes.get("leading_project", "")).strip() or "professor leading project" for record in self.records]
        paper_texts = [str(record.attributes.get("paper", "")).strip() or "professor paper" for record in self.records]
        deeptech_texts = [self._deeptech_record_text(record).strip() or "professor deeptech" for record in self.records]
        return interests_texts, project_texts, paper_texts, deeptech_texts

    def _build_field_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.records:
            empty = np.empty((0, 0), dtype=float)
            return empty, empty, empty, empty

        interests_texts, project_texts, paper_texts, deeptech_texts = self._build_field_texts()

        if self.embedder.backend == "tfidf" and self.embedder.vectorizer is not None:
            self.embedder.fit([*interests_texts, *project_texts, *paper_texts, *deeptech_texts])

        return (
            self.embedder.encode(interests_texts),
            self.embedder.encode(project_texts),
            self.embedder.encode(paper_texts),
            self.embedder.encode(deeptech_texts),
        )

    def _score_field_similarity(self, query_vec: np.ndarray, field_embeddings: np.ndarray) -> np.ndarray:
        if not self.records or field_embeddings.size == 0:
            return np.zeros(len(self.records), dtype=float)

        try:
            return cosine_similarity(query_vec, field_embeddings)[0]
        except Exception:
            return np.zeros(len(self.records), dtype=float)

    def _score_semantic(self, query_vec: np.ndarray) -> np.ndarray:
        if not self.records:
            return np.zeros(len(self.records), dtype=float)

        interests_scores = self._score_field_similarity(query_vec, self.interests_embeddings)
        project_scores = self._score_field_similarity(query_vec, self.project_embeddings)
        paper_scores = self._score_field_similarity(query_vec, self.paper_embeddings)
        deeptech_scores = self._score_field_similarity(query_vec, self.deeptech_embeddings)

        w_interests = self.semantic_weights.get("interests", 0.20)
        w_project = self.semantic_weights.get("project", 0.10)
        w_paper = self.semantic_weights.get("paper", 0.30)
        w_deeptech = self.semantic_weights.get("deeptech", 0.40)

        weighted_scores = (
            (w_interests * interests_scores)
            + (w_project * project_scores)
            + (w_paper * paper_scores)
            + (w_deeptech * deeptech_scores)
        )

        field_scores = np.vstack((interests_scores, project_scores, paper_scores, deeptech_scores))
        weight_vector = np.array([w_interests, w_project, w_paper, w_deeptech], dtype=float).reshape(-1, 1)
        coverage = (
            ((field_scores >= self.min_field_similarity).astype(float) * weight_vector).sum(axis=0)
            / max(float(weight_vector.sum()), 1e-9)
        )

        strict_scores = weighted_scores * (self.lower_bound + ((1.0 - self.lower_bound) * coverage))
        return np.maximum(strict_scores, 0.0)

    @staticmethod
    def _deeptech_project_text(project) -> str:
        return " ".join(
            part
            for part in [
                project.overview,
                project.tech_edges,
                " ".join(project.applications),
                " ".join(project.industries),
            ]
            if part
        )

    def _rank_deeptech_projects(self, record: ProfessorRecord, query_vec: np.ndarray) -> List[Dict[str, object]]:
        projects_payload: List[Dict[str, object]] = []
        if not record.deeptech_projects:
            return projects_payload

        project_texts = [self._deeptech_project_text(project) for project in record.deeptech_projects]
        if not any(project_texts):
            for project in record.deeptech_projects:
                projects_payload.append(
                    {
                        "source": project.source,
                        "cluster": project.cluster,
                        "technology_title": project.technology_title,
                        "trl": project.trl,
                        "ip_status": project.ip_status,
                        "overview": project.overview,
                        "tech_edges": project.tech_edges,
                        "applications": project.applications,
                        "industries": project.industries,
                        "relevance_score": 0.0,
                    }
                )
            return projects_payload

        project_embeddings = self.embedder.encode(project_texts)
        project_sims = cosine_similarity(query_vec, project_embeddings)[0]

        ranked = sorted(
            enumerate(record.deeptech_projects),
            key=lambda item: float(project_sims[item[0]]),
            reverse=True,
        )

        for idx, project in ranked:
            projects_payload.append(
                {
                    "source": project.source,
                    "cluster": project.cluster,
                    "technology_title": project.technology_title,
                    "trl": project.trl,
                    "ip_status": project.ip_status,
                    "overview": project.overview,
                    "tech_edges": project.tech_edges,
                    "applications": project.applications,
                    "industries": project.industries,
                    "relevance_score": float(project_sims[idx]),
                }
            )

        return projects_payload

    def _graph_neighbor_scores(self, semantic_scores: np.ndarray) -> np.ndarray:
        if not self.graph:
            return np.zeros_like(semantic_scores)

        neighbor_scores = np.zeros_like(semantic_scores)
        for record in self.records:
            idx = self.name_to_index.get(record.name)
            if idx is None:
                continue
            neighbors = list(self.graph.neighbors(record.name))
            if not neighbors:
                continue
            neighbor_values = [semantic_scores[self.name_to_index[n]] for n in neighbors if n in self.name_to_index]
            if neighbor_values:
                neighbor_scores[idx] = float(np.mean(neighbor_values))
        return neighbor_scores

    def match(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.8,
        beta: float = 0.2,
        graph_neighbor_weight: float = 0.1,
        validate_query: bool = True,
        use_keyword_extraction: bool = True,
    ) -> Dict[str, object]:
        """
        Match a query to professors.
        
        Args:
            query: The industry query
            top_k: Number of results to return
            alpha: Weight for similarity score
            beta: Weight for priority score
            graph_neighbor_weight: Weight for graph neighbor scores
            validate_query: Whether to validate query relevance
            use_keyword_extraction: Whether to use keyword extraction
            
        Returns:
            Dict containing:
                - status: Query validation status
                - message: Validation message
                - suggestions: List of suggestions if query needs improvement
                - results: List of matched professors (empty if invalid)
                - keywords: Extracted keywords (if enabled)
                - enhanced_query: The query used for matching (may be enhanced)
        """
        # Process the query
        if validate_query:
            enhanced_query, validation, keywords = self.query_processor.get_enhanced_query(query)
        else:
            validation = QueryValidationResult(
                status=QueryStatus.VALID,
                message="Query validation skipped.",
                suggestions=[],
                confidence=1.0
            )
            keywords = None
            enhanced_query = query
        
        # If query is invalid, return early with no results
        if validation.status == QueryStatus.INVALID:
            return {
                "status": validation.status.value,
                "message": validation.message,
                "suggestions": validation.suggestions,
                "results": [],
                "keywords": [],
                "enhanced_query": query,
            }
        
        # Use enhanced query if keyword extraction produced one
        if use_keyword_extraction and keywords and keywords.filtered_query:
            match_query = enhanced_query
        else:
            match_query = query
        
        query_vec = self.embedder.encode([match_query])
        semantic_scores = self._score_semantic(query_vec)
        priorities = np.array([record.priority_score for record in self.records])
        neighbor_scores = self._graph_neighbor_scores(semantic_scores)

        final_scores = (alpha * semantic_scores) + (beta * priorities) + (graph_neighbor_weight * neighbor_scores)
        deeptech_payloads: List[List[Dict[str, object]]] = []

        for idx, record in enumerate(self.records):
            payload = self._rank_deeptech_projects(record, query_vec)
            deeptech_payloads.append(payload)
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            record = self.records[idx]
            deeptech_projects = deeptech_payloads[idx]
            results.append(
                {
                    "name": record.name,
                    "department": record.department,
                    "title": record.title,
                    "url": record.url,
                    "research_interests": record.research_interests,
                    "score": float(final_scores[idx]),
                    "similarity": float(semantic_scores[idx]),
                    "priority_score": float(record.priority_score),
                    "deeptech_projects": deeptech_projects,
                }
            )
        
        return {
            "status": validation.status.value,
            "message": validation.message,
            "suggestions": validation.suggestions,
            "results": results,
            "keywords": [(kw, float(score)) for kw, score in keywords.keywords] if keywords else [],
            "enhanced_query": match_query,
        }
    
    def match_simple(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.8,
        beta: float = 0.2,
        graph_neighbor_weight: float = 0.1,
    ) -> List[Dict[str, object]]:
        """
        Simple match without query validation (backward compatible).
        """
        query_vec = self.embedder.encode([query])
        semantic_scores = self._score_semantic(query_vec)
        priorities = np.array([record.priority_score for record in self.records])
        neighbor_scores = self._graph_neighbor_scores(semantic_scores)

        final_scores = (alpha * semantic_scores) + (beta * priorities) + (graph_neighbor_weight * neighbor_scores)
        deeptech_payloads: List[List[Dict[str, object]]] = []

        for idx, record in enumerate(self.records):
            payload = self._rank_deeptech_projects(record, query_vec)
            deeptech_payloads.append(payload)
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            record = self.records[idx]
            deeptech_projects = deeptech_payloads[idx]
            results.append(
                {
                    "name": record.name,
                    "department": record.department,
                    "title": record.title,
                    "url": record.url,
                    "research_interests": record.research_interests,
                    "score": float(final_scores[idx]),
                    "similarity": float(semantic_scores[idx]),
                    "priority_score": float(record.priority_score),
                    "deeptech_projects": deeptech_projects,
                }
            )
        return results
