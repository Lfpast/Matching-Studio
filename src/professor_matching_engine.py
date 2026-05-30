from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np

from .embedding_model import TextEmbedder
from .hybrid_evidence_scorer import (
    HybridEvidenceScorer,
    QueryScoreResult,
    build_professor_record_chunks,
    normalize_weights,
)
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
        self.name_to_index = {record.name: idx for idx, record in enumerate(self.records)}
        self.scorer = self._build_scorer()

        self._domain_texts = self._build_domain_texts()
        if self.embedder.backend == "tfidf" and self.embedder.vectorizer is not None and self._domain_texts:
            self.embedder.fit(self._domain_texts)
        self._domain_embeddings = self.embedder.encode(self._domain_texts) if self._domain_texts else None

        self.query_processor = EnhancedQueryProcessor(
            embedder=self.embedder,
            domain_embeddings=self._domain_embeddings,
            domain_texts=self._domain_texts,
            similarity_threshold=self.query_config.get("similarity_threshold", 0.25),
            weak_threshold=self.query_config.get("weak_threshold", 0.35),
        )

    def _build_domain_texts(self) -> List[str]:
        return [record.research_interests for record in self.records if record.research_interests]

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _load_semantic_weights(self) -> Dict[str, float]:
        defaults = {
            "interests": 0.25,
            "project": 0.20,
            "paper": 0.20,
            "deeptech": 0.35,
        }
        configured = self.semantic_config.get("field_weights", {}) if isinstance(self.semantic_config, dict) else {}
        return normalize_weights(configured, defaults)

    def _build_scorer(self) -> HybridEvidenceScorer:
        return HybridEvidenceScorer(
            record_chunks=build_professor_record_chunks(self.records),
            embedder=self.embedder,
            field_weights=self.semantic_weights,
            dense_weight=self._safe_float(self.semantic_config.get("dense_weight", 0.75), 0.75),
            lexical_weight=self._safe_float(self.semantic_config.get("lexical_weight", 0.25), 0.25),
            top_k_chunks=self._safe_int(self.semantic_config.get("top_k_chunks", 3), 3),
            coverage_bonus=self._safe_float(self.semantic_config.get("coverage_bonus", 0.08), 0.08),
            calibration_enabled=bool(self.semantic_config.get("calibration_enabled", True)),
            calibration_floor=self._safe_float(self.semantic_config.get("calibration_floor", 0.30), 0.30),
            calibration_ceiling=self._safe_float(self.semantic_config.get("calibration_ceiling", 0.97), 0.97),
            calibration_min_top_score=self._safe_float(self.semantic_config.get("calibration_min_top_score", 0.12), 0.12),
            weak_match_threshold=self._safe_float(self.semantic_config.get("weak_match_threshold", 0.08), 0.08),
        )

    def _score_semantic(self, query: str) -> QueryScoreResult:
        return self.scorer.score_query(query)

    def _rank_deeptech_projects(self, record: ProfessorRecord, query: str) -> List[Dict[str, object]]:
        if not record.deeptech_projects:
            return []

        record_idx = self.name_to_index.get(record.name, -1)
        project_scores = self.scorer.field_source_scores(query, record_idx, "deeptech")
        ranked = sorted(
            enumerate(record.deeptech_projects),
            key=lambda item: float(project_scores.get(str(item[0]), 0.0)),
            reverse=True,
        )

        projects_payload: List[Dict[str, object]] = []
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
                    "relevance_score": float(project_scores.get(str(idx), 0.0)),
                }
            )
        return projects_payload

    def _graph_neighbor_scores(self, semantic_scores: np.ndarray) -> np.ndarray:
        if not self.graph:
            return np.zeros_like(semantic_scores)

        neighbor_scores = np.zeros_like(semantic_scores)
        for record in self.records:
            idx = self.name_to_index.get(record.name)
            if idx is None or not self.graph.has_node(record.name):
                continue
            neighbors = list(self.graph.neighbors(record.name))
            if not neighbors:
                continue
            values = [semantic_scores[self.name_to_index[n]] for n in neighbors if n in self.name_to_index]
            if values:
                neighbor_scores[idx] = float(np.mean(values))
        return neighbor_scores

    @staticmethod
    def _combine_final_scores(
        relevance_scores: np.ndarray,
        priority_scores: np.ndarray,
        neighbor_scores: np.ndarray,
        alpha: float,
        beta: float,
        graph_neighbor_weight: float,
    ) -> np.ndarray:
        alpha_weight = max(0.0, float(alpha))
        beta_weight = max(0.0, float(beta))
        graph_weight = max(0.0, float(graph_neighbor_weight))
        total_weight = alpha_weight + beta_weight + graph_weight
        if total_weight <= 0:
            return np.clip(relevance_scores, 0.0, 1.0)

        combined = (
            (alpha_weight * relevance_scores)
            + (beta_weight * relevance_scores * priority_scores)
            + (graph_weight * neighbor_scores)
        ) / total_weight
        return np.clip(combined, 0.0, 1.0)

    def _match_validated(
        self,
        query: str,
        top_k: int,
        alpha: float,
        beta: float,
        graph_neighbor_weight: float,
    ) -> List[Dict[str, object]]:
        if not self.records:
            return []

        score_result = self._score_semantic(query)
        semantic_scores = np.clip(score_result.calibrated_scores, 0.0, 1.0)
        priorities = np.array([record.priority_score for record in self.records], dtype=float)
        neighbor_scores = self._graph_neighbor_scores(semantic_scores)

        relevance_scores = np.clip(semantic_scores, 0.0, 1.0)
        final_scores = self._combine_final_scores(
            relevance_scores=relevance_scores,
            priority_scores=np.clip(priorities, 0.0, 1.0),
            neighbor_scores=np.clip(neighbor_scores, 0.0, 1.0),
            alpha=alpha,
            beta=beta,
            graph_neighbor_weight=graph_neighbor_weight,
        )
        ranked_indices = np.argsort(final_scores)[::-1][: max(1, int(top_k))]

        results: List[Dict[str, object]] = []
        for idx in ranked_indices:
            record = self.records[int(idx)]
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
                    "deeptech_projects": self._rank_deeptech_projects(record, query),
                }
            )
        return results

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
        if validate_query:
            enhanced_query, validation, keywords = self.query_processor.get_enhanced_query(query)
        else:
            validation = QueryValidationResult(
                status=QueryStatus.VALID,
                message="Query validation skipped.",
                suggestions=[],
                confidence=1.0,
            )
            keywords = None
            enhanced_query = query

        if validation.status == QueryStatus.INVALID:
            return {
                "status": validation.status.value,
                "message": validation.message,
                "suggestions": validation.suggestions,
                "results": [],
                "keywords": [],
                "enhanced_query": query,
            }

        match_query = enhanced_query if use_keyword_extraction and keywords and keywords.filtered_query else query
        results = self._match_validated(
            query=match_query,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            graph_neighbor_weight=graph_neighbor_weight,
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
        return self._match_validated(
            query=query,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            graph_neighbor_weight=graph_neighbor_weight,
        )
