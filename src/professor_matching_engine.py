from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from .professor_preprocessing import ProfessorRecord, build_professor_text
from .embedding_model import TextEmbedder
from .query_processor import (
    EnhancedQueryProcessor,
    QueryStatus,
    QueryValidationResult,
    ExtractedKeywords,
)


class MatchingEngine:
    def __init__(
        self,
        records: Iterable[ProfessorRecord],
        embedder: TextEmbedder,
        graph: Optional[nx.Graph] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        query_config: Optional[Dict] = None,
    ) -> None:
        self.records = list(records)
        self.embedder = embedder
        self.graph = graph
        self.attribute_weights = attribute_weights or {}
        self.query_config = query_config or {}
        self.name_to_index = {record.name: idx for idx, record in enumerate(self.records)}
        self.embeddings = self._build_embeddings()
        
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

    def _build_embeddings(self) -> np.ndarray:
        texts = [build_professor_text(record, self.attribute_weights) for record in self.records]
        if self.embedder.backend == "tfidf":
            self.embedder.fit(texts)
        return self.embedder.encode(texts)

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

    def _graph_neighbor_scores(self, similarities: np.ndarray) -> np.ndarray:
        if not self.graph:
            return np.zeros_like(similarities)

        neighbor_scores = np.zeros_like(similarities)
        for record in self.records:
            idx = self.name_to_index.get(record.name)
            if idx is None:
                continue
            neighbors = list(self.graph.neighbors(record.name))
            if not neighbors:
                continue
            neighbor_sims = [similarities[self.name_to_index[n]] for n in neighbors if n in self.name_to_index]
            if neighbor_sims:
                neighbor_scores[idx] = float(np.mean(neighbor_sims))
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
        
        # Perform the matching
        query_vec = self.embedder.encode([match_query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        priorities = np.array([record.priority_score for record in self.records])
        neighbor_scores = self._graph_neighbor_scores(similarities)

        base_scores = (alpha * similarities) + (beta * priorities) + (graph_neighbor_weight * neighbor_scores)
        deeptech_payloads: List[List[Dict[str, object]]] = []
        deeptech_boosts = np.zeros(len(self.records), dtype=float)

        for idx, record in enumerate(self.records):
            payload = self._rank_deeptech_projects(record, query_vec)
            deeptech_payloads.append(payload)
            if payload:
                top_relevance = max(project["relevance_score"] for project in payload)
                deeptech_boosts[idx] = 0.5 * float(top_relevance)

        final_scores = base_scores + deeptech_boosts
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
                    "similarity": float(similarities[idx]),
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
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        priorities = np.array([record.priority_score for record in self.records])
        neighbor_scores = self._graph_neighbor_scores(similarities)

        base_scores = (alpha * similarities) + (beta * priorities) + (graph_neighbor_weight * neighbor_scores)
        deeptech_payloads: List[List[Dict[str, object]]] = []
        deeptech_boosts = np.zeros(len(self.records), dtype=float)

        for idx, record in enumerate(self.records):
            payload = self._rank_deeptech_projects(record, query_vec)
            deeptech_payloads.append(payload)
            if payload:
                top_relevance = max(project["relevance_score"] for project in payload)
                deeptech_boosts[idx] = 0.5 * float(top_relevance)

        final_scores = base_scores + deeptech_boosts
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
                    "similarity": float(similarities[idx]),
                    "priority_score": float(record.priority_score),
                    "deeptech_projects": deeptech_projects,
                }
            )
        return results
