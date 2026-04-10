from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_model import TextEmbedder
from .query_processor import EnhancedQueryProcessor, QueryStatus, QueryValidationResult
from .startup_preprocessing import StartupRecord


class StartupMatchingEngine:
    def __init__(
        self,
        records: Iterable[StartupRecord],
        embedder: TextEmbedder,
        graph: Optional[nx.Graph],
        query_processor: Optional[EnhancedQueryProcessor],
        config: Optional[Dict],
    ) -> None:
        self.records = list(records)
        self.embedder = embedder
        self.graph = graph
        self.query_processor = query_processor
        self.config = config or {}
        self.embedding_weights = self.config.get("embedding_weights", {})
        self.semantic_cfg = self.config.get("semantic_matching", {}) if isinstance(self.config, dict) else {}
        self.semantic_weights = self._load_semantic_weights()
        self.min_field_similarity = float(self.semantic_cfg.get("min_field_similarity", 0.08))
        self.id_to_index = {record.startup_id: idx for idx, record in enumerate(self.records)}
        (
            self.company_embeddings,
            self.description_embeddings,
            self.category_embeddings,
        ) = self._build_field_embeddings()

    def _load_semantic_weights(self) -> Dict[str, float]:
        defaults = {
            "company_name": 0.28,
            "description": 0.52,
            "category": 0.20,
        }

        configured_weights = self.semantic_cfg.get("field_weights", {}) if isinstance(self.semantic_cfg, dict) else {}
        if not isinstance(configured_weights, dict):
            configured_weights = {}

        fallback_weights = self.embedding_weights if isinstance(self.embedding_weights, dict) else {}

        raw_weights: Dict[str, float] = {}
        for key, default_value in defaults.items():
            raw_value = configured_weights.get(key, fallback_weights.get(key, default_value))
            try:
                raw_weights[key] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                raw_weights[key] = default_value

        total = sum(raw_weights.values())
        if total <= 0:
            return defaults

        return {key: (value / total) for key, value in raw_weights.items()}

    def _build_field_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.records:
            empty = np.empty((0, 0), dtype=float)
            return empty, empty, empty

        company_texts = [str(record.company_name).strip() or "startup company name" for record in self.records]
        description_texts = [str(record.description).strip() or "startup brief description" for record in self.records]
        category_texts = [", ".join(record.categories).strip() or "startup category" for record in self.records]

        if self.embedder.backend == "tfidf" and self.embedder.vectorizer is not None:
            # Reuse an already-fitted shared vectorizer when possible, avoiding professor embedding drift.
            if not hasattr(self.embedder.vectorizer, "vocabulary_"):
                self.embedder.fit([*company_texts, *description_texts, *category_texts])

        return (
            self.embedder.encode(company_texts),
            self.embedder.encode(description_texts),
            self.embedder.encode(category_texts),
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

        company_scores = self._score_field_similarity(query_vec, self.company_embeddings)
        description_scores = self._score_field_similarity(query_vec, self.description_embeddings)
        category_scores = self._score_field_similarity(query_vec, self.category_embeddings)

        w_company = self.semantic_weights["company_name"]
        w_description = self.semantic_weights["description"]
        w_category = self.semantic_weights["category"]

        weighted_scores = (
            (w_company * company_scores)
            + (w_description * description_scores)
            + (w_category * category_scores)
        )

        field_scores = np.vstack((company_scores, description_scores, category_scores))
        weight_vector = np.array([w_company, w_description, w_category], dtype=float).reshape(-1, 1)
        coverage = (
            ((field_scores >= self.min_field_similarity).astype(float) * weight_vector).sum(axis=0)
            / max(float(weight_vector.sum()), 1e-9)
        )

        # Favor records that match across multiple semantic fields (stricter relevance).
        strict_scores = weighted_scores * (0.55 + (0.45 * coverage))
        return np.maximum(strict_scores, 0.0)

    def _score_graph_boost(self, base_scores: np.ndarray, graph_neighbor_weight: float) -> np.ndarray:
        if not self.graph or not self.records:
            return np.zeros(len(self.records), dtype=float)

        neighbor_scores = np.zeros(len(self.records), dtype=float)
        for record in self.records:
            idx = self.id_to_index.get(record.startup_id)
            if idx is None or not self.graph.has_node(record.startup_id):
                continue

            neighbors = list(self.graph.neighbors(record.startup_id))
            if not neighbors:
                continue

            vals = [base_scores[self.id_to_index[n]] for n in neighbors if n in self.id_to_index]
            if vals:
                neighbor_scores[idx] = float(np.mean(vals))

        return graph_neighbor_weight * neighbor_scores

    def _extract_highlight_keywords(self, query: str, use_keyword_extraction: bool) -> List[Tuple[str, float]]:
        if not use_keyword_extraction or not self.query_processor:
            return []

        try:
            extracted = self.query_processor.keyword_extractor.extract(query)
            return [(kw, float(score)) for kw, score in extracted.keywords]
        except Exception:
            return []

    def _build_match_query(
        self,
        query: str,
        enhanced_query: str,
        extracted,
        use_keyword_extraction: bool,
    ) -> str:
        if not use_keyword_extraction or extracted is None:
            return query

        base_query = enhanced_query if extracted.filtered_query else query
        top_keywords = [keyword for keyword, score in extracted.keywords if float(score) >= 0.45][:6]
        if not top_keywords:
            return base_query

        existing_tokens = {token.strip().lower() for token in str(base_query).split() if token.strip()}
        appended_tokens = [token for token in top_keywords if token.strip().lower() not in existing_tokens]
        if not appended_tokens:
            return base_query

        return f"{base_query} {' '.join(appended_tokens)}".strip()

    def _build_matched_keywords(
        self,
        record: StartupRecord,
        keywords: List[Tuple[str, float]],
    ) -> List[str]:
        if not keywords:
            return []

        company_text = str(record.company_name or "").lower()
        description_text = str(record.description or "").lower()
        category_text = " ".join(record.categories or []).lower()

        matched: Dict[str, Tuple[str, float]] = {}
        for keyword, base_weight in keywords:
            token = str(keyword or "").strip()
            if not token:
                continue

            normalized = token.lower()
            score = float(base_weight)
            is_matched = False

            if normalized in description_text:
                score += 0.25
                is_matched = True
            if normalized in company_text:
                score += 0.2
                is_matched = True
            if normalized in category_text:
                score += 0.15
                is_matched = True

            if not is_matched:
                continue

            previous = matched.get(normalized)
            if previous is None or score > previous[1]:
                matched[normalized] = (token, score)

        ranked = sorted(matched.values(), key=lambda item: item[1], reverse=True)
        return [token for token, _score in ranked[:8]]

    def _format_result_item(
        self,
        record: StartupRecord,
        score: float,
        keywords: List[Tuple[str, float]],
    ) -> Dict[str, object]:
        matched_keywords = self._build_matched_keywords(record, keywords)
        return {
            "startup_id": record.startup_id,
            "company_name": record.company_name,
            "website": record.website,
            "people": record.people,
            "ref_code": record.ref_code,
            "categories": record.categories,
            "source_year": record.source_year,
            "description": record.description,
            "tels": record.tels,
            "emails": record.emails,
            "funding": record.funding,
            "background_year": record.background_year,
            "matched_keywords": matched_keywords,
            "score": float(score),
        }

    def match(
        self,
        query: str,
        top_k: int,
        alpha: float,
        beta: float,
        graph_neighbor_weight: float,
        validate_query: bool,
        use_keyword_extraction: bool,
    ) -> Dict[str, object]:
        if validate_query and self.query_processor is not None:
            enhanced_query, validation, extracted = self.query_processor.get_enhanced_query(query)
        else:
            validation = QueryValidationResult(
                status=QueryStatus.VALID,
                message="Query validation skipped.",
                suggestions=[],
                confidence=1.0,
            )
            extracted = None
            enhanced_query = query

        if validation.status == QueryStatus.INVALID:
            return {
                "status": validation.status.value,
                "message": validation.message,
                "suggestions": validation.suggestions,
                "results": [],
                "startup_results": [],
                "keywords": [],
                "enhanced_query": query,
            }

        match_query = self._build_match_query(
            query=query,
            enhanced_query=enhanced_query,
            extracted=extracted,
            use_keyword_extraction=use_keyword_extraction,
        )

        if not self.records:
            keyword_payload = self._extract_highlight_keywords(query, use_keyword_extraction)
            return {
                "status": validation.status.value,
                "message": validation.message,
                "suggestions": validation.suggestions,
                "results": [],
                "startup_results": [],
                "keywords": keyword_payload,
                "enhanced_query": match_query,
            }

        query_vec = self.embedder.encode([match_query])
        semantic_scores = self._score_semantic(query_vec)

        # Keep beta for forward compatibility; startup priority is intentionally disabled.
        base_scores = (alpha * semantic_scores) + (beta * 0.0)
        graph_boost = self._score_graph_boost(semantic_scores, graph_neighbor_weight)
        final_scores = base_scores + graph_boost

        ranked_indices = np.argsort(final_scores)[::-1][: max(1, int(top_k))]

        if use_keyword_extraction and extracted is not None:
            keyword_payload = [(kw, float(score)) for kw, score in extracted.keywords]
        else:
            keyword_payload = self._extract_highlight_keywords(query, use_keyword_extraction)

        startup_results: List[Dict[str, object]] = []
        for idx in ranked_indices:
            startup_results.append(
                self._format_result_item(
                    record=self.records[idx],
                    score=float(final_scores[idx]),
                    keywords=keyword_payload,
                )
            )

        return {
            "status": validation.status.value,
            "message": validation.message,
            "suggestions": validation.suggestions,
            "results": [],
            "startup_results": startup_results,
            "keywords": keyword_payload,
            "enhanced_query": match_query,
        }
