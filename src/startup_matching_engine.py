from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import re

import networkx as nx
import numpy as np

from .embedding_model import TextEmbedder
from .hybrid_evidence_scorer import (
    HybridEvidenceScorer,
    QueryScoreResult,
    build_startup_record_chunks,
    normalize_weights,
)
from .query_processor import EnhancedQueryProcessor, QueryStatus, QueryValidationResult
from .startup_preprocessing import StartupRecord


_KEYWORD_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "yet", "so",
    "for", "from", "with", "without", "into", "onto", "to", "of", "in", "on", "at", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "should",
    "would", "will", "may", "might", "must", "this", "that", "these", "those", "it", "its", "their",
    "there", "here", "what", "which", "when", "where", "who", "whom", "how", "why", "if", "then", "than",
    "和", "及", "与", "以及", "并且", "或者",
}


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
        self.keyword_cfg = self.semantic_cfg.get("keyword_matching", {}) if isinstance(self.semantic_cfg, dict) else {}
        self.semantic_weights = self._load_semantic_weights()
        self.keyword_similarity_threshold = self._safe_float(
            self.keyword_cfg.get("similarity_threshold", 0.24),
            0.24,
        )
        self.keyword_weight_threshold = self._safe_float(
            self.keyword_cfg.get("query_weight_threshold", 0.45),
            0.45,
        )
        self.keyword_max_count = max(1, self._safe_int(self.keyword_cfg.get("max_keywords", 6), 6))
        self.id_to_index = {record.startup_id: idx for idx, record in enumerate(self.records)}
        self.scorer = self._build_scorer()

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
            "company": 0.05,
            "category": 0.35,
            "description": 0.55,
            "meta": 0.05,
        }
        configured = self.semantic_cfg.get("field_weights", {}) if isinstance(self.semantic_cfg, dict) else {}
        fallback = self.embedding_weights if isinstance(self.embedding_weights, dict) else {}

        merged = {
            "company": configured.get("company", fallback.get("company_name", defaults["company"])),
            "category": configured.get("category", fallback.get("category", defaults["category"])),
            "description": configured.get("description", fallback.get("description", defaults["description"])),
            "meta": configured.get("meta", fallback.get("meta", defaults["meta"])),
        }
        return normalize_weights(merged, defaults)

    def _build_scorer(self) -> HybridEvidenceScorer:
        return HybridEvidenceScorer(
            record_chunks=build_startup_record_chunks(self.records),
            embedder=self.embedder,
            field_weights=self.semantic_weights,
            dense_weight=self._safe_float(self.semantic_cfg.get("dense_weight", 0.75), 0.75),
            lexical_weight=self._safe_float(self.semantic_cfg.get("lexical_weight", 0.25), 0.25),
            top_k_chunks=self._safe_int(self.semantic_cfg.get("top_k_chunks", 3), 3),
            coverage_bonus=self._safe_float(self.semantic_cfg.get("coverage_bonus", 0.08), 0.08),
            calibration_enabled=bool(self.semantic_cfg.get("calibration_enabled", True)),
            calibration_floor=self._safe_float(self.semantic_cfg.get("calibration_floor", 0.30), 0.30),
            calibration_ceiling=self._safe_float(self.semantic_cfg.get("calibration_ceiling", 0.97), 0.97),
            calibration_min_top_score=self._safe_float(self.semantic_cfg.get("calibration_min_top_score", 0.12), 0.12),
            weak_match_threshold=self._safe_float(self.semantic_cfg.get("weak_match_threshold", 0.08), 0.08),
        )

    def _score_semantic(self, query: str) -> QueryScoreResult:
        return self.scorer.score_query(query)

    def _graph_neighbor_scores(self, base_scores: np.ndarray) -> np.ndarray:
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
            values = [base_scores[self.id_to_index[n]] for n in neighbors if n in self.id_to_index]
            if values:
                neighbor_scores[idx] = float(np.mean(values))

        return neighbor_scores

    @staticmethod
    def _combine_final_scores(
        relevance_scores: np.ndarray,
        neighbor_scores: np.ndarray,
        alpha: float,
        graph_neighbor_weight: float,
    ) -> np.ndarray:
        alpha_weight = max(0.0, float(alpha))
        graph_weight = max(0.0, float(graph_neighbor_weight))
        total_weight = alpha_weight + graph_weight
        if total_weight <= 0:
            return np.clip(relevance_scores, 0.0, 1.0)

        combined = ((alpha_weight * relevance_scores) + (graph_weight * neighbor_scores)) / total_weight
        return np.clip(combined, 0.0, 1.0)

    def _extract_highlight_keywords(self, query: str, use_keyword_extraction: bool) -> List[Tuple[str, float]]:
        if not use_keyword_extraction or not self.query_processor:
            return []

        try:
            extracted = self.query_processor.keyword_extractor.extract(query)
            return [(kw, float(score)) for kw, score in extracted.keywords]
        except Exception:
            return []

    @staticmethod
    def _normalize_keyword_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _is_valid_keyword_token(self, token: str) -> bool:
        normalized = self._normalize_keyword_text(token)
        if not normalized:
            return False

        lowered = normalized.lower()
        if lowered in _KEYWORD_STOPWORDS:
            return False

        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", normalized))
        if has_cjk:
            return len(normalized) >= 2
        return len(lowered) >= 3

    def _extract_keyword_candidates(
        self,
        query: str,
        extracted,
        use_keyword_extraction: bool,
    ) -> List[Tuple[str, float]]:
        candidates: Dict[str, Tuple[str, float]] = {}

        if use_keyword_extraction and extracted is not None:
            for raw_keyword, raw_score in extracted.keywords:
                token = self._normalize_keyword_text(raw_keyword)
                if not self._is_valid_keyword_token(token):
                    continue

                score = self._safe_float(raw_score, 0.0)
                if score < self.keyword_weight_threshold:
                    continue

                key = token.lower()
                previous = candidates.get(key)
                if previous is None or score > previous[1]:
                    candidates[key] = (token, score)

        if candidates:
            ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
            return ranked[: max(self.keyword_max_count * 2, self.keyword_max_count)]

        source_query = query
        if use_keyword_extraction and extracted is not None and extracted.filtered_query:
            source_query = extracted.filtered_query

        for raw in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", str(source_query)):
            token = self._normalize_keyword_text(raw)
            if not self._is_valid_keyword_token(token):
                continue
            key = token.lower()
            if key not in candidates:
                candidates[key] = (token, 0.4)

        ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
        return ranked[: max(self.keyword_max_count * 2, self.keyword_max_count)]

    def _record_keyword_similarity(self, record_idx: int, token: str) -> float:
        if record_idx < 0 or record_idx >= len(self.records):
            return 0.0

        try:
            score_result = self.scorer.score_query(token)
        except Exception:
            return 0.0
        if record_idx >= len(score_result.raw_scores):
            return 0.0
        return max(0.0, float(score_result.raw_scores[record_idx]))

    def _build_display_keywords(
        self,
        query: str,
        extracted,
        use_keyword_extraction: bool,
        ranked_indices: np.ndarray,
    ) -> List[Tuple[str, float]]:
        candidates = self._extract_keyword_candidates(
            query=query,
            extracted=extracted,
            use_keyword_extraction=use_keyword_extraction,
        )
        if not candidates:
            return []

        focus_indices = [int(idx) for idx in ranked_indices[: max(1, min(6, len(ranked_indices)))]]
        scored: List[Tuple[str, float]] = []
        for token, base_weight in candidates:
            best_similarity = 0.0
            for record_idx in focus_indices:
                best_similarity = max(best_similarity, self._record_keyword_similarity(record_idx, token))
            scored.append((token, (0.65 * best_similarity) + (0.35 * float(base_weight))))

        filtered = [(token, score) for token, score in scored if score >= self.keyword_similarity_threshold]
        filtered.sort(key=lambda item: item[1], reverse=True)
        if filtered:
            return filtered[: self.keyword_max_count]

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: min(2, len(scored))]

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

        record_idx = self.id_to_index.get(record.startup_id)
        if record_idx is None:
            return []

        searchable_text = " ".join(
            [
                record.company_name,
                record.description,
                " ".join(record.categories),
            ]
        ).lower()

        matched: Dict[str, Tuple[str, float]] = {}
        for keyword, base_weight in keywords:
            token = self._normalize_keyword_text(keyword)
            if not self._is_valid_keyword_token(token):
                continue

            semantic_score = self._record_keyword_similarity(record_idx, token)
            score = semantic_score + (0.2 * float(base_weight))
            if token.lower() in searchable_text:
                score += 0.1
            elif semantic_score < (self.keyword_similarity_threshold + 0.04):
                continue

            if score < self.keyword_similarity_threshold:
                continue

            key = token.lower()
            previous = matched.get(key)
            if previous is None or score > previous[1]:
                matched[key] = (token, score)

        ranked = sorted(matched.values(), key=lambda item: item[1], reverse=True)
        return [token for token, _score in ranked[: self.keyword_max_count]]

    def _collect_result_highlight_keywords(
        self,
        startup_results: List[Dict[str, object]],
        fallback_keywords: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        fallback_map = {
            self._normalize_keyword_text(keyword).lower(): float(score)
            for keyword, score in fallback_keywords
            if self._is_valid_keyword_token(keyword)
        }

        merged: Dict[str, Tuple[str, float]] = {}
        for result in startup_results:
            for raw_token in result.get("matched_keywords", []) or []:
                token = self._normalize_keyword_text(raw_token)
                if not self._is_valid_keyword_token(token):
                    continue

                key = token.lower()
                score = fallback_map.get(key, self.keyword_similarity_threshold)
                previous = merged.get(key)
                if previous is None or score > previous[1]:
                    merged[key] = (token, score)

        if merged:
            ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
            return ranked[: self.keyword_max_count]
        return fallback_keywords[: self.keyword_max_count]

    def _format_result_item(
        self,
        record: StartupRecord,
        score: float,
        keywords: List[Tuple[str, float]],
    ) -> Dict[str, object]:
        return {
            "startup_id": record.startup_id,
            "company_name": record.company_name,
            "website": record.website,
            "people": record.people,
            "ref_code": record.ref_code,
            "ref_code_link": record.ref_code_link,
            "categories": record.categories,
            "source_year": record.source_year,
            "description": record.description,
            "tels": record.tels,
            "emails": record.emails,
            "funding": record.funding,
            "background_year": record.background_year,
            "matched_keywords": self._build_matched_keywords(record, keywords),
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

        score_result = self._score_semantic(match_query)
        semantic_scores = np.clip(score_result.calibrated_scores, 0.0, 1.0)

        graph_scores = self._graph_neighbor_scores(semantic_scores)
        final_scores = self._combine_final_scores(
            relevance_scores=semantic_scores,
            neighbor_scores=np.clip(graph_scores, 0.0, 1.0),
            alpha=alpha,
            graph_neighbor_weight=graph_neighbor_weight,
        ) + (float(beta) * 0.0)
        final_scores = np.clip(final_scores, 0.0, 1.0)
        ranked_indices = np.argsort(final_scores)[::-1][: max(1, int(top_k))]

        keyword_payload = self._build_display_keywords(
            query=query,
            extracted=extracted,
            use_keyword_extraction=use_keyword_extraction,
            ranked_indices=ranked_indices,
        )

        startup_results = [
            self._format_result_item(
                record=self.records[int(idx)],
                score=float(final_scores[idx]),
                keywords=keyword_payload,
            )
            for idx in ranked_indices
        ]
        keyword_payload = self._collect_result_highlight_keywords(
            startup_results=startup_results,
            fallback_keywords=keyword_payload,
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
