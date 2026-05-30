from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9+\-_/]*")


@dataclass(frozen=True)
class EvidenceChunk:
    record_index: int
    field: str
    text: str
    source_id: Optional[str] = None


@dataclass
class RecordScore:
    raw_score: float
    calibrated_score: float
    field_scores: Dict[str, float] = field(default_factory=dict)
    field_dense_scores: Dict[str, float] = field(default_factory=dict)
    field_lexical_scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)


@dataclass
class QueryScoreResult:
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray
    records: List[RecordScore]


def split_weighted_text(value: object, separator: str = "|") -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(separator) if part.strip()]


def compact_text(parts: Iterable[object]) -> str:
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip())


def build_professor_record_chunks(records: Sequence[object]) -> List[List[EvidenceChunk]]:
    all_chunks: List[List[EvidenceChunk]] = []
    for record_index, record in enumerate(records):
        chunks: List[EvidenceChunk] = []

        interests = str(getattr(record, "research_interests", "") or "").strip()
        if interests:
            chunks.append(EvidenceChunk(record_index=record_index, field="interests", text=interests))

        attributes = getattr(record, "attributes", {}) or {}
        for project in split_weighted_text(attributes.get("leading_project", "")):
            chunks.append(EvidenceChunk(record_index=record_index, field="project", text=project))

        for paper in split_weighted_text(attributes.get("paper", "")):
            chunks.append(EvidenceChunk(record_index=record_index, field="paper", text=paper))

        for project_index, project in enumerate(getattr(record, "deeptech_projects", []) or []):
            text = compact_text(
                [
                    getattr(project, "technology_title", ""),
                    getattr(project, "overview", ""),
                    getattr(project, "tech_edges", ""),
                    " ".join(getattr(project, "applications", []) or []),
                    " ".join(getattr(project, "industries", []) or []),
                ]
            )
            if text:
                chunks.append(
                    EvidenceChunk(
                        record_index=record_index,
                        field="deeptech",
                        text=text,
                        source_id=str(project_index),
                    )
                )

        if not chunks:
            chunks.append(
                EvidenceChunk(
                    record_index=record_index,
                    field="interests",
                    text="professor research expertise",
                )
            )
        all_chunks.append(chunks)
    return all_chunks


def build_startup_record_chunks(records: Sequence[object]) -> List[List[EvidenceChunk]]:
    all_chunks: List[List[EvidenceChunk]] = []
    for record_index, record in enumerate(records):
        chunks: List[EvidenceChunk] = []

        company = str(getattr(record, "company_name", "") or "").strip()
        if company:
            chunks.append(EvidenceChunk(record_index=record_index, field="company", text=company))

        categories = ", ".join(getattr(record, "categories", []) or []).strip()
        if categories:
            chunks.append(EvidenceChunk(record_index=record_index, field="category", text=categories))

        description = str(getattr(record, "description", "") or "").strip()
        if description:
            chunks.append(EvidenceChunk(record_index=record_index, field="description", text=description))

        meta = compact_text(
            [
                " ".join(getattr(record, "people", []) or []),
                getattr(record, "funding", ""),
                getattr(record, "background_year", ""),
            ]
        )
        if meta:
            chunks.append(EvidenceChunk(record_index=record_index, field="meta", text=meta))

        if not chunks:
            chunks.append(EvidenceChunk(record_index=record_index, field="description", text="startup technology project"))
        all_chunks.append(chunks)
    return all_chunks


def normalize_weights(weights: Mapping[str, float], defaults: Mapping[str, float]) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for key, default_value in defaults.items():
        raw_value = weights.get(key, default_value) if isinstance(weights, Mapping) else default_value
        try:
            raw[key] = max(0.0, float(raw_value))
        except (TypeError, ValueError):
            raw[key] = max(0.0, float(default_value))

    total = sum(raw.values())
    if total <= 0:
        fallback_total = sum(max(0.0, float(value)) for value in defaults.values())
        if fallback_total <= 0:
            return {key: 1.0 / max(1, len(defaults)) for key in defaults}
        return {key: max(0.0, float(value)) / fallback_total for key, value in defaults.items()}
    return {key: value / total for key, value in raw.items()}


def extract_query_terms(query: str) -> List[str]:
    terms: List[str] = []
    seen = set()
    for match in _WORD_RE.finditer(str(query or "").lower()):
        term = match.group(0).strip("-_/")
        if len(term) < 3 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


class HybridEvidenceScorer:
    def __init__(
        self,
        record_chunks: Sequence[Sequence[EvidenceChunk]],
        embedder,
        field_weights: Mapping[str, float],
        dense_weight: float = 0.75,
        lexical_weight: float = 0.25,
        top_k_chunks: int = 3,
        coverage_bonus: float = 0.08,
        calibration_enabled: bool = True,
        calibration_floor: float = 0.30,
        calibration_ceiling: float = 0.97,
        calibration_min_top_score: float = 0.12,
        weak_match_threshold: float = 0.08,
    ) -> None:
        self.record_chunks = [list(chunks) for chunks in record_chunks]
        self.embedder = embedder
        self.field_weights = dict(field_weights)
        self.fields = list(self.field_weights.keys())
        self.dense_weight = self._safe_float(dense_weight, 0.75)
        self.lexical_weight = self._safe_float(lexical_weight, 0.25)
        self.top_k_chunks = max(1, self._safe_int(top_k_chunks, 3))
        self.coverage_bonus = max(0.0, self._safe_float(coverage_bonus, 0.08))
        self.calibration_enabled = bool(calibration_enabled)
        self.calibration_floor = min(0.95, max(0.0, self._safe_float(calibration_floor, 0.30)))
        self.calibration_ceiling = min(1.0, max(0.01, self._safe_float(calibration_ceiling, 0.97)))
        self.calibration_min_top_score = max(0.0, self._safe_float(calibration_min_top_score, 0.12))
        self.weak_match_threshold = max(0.0, self._safe_float(weak_match_threshold, 0.08))

        total_mix = self.dense_weight + self.lexical_weight
        if total_mix <= 0:
            self.dense_weight, self.lexical_weight = 0.75, 0.25
        else:
            self.dense_weight /= total_mix
            self.lexical_weight /= total_mix

        self.chunks: List[EvidenceChunk] = [
            chunk
            for chunks in self.record_chunks
            for chunk in chunks
            if str(chunk.text or "").strip()
        ]
        self.chunk_texts = [chunk.text for chunk in self.chunks]
        self._chunk_embeddings = self._build_chunk_embeddings()
        self._lexical_vectorizer, self._lexical_matrix = self._build_lexical_matrix()
        self._record_profile_texts = self._build_record_profile_texts()
        self._query_score_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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

    def _build_chunk_embeddings(self) -> np.ndarray:
        if not self.chunk_texts:
            return np.empty((0, 0), dtype=float)

        # The TextEmbedder TF-IDF fallback is mutable and shared; using it for stored
        # dense vectors can drift when another catalog refits it. In that case the
        # internal lexical matrix becomes the retrieval backbone instead.
        if getattr(self.embedder, "backend", "") == "tfidf":
            return np.empty((0, 0), dtype=float)

        try:
            return self.embedder.encode(self.chunk_texts)
        except Exception:
            return np.empty((0, 0), dtype=float)

    def _build_lexical_matrix(self) -> Tuple[Optional[TfidfVectorizer], object]:
        if not self.chunk_texts:
            return None, None

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[\w+\-]{2,}\b",
        )
        try:
            matrix = vectorizer.fit_transform(self.chunk_texts)
        except ValueError:
            return None, None
        return vectorizer, matrix

    def _build_record_profile_texts(self) -> List[str]:
        profiles: List[str] = []
        for chunks in self.record_chunks:
            parts = [chunk.text for chunk in chunks if str(chunk.text or "").strip()]
            profiles.append(" . ".join(parts))
        return profiles

    def _dense_similarities(self, query: str) -> np.ndarray:
        if not self.chunk_texts:
            return np.zeros(0, dtype=float)

        if self._chunk_embeddings.size:
            try:
                query_vec = self.embedder.encode([query])
                return np.maximum(cosine_similarity(query_vec, self._chunk_embeddings)[0], 0.0)
            except Exception:
                return np.zeros(len(self.chunk_texts), dtype=float)

        if self._lexical_vectorizer is not None and self._lexical_matrix is not None:
            try:
                query_vec = self._lexical_vectorizer.transform([query])
                return np.maximum(cosine_similarity(query_vec, self._lexical_matrix)[0], 0.0)
            except Exception:
                return np.zeros(len(self.chunk_texts), dtype=float)

        return np.zeros(len(self.chunk_texts), dtype=float)

    def _lexical_similarities(self, query: str) -> np.ndarray:
        if not self.chunk_texts or self._lexical_vectorizer is None or self._lexical_matrix is None:
            return np.zeros(len(self.chunk_texts), dtype=float)

        try:
            query_vec = self._lexical_vectorizer.transform([query])
            tfidf_scores = np.maximum(cosine_similarity(query_vec, self._lexical_matrix)[0], 0.0)
        except Exception:
            tfidf_scores = np.zeros(len(self.chunk_texts), dtype=float)

        terms = extract_query_terms(query)
        if not terms:
            return tfidf_scores

        exact_scores = np.zeros(len(self.chunk_texts), dtype=float)
        for idx, text in enumerate(self.chunk_texts):
            lowered = text.lower()
            hits = sum(1 for term in terms if term in lowered)
            if hits:
                exact_scores[idx] = min(1.0, hits / max(3.0, float(len(terms))))

        return np.maximum(tfidf_scores, exact_scores)

    def _chunk_similarities(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = str(query or "")
        cached = self._query_score_cache.get(cache_key)
        if cached is not None:
            return cached

        scores = (self._dense_similarities(cache_key), self._lexical_similarities(cache_key))
        if len(self._query_score_cache) > 128:
            self._query_score_cache.clear()
        self._query_score_cache[cache_key] = scores
        return scores

    def _aggregate_field(
        self,
        chunk_indices: List[int],
        dense_scores: np.ndarray,
        lexical_scores: np.ndarray,
    ) -> Tuple[float, float, float, List[Tuple[str, float]]]:
        if not chunk_indices:
            return 0.0, 0.0, 0.0, []

        combined: List[Tuple[int, float, float, float]] = []
        for chunk_idx in chunk_indices:
            dense = float(dense_scores[chunk_idx]) if chunk_idx < len(dense_scores) else 0.0
            lexical = float(lexical_scores[chunk_idx]) if chunk_idx < len(lexical_scores) else 0.0
            score = (self.dense_weight * dense) + (self.lexical_weight * lexical)
            combined.append((chunk_idx, score, dense, lexical))

        combined.sort(key=lambda item: item[1], reverse=True)
        selected = combined[: self.top_k_chunks]
        if not selected:
            return 0.0, 0.0, 0.0, []

        weights = np.array([1.0 / (rank + 1) for rank in range(len(selected))], dtype=float)
        scores = np.array([item[1] for item in selected], dtype=float)
        dense_values = np.array([item[2] for item in selected], dtype=float)
        lexical_values = np.array([item[3] for item in selected], dtype=float)

        field_score = float(np.dot(scores, weights) / max(float(weights.sum()), 1e-9))
        dense_score = float(np.dot(dense_values, weights) / max(float(weights.sum()), 1e-9))
        lexical_score = float(np.dot(lexical_values, weights) / max(float(weights.sum()), 1e-9))
        evidence = [(self.chunks[item[0]].text, float(item[1])) for item in selected]
        return field_score, dense_score, lexical_score, evidence

    def score_query(self, query: str) -> QueryScoreResult:
        record_count = len(self.record_chunks)
        if record_count == 0:
            empty = np.zeros(0, dtype=float)
            return QueryScoreResult(raw_scores=empty, calibrated_scores=empty, records=[])

        dense_scores, lexical_scores = self._chunk_similarities(query)

        indices_by_record_field: List[Dict[str, List[int]]] = [
            {field_name: [] for field_name in self.fields}
            for _ in range(record_count)
        ]
        for chunk_idx, chunk in enumerate(self.chunks):
            if 0 <= chunk.record_index < record_count:
                indices_by_record_field[chunk.record_index].setdefault(chunk.field, []).append(chunk_idx)

        raw_scores = np.zeros(record_count, dtype=float)
        records: List[RecordScore] = []
        for record_idx in range(record_count):
            field_scores: Dict[str, float] = {}
            field_dense_scores: Dict[str, float] = {}
            field_lexical_scores: Dict[str, float] = {}
            evidence: Dict[str, List[Tuple[str, float]]] = {}
            matched_weight = 0.0
            weighted_score = 0.0

            for field_name, field_weight in self.field_weights.items():
                score, dense, lexical, field_evidence = self._aggregate_field(
                    indices_by_record_field[record_idx].get(field_name, []),
                    dense_scores,
                    lexical_scores,
                )
                field_scores[field_name] = score
                field_dense_scores[field_name] = dense
                field_lexical_scores[field_name] = lexical
                evidence[field_name] = field_evidence
                weighted_score += float(field_weight) * score
                if score >= self.weak_match_threshold:
                    matched_weight += float(field_weight)

            coverage = min(1.0, max(0.0, matched_weight))
            raw_score = min(1.0, max(0.0, weighted_score + (self.coverage_bonus * coverage * weighted_score)))
            raw_scores[record_idx] = raw_score
            records.append(
                RecordScore(
                    raw_score=raw_score,
                    calibrated_score=raw_score,
                    field_scores=field_scores,
                    field_dense_scores=field_dense_scores,
                    field_lexical_scores=field_lexical_scores,
                    evidence=evidence,
                )
            )

        calibrated = self.calibrate_scores(raw_scores)
        for idx, score in enumerate(calibrated):
            records[idx].calibrated_score = float(score)
        return QueryScoreResult(raw_scores=raw_scores, calibrated_scores=calibrated, records=records)

    def calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        if not self.calibration_enabled or scores.size == 0:
            return scores

        top = float(np.max(scores))
        if top < max(self.weak_match_threshold, self.calibration_min_top_score):
            return scores

        relative = np.clip(scores / max(top, 1e-9), 0.0, 1.0)
        relative_scaled = self.calibration_ceiling * np.power(relative, 0.85)
        calibrated = (0.55 * relative_scaled) + (0.45 * scores)

        meaningful = scores >= self.weak_match_threshold
        if self.calibration_floor > 0:
            calibrated = np.where(meaningful, np.maximum(calibrated, self.calibration_floor), calibrated)

        calibrated = np.where(scores > 0, calibrated, 0.0)
        return np.clip(calibrated, 0.0, self.calibration_ceiling)

    def top_evidence(self, record_index: int, limit: int = 6) -> List[Tuple[str, str, float]]:
        if record_index < 0 or record_index >= len(self.record_chunks):
            return []
        items: List[Tuple[str, str, float]] = []
        for chunk in self.record_chunks[record_index]:
            items.append((chunk.field, chunk.text, 0.0))
        return items[: max(0, int(limit))]

    def field_source_scores(self, query: str, record_index: int, field: str) -> Dict[str, float]:
        if record_index < 0 or record_index >= len(self.record_chunks):
            return {}

        dense_scores, lexical_scores = self._chunk_similarities(query)
        scores: Dict[str, float] = {}
        for chunk_idx, chunk in enumerate(self.chunks):
            if chunk.record_index != record_index or chunk.field != field:
                continue
            source_id = chunk.source_id if chunk.source_id is not None else str(chunk_idx)
            dense = float(dense_scores[chunk_idx]) if chunk_idx < len(dense_scores) else 0.0
            lexical = float(lexical_scores[chunk_idx]) if chunk_idx < len(lexical_scores) else 0.0
            score = (self.dense_weight * dense) + (self.lexical_weight * lexical)
            scores[source_id] = max(scores.get(source_id, 0.0), float(score))
        return scores

    def profile_similarity_matrix(self) -> np.ndarray:
        count = len(self._record_profile_texts)
        if count == 0:
            return np.zeros((0, 0), dtype=float)

        if getattr(self.embedder, "backend", "") != "tfidf":
            try:
                embeddings = self.embedder.encode(self._record_profile_texts)
                dense_matrix = np.maximum(cosine_similarity(embeddings), 0.0)
            except Exception:
                dense_matrix = np.zeros((count, count), dtype=float)
        else:
            dense_matrix = np.zeros((count, count), dtype=float)

        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            lexical = vectorizer.fit_transform(self._record_profile_texts)
            lexical_matrix = np.maximum(cosine_similarity(lexical), 0.0)
        except ValueError:
            lexical_matrix = np.zeros((count, count), dtype=float)

        if dense_matrix.any():
            return (self.dense_weight * dense_matrix) + (self.lexical_weight * lexical_matrix)
        return lexical_matrix
