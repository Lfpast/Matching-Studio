from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .embedding_model import TextEmbedder
from .ollama_client import OllamaAPIError, OllamaClient, OllamaSettings, OllamaTextEmbedder
from .professor_graph_builder import build_graph
from .professor_matching_engine import MatchingEngine
from .professor_preprocessing import ProfessorRecord
from .query_processor import ExtractedKeywords, QueryStatus, QueryValidationResult
from .startup_graph_builder import build_startup_graph
from .startup_matching_engine import StartupMatchingEngine
from .startup_preprocessing import StartupRecord


class ExpertModeError(RuntimeError):
    """Raised when the Ollama-backed expert pipeline is unavailable."""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate_text(value: Any, limit: int = 320) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}..."


def _clean_token_list(values: Any, limit: int = 8) -> List[str]:
    if not isinstance(values, list):
        return []

    deduped: List[str] = []
    seen = set()
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
        if len(deduped) >= limit:
            break
    return deduped


class ExpertModeService:
    def __init__(
        self,
        professor_records: List[ProfessorRecord],
        startup_records: List[StartupRecord],
        config: Dict[str, Any],
    ) -> None:
        self.professor_records = list(professor_records)
        self.startup_records = list(startup_records)
        self.config = config or {}
        self.settings = OllamaSettings.from_config(self.config)
        self.client = OllamaClient(self.settings)
        self.fallback_embedder = self._build_fallback_embedder(self.config)
        self.embedder = OllamaTextEmbedder(
            self.client,
            self.settings.embedding_model,
            fallback_embedder=self.fallback_embedder,
            batch_size=self._embedding_batch_size(self.config),
        )
        self.professor_by_name = {record.name: record for record in self.professor_records}
        self.startup_by_id = {record.startup_id: record for record in self.startup_records}
        self.professor_engine: Optional[MatchingEngine] = None
        self.startup_engine: Optional[StartupMatchingEngine] = None

    @staticmethod
    def _build_fallback_embedder(config: Dict[str, Any]) -> TextEmbedder:
        embedding_cfg = config.get("embedding", {}) if isinstance(config, dict) else {}
        model_name = str(embedding_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2")).strip()
        batch_size = embedding_cfg.get("batch_size", 4)
        if not model_name:
            model_name = "sentence-transformers/all-mpnet-base-v2"
        return TextEmbedder(model_name=model_name, batch_size=batch_size)

    @staticmethod
    def _embedding_batch_size(config: Dict[str, Any]) -> int:
        embedding_cfg = config.get("embedding", {}) if isinstance(config, dict) else {}
        return embedding_cfg.get("batch_size", 4)

    def refresh(
        self,
        professor_records: List[ProfessorRecord],
        startup_records: List[StartupRecord],
        config: Dict[str, Any],
    ) -> None:
        self.professor_records = list(professor_records)
        self.startup_records = list(startup_records)
        self.config = config or {}
        self.settings = OllamaSettings.from_config(self.config)
        self.client = OllamaClient(self.settings)
        self.fallback_embedder = self._build_fallback_embedder(self.config)
        self.embedder = OllamaTextEmbedder(
            self.client,
            self.settings.embedding_model,
            fallback_embedder=self.fallback_embedder,
            batch_size=self._embedding_batch_size(self.config),
        )
        self.professor_by_name = {record.name: record for record in self.professor_records}
        self.startup_by_id = {record.startup_id: record for record in self.startup_records}
        self.professor_engine = None
        self.startup_engine = None

    def uses_embedding_fallback(self) -> bool:
        return not getattr(self.embedder, "_remote_embeddings_enabled", True)

    def active_chat_model(self) -> str:
        return str(getattr(self.client, "_resolved_chat_model", "") or self.settings.chat_model).strip()

    def backend_label(self) -> str:
        chat_model = self.active_chat_model()
        if self.uses_embedding_fallback():
            return f"Ollama · {chat_model} + MPNet retrieval"
        return f"Ollama · {chat_model}"

    def status_text(self) -> str:
        return "EXPERT · LLM ready"

    def probe_and_warm(self) -> None:
        try:
            self.client.probe(include_embeddings=False)
            self._ensure_engines()
        except OllamaAPIError as exc:
            raise ExpertModeError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive conversion for endpoint safety
            raise ExpertModeError(f"Failed to initialize expert mode: {exc}") from exc

    def _ensure_engines(self) -> None:
        if self.professor_engine is not None and self.startup_engine is not None:
            return

        professor_cfg = self.config.get("professor", {}) if isinstance(self.config, dict) else {}
        startup_cfg = self.config.get("startup", {}) if isinstance(self.config, dict) else {}
        query_cfg = self.config.get("query", {}) if isinstance(self.config, dict) else {}

        professor_graph_cfg = professor_cfg.get("graph", {}) if isinstance(professor_cfg, dict) else {}
        professor_graph = build_graph(
            self.professor_records,
            embedder=self.embedder,
            similarity_threshold=_safe_float(professor_graph_cfg.get("similarity_threshold", 0.2), 0.2),
            interests_weight=_safe_float(professor_graph_cfg.get("interests_weight", 0.2), 0.2),
            project_weight=_safe_float(professor_graph_cfg.get("project_weight", 0.1), 0.1),
            paper_weight=_safe_float(professor_graph_cfg.get("paper_weight", 0.3), 0.3),
            deeptech_weight=_safe_float(professor_graph_cfg.get("deeptech_weight", 0.4), 0.4),
        )

        professor_semantic_cfg = professor_cfg.get("semantic_matching", {}) if isinstance(professor_cfg, dict) else {}
        self.professor_engine = MatchingEngine(
            records=self.professor_records,
            embedder=self.embedder,
            graph=professor_graph,
            query_config=query_cfg,
            semantic_config=professor_semantic_cfg,
        )

        startup_graph_cfg = startup_cfg.get("graph", {}) if isinstance(startup_cfg, dict) else {}
        category_weight = _safe_float(startup_graph_cfg.get("category_weight", 0.6), 0.6)
        description_weight = _safe_float(startup_graph_cfg.get("description_weight", 0.4), 0.4)
        total_weight = category_weight + description_weight
        if total_weight <= 0:
            category_weight, description_weight = 0.6, 0.4
        else:
            category_weight = category_weight / total_weight
            description_weight = description_weight / total_weight

        startup_graph = build_startup_graph(
            self.startup_records,
            embedder=self.embedder,
            similarity_threshold=_safe_float(startup_graph_cfg.get("similarity_threshold", 0.2), 0.2),
            category_weight=category_weight,
            description_weight=description_weight,
        )

        self.startup_engine = StartupMatchingEngine(
            records=self.startup_records,
            embedder=self.embedder,
            graph=startup_graph,
            query_processor=self.professor_engine.query_processor,
            config=startup_cfg,
        )

    def _default_validation(self) -> QueryValidationResult:
        return QueryValidationResult(
            status=QueryStatus.VALID,
            message="Query validation skipped.",
            suggestions=[],
            confidence=1.0,
        )

    def _validate_query(
        self,
        *,
        query: str,
        target: str,
        validate_query: bool,
    ) -> Tuple[QueryValidationResult, Optional[ExtractedKeywords]]:
        if not validate_query:
            return self._default_validation(), None

        processor = None
        if target == "startup" and self.startup_engine is not None:
            processor = self.startup_engine.query_processor
        elif self.professor_engine is not None:
            processor = self.professor_engine.query_processor

        if processor is None:
            return self._default_validation(), None

        return processor.process(query)

    @staticmethod
    def _query_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "normalized_query": {"type": "string"},
                "intent_summary": {"type": "string"},
                "technology_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "application_scenarios": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "target_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "normalized_query",
                "intent_summary",
                "technology_keywords",
                "application_scenarios",
                "constraints",
                "target_entities",
            ],
        }

    def _fallback_query_payload(
        self,
        query: str,
        extracted: Optional[ExtractedKeywords],
        target: str,
    ) -> Dict[str, Any]:
        extracted_keywords = []
        if extracted is not None:
            extracted_keywords = [keyword for keyword, _score in extracted.keywords[:6]]

        normalized_query = extracted.filtered_query if extracted and extracted.filtered_query else query
        return {
            "normalized_query": normalized_query,
            "intent_summary": f"{target} matching request",
            "technology_keywords": extracted_keywords,
            "application_scenarios": [],
            "constraints": [],
            "target_entities": [target],
        }

    def _parse_query(
        self,
        *,
        query: str,
        target: str,
        extracted: Optional[ExtractedKeywords],
    ) -> Dict[str, Any]:
        extracted_keywords = []
        if extracted is not None:
            extracted_keywords = [keyword for keyword, _score in extracted.keywords[:6]]

        fallback = self._fallback_query_payload(query, extracted, target)
        system_prompt = (
            "You structure enterprise technology matching queries for retrieval and reranking. "
            "Return strict JSON only. Keep keywords short, technical, and non-redundant. "
            "Do not invent requirements that are not implied by the user query."
        )
        user_prompt = json.dumps(
            {
                "target_catalog": target,
                "query": query,
                "existing_keywords": extracted_keywords,
                "required_fields": [
                    "normalized_query",
                    "intent_summary",
                    "technology_keywords",
                    "application_scenarios",
                    "constraints",
                    "target_entities",
                ],
            },
            ensure_ascii=False,
        )

        try:
            response = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self._query_schema(),
            )
        except Exception:
            return fallback

        normalized_query = str(response.get("normalized_query", fallback["normalized_query"]) or fallback["normalized_query"]).strip()
        intent_summary = str(response.get("intent_summary", fallback["intent_summary"]) or fallback["intent_summary"]).strip()
        technology_keywords = _clean_token_list(response.get("technology_keywords"), limit=8) or fallback["technology_keywords"]
        application_scenarios = _clean_token_list(response.get("application_scenarios"), limit=6)
        constraints = _clean_token_list(response.get("constraints"), limit=6)
        target_entities = _clean_token_list(response.get("target_entities"), limit=4) or [target]

        return {
            "normalized_query": normalized_query or fallback["normalized_query"],
            "intent_summary": intent_summary or fallback["intent_summary"],
            "technology_keywords": technology_keywords,
            "application_scenarios": application_scenarios,
            "constraints": constraints,
            "target_entities": target_entities,
        }

    @staticmethod
    def _build_match_query(parsed_query: Dict[str, Any], original_query: str) -> str:
        parts: List[str] = []

        normalized_query = str(parsed_query.get("normalized_query", "")).strip()
        if normalized_query:
            parts.append(normalized_query)

        intent_summary = str(parsed_query.get("intent_summary", "")).strip()
        if intent_summary:
            parts.append(f"Intent: {intent_summary}")

        technology_keywords = _clean_token_list(parsed_query.get("technology_keywords"), limit=8)
        if technology_keywords:
            parts.append(f"Technology focus: {', '.join(technology_keywords)}")

        application_scenarios = _clean_token_list(parsed_query.get("application_scenarios"), limit=6)
        if application_scenarios:
            parts.append(f"Applications: {', '.join(application_scenarios)}")

        constraints = _clean_token_list(parsed_query.get("constraints"), limit=6)
        if constraints:
            parts.append(f"Constraints: {', '.join(constraints)}")

        return " | ".join(parts) if parts else original_query

    @staticmethod
    def _merge_keywords(
        parsed_query: Dict[str, Any],
        extracted: Optional[ExtractedKeywords],
    ) -> List[Tuple[str, float]]:
        keyword_payload: List[Tuple[str, float]] = []
        seen = set()

        parsed_tokens = _clean_token_list(parsed_query.get("technology_keywords"), limit=8)
        parsed_tokens.extend(_clean_token_list(parsed_query.get("application_scenarios"), limit=6))

        for index, token in enumerate(parsed_tokens):
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            keyword_payload.append((token, max(0.5, 1.0 - (index * 0.08))))

        if extracted is not None:
            for keyword, weight in extracted.keywords[:8]:
                token = str(keyword or "").strip()
                if not token:
                    continue
                key = token.lower()
                if key in seen:
                    continue
                seen.add(key)
                keyword_payload.append((token, _safe_float(weight, 0.4)))

        return keyword_payload[:8]

    @staticmethod
    def _rerank_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ranked_candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_id", "relevance_score", "reason"],
                    },
                }
            },
            "required": ["ranked_candidates"],
        }

    @staticmethod
    def _normalize_rerank_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            candidates = payload.get("ranked_candidates", [])
            return candidates if isinstance(candidates, list) else []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _professor_summary(self, result: Dict[str, Any]) -> str:
        record = self.professor_by_name.get(str(result.get("name", "")))
        if record is None:
            return _truncate_text(result.get("research_interests", ""), limit=420)

        deeptech_titles = [project.technology_title for project in record.deeptech_projects[:3] if project.technology_title]
        deeptech_overview = " ".join(project.overview for project in record.deeptech_projects[:2] if project.overview)

        parts = [
            f"Name: {record.name}",
            f"Department: {record.department}",
            f"Title: {record.title}",
            f"Research interests: {_truncate_text(record.research_interests, 260)}",
            f"Leading project: {_truncate_text(record.attributes.get('leading_project', ''), 180)}",
            f"Publications: {_truncate_text(record.attributes.get('paper', ''), 180)}",
        ]
        if deeptech_titles:
            parts.append(f"DeepTech titles: {', '.join(deeptech_titles)}")
        if deeptech_overview:
            parts.append(f"DeepTech overview: {_truncate_text(deeptech_overview, 200)}")
        return "\n".join(part for part in parts if part and not part.endswith(": "))

    def _startup_summary(self, result: Dict[str, Any]) -> str:
        record = self.startup_by_id.get(str(result.get("startup_id", "")))
        if record is None:
            return _truncate_text(result.get("description", ""), limit=420)

        parts = [
            f"Company: {record.company_name}",
            f"Categories: {', '.join(record.categories[:6])}",
            f"Description: {_truncate_text(record.description, 260)}",
            f"People: {_truncate_text('; '.join(record.people[:6]), 160)}",
            f"Funding: {_truncate_text(record.funding, 120)}",
            f"Background: {_truncate_text(record.background_year, 120)}",
        ]
        return "\n".join(part for part in parts if part and not part.endswith(": "))

    def _rerank_results(
        self,
        *,
        target: str,
        original_query: str,
        parsed_query: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        candidate_id_key: str,
        summary_builder,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        candidate_payloads = []
        for candidate in candidates:
            candidate_id = str(candidate.get(candidate_id_key, "")).strip()
            if not candidate_id:
                continue
            candidate_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "summary": summary_builder(candidate),
                }
            )

        if not candidate_payloads:
            return candidates

        system_prompt = (
            "You rerank technology-matching candidates for enterprise collaboration. "
            "Use only the provided summaries. Favor technical fit, application alignment, and practical collaboration value. "
            "Return strict JSON only."
        )
        user_prompt = json.dumps(
            {
                "target_catalog": target,
                "original_query": original_query,
                "structured_query": parsed_query,
                "candidates": candidate_payloads,
                "instructions": "Return a JSON array sorted best-first. Each item must have candidate_id, score, and reason. Keep reasons under 180 characters.",
            },
            ensure_ascii=False,
        )

        try:
            response = self.client.chat_json_value(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                format_spec="json",
            )
        except Exception:
            return candidates

        ranked_items = self._normalize_rerank_items(response)
        if not ranked_items:
            return candidates

        by_id = {str(candidate.get(candidate_id_key, "")): candidate for candidate in candidates}
        reranked: List[Dict[str, Any]] = []
        seen = set()
        for item in ranked_items:
            candidate_id = str(item.get("candidate_id", "")).strip()
            if not candidate_id or candidate_id not in by_id or candidate_id in seen:
                continue

            candidate = dict(by_id[candidate_id])
            base_score = _safe_float(candidate.get("score"), 0.0)
            rerank_raw_score = item.get("relevance_score", item.get("score", base_score))
            rerank_score = min(1.0, max(0.0, _safe_float(rerank_raw_score, base_score)))
            candidate["score"] = float((0.72 * base_score) + (0.28 * rerank_score))
            candidate["expert_reason"] = _truncate_text(item.get("reason", ""), limit=180)
            reranked.append(candidate)
            seen.add(candidate_id)

        for candidate in candidates:
            candidate_id = str(candidate.get(candidate_id_key, "")).strip()
            if candidate_id in seen:
                continue
            reranked.append(candidate)

        return reranked

    def match_professor(
        self,
        *,
        query: str,
        top_k: int,
        alpha: float,
        beta: float,
        graph_neighbor_weight: float,
        validate_query: bool,
        use_keyword_extraction: bool,
    ) -> Dict[str, Any]:
        self._ensure_engines()
        validation, extracted = self._validate_query(query=query, target="professor", validate_query=validate_query)
        if validation.status == QueryStatus.INVALID:
            return {
                "status": validation.status.value,
                "message": validation.message,
                "suggestions": validation.suggestions,
                "results": [],
                "keywords": [],
                "enhanced_query": query,
            }

        parsed_query = self._parse_query(query=query, target="professor", extracted=extracted if use_keyword_extraction else None)
        match_query = self._build_match_query(parsed_query, query)
        candidate_count = max(max(1, int(top_k)), int(self.settings.rerank_candidate_count))

        assert self.professor_engine is not None
        base_result = self.professor_engine.match(
            query=match_query,
            top_k=candidate_count,
            alpha=alpha,
            beta=beta,
            graph_neighbor_weight=graph_neighbor_weight,
            validate_query=False,
            use_keyword_extraction=False,
        )
        candidates = list(base_result.get("results", []))
        reranked = self._rerank_results(
            target="professor",
            original_query=query,
            parsed_query=parsed_query,
            candidates=candidates,
            candidate_id_key="name",
            summary_builder=self._professor_summary,
        )

        return {
            "status": validation.status.value,
            "message": validation.message,
            "suggestions": validation.suggestions,
            "results": reranked[: max(1, int(top_k))],
            "keywords": self._merge_keywords(parsed_query, extracted if use_keyword_extraction else None),
            "enhanced_query": match_query,
        }

    def match_startup(
        self,
        *,
        query: str,
        top_k: int,
        graph_neighbor_weight: float,
        validate_query: bool,
        use_keyword_extraction: bool,
    ) -> Dict[str, Any]:
        self._ensure_engines()
        validation, extracted = self._validate_query(query=query, target="startup", validate_query=validate_query)
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

        parsed_query = self._parse_query(query=query, target="startup", extracted=extracted if use_keyword_extraction else None)
        match_query = self._build_match_query(parsed_query, query)
        candidate_count = max(max(1, int(top_k)), int(self.settings.rerank_candidate_count))

        assert self.startup_engine is not None
        base_result = self.startup_engine.match(
            query=match_query,
            top_k=candidate_count,
            alpha=1.0,
            beta=0.0,
            graph_neighbor_weight=graph_neighbor_weight,
            validate_query=False,
            use_keyword_extraction=False,
        )
        candidates = list(base_result.get("startup_results", []))
        reranked = self._rerank_results(
            target="startup",
            original_query=query,
            parsed_query=parsed_query,
            candidates=candidates,
            candidate_id_key="startup_id",
            summary_builder=self._startup_summary,
        )

        return {
            "status": validation.status.value,
            "message": validation.message,
            "suggestions": validation.suggestions,
            "results": [],
            "startup_results": reranked[: max(1, int(top_k))],
            "keywords": self._merge_keywords(parsed_query, extracted if use_keyword_extraction else None),
            "enhanced_query": match_query,
        }