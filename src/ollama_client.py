from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, Iterable, List, Sequence
from urllib import error, request

import numpy as np

from .env_loader import load_project_env


load_project_env()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


class OllamaAPIError(RuntimeError):
    """Raised when an Ollama API call cannot be completed."""


@dataclass(frozen=True)
class OllamaSettings:
    host: str = "https://ollama.com"
    api_key_env: str = "OLLAMA_API_KEY"
    chat_model: str = "gemma4:cloud"
    embedding_model: str = "embeddinggemma"
    timeout_seconds: int = 60
    rerank_candidate_count: int = 12

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OllamaSettings":
        ollama_cfg = config.get("ollama", {}) if isinstance(config, dict) else {}
        timeout_value = ollama_cfg.get("timeout_seconds", 60)
        rerank_value = ollama_cfg.get("rerank_candidate_count", 12)

        try:
            timeout_seconds = max(5, int(timeout_value))
        except (TypeError, ValueError):
            timeout_seconds = 60

        try:
            rerank_candidate_count = max(5, int(rerank_value))
        except (TypeError, ValueError):
            rerank_candidate_count = 12

        host = str(ollama_cfg.get("host", "https://ollama.com")).strip() or "https://ollama.com"
        api_key_env = str(ollama_cfg.get("api_key_env", "OLLAMA_API_KEY")).strip() or "OLLAMA_API_KEY"
        chat_model = str(ollama_cfg.get("chat_model", ollama_cfg.get("llm_model", "gemma4:cloud"))).strip() or "gemma4:cloud"
        embedding_model = str(ollama_cfg.get("embedding_model", "embeddinggemma")).strip() or "embeddinggemma"

        return cls(
            host=host,
            api_key_env=api_key_env,
            chat_model=chat_model,
            embedding_model=embedding_model,
            timeout_seconds=timeout_seconds,
            rerank_candidate_count=rerank_candidate_count,
        )


class OllamaClient:
    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings
        self.api_base = self._build_api_base(settings.host)
        self._resolved_chat_model = settings.chat_model
        self._resolved_embedding_model = settings.embedding_model
        self._cached_remote_models: List[str] | None = None

    @staticmethod
    def _build_api_base(host: str) -> str:
        normalized = str(host or "https://ollama.com").strip().rstrip("/")
        if not normalized:
            normalized = "https://ollama.com"
        if normalized.endswith("/api"):
            return normalized
        return f"{normalized}/api"

    def _is_local_host(self) -> bool:
        lower = self.api_base.lower()
        return "localhost" in lower or "127.0.0.1" in lower or "0.0.0.0" in lower

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._is_local_host():
            return headers

        api_key = os.environ.get(self.settings.api_key_env, "").strip()
        if not api_key:
            raise OllamaAPIError(
                f"Missing {self.settings.api_key_env} for Ollama cloud access. "
                "Please configure the API key or switch to a local signed-in Ollama host."
            )

        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _candidate_models(model_name: str) -> List[str]:
        normalized = str(model_name or "").strip()
        if not normalized:
            return []

        candidates: List[str] = [normalized]
        if normalized.endswith(":cloud"):
            candidates.append(normalized[: -len(":cloud")])
        if normalized.endswith("-cloud"):
            candidates.append(normalized[: -len("-cloud")])

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if not candidate or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _extract_error_message(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return "Unknown Ollama API error"

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ("error", "detail", "message"):
                    value = payload.get(key)
                    if value:
                        return str(value)
        except json.JSONDecodeError:
            pass

        return text

    @staticmethod
    def _looks_like_auth_error(message: str) -> bool:
        lowered = str(message or "").lower()
        return "unauthorized" in lowered or "forbidden" in lowered or "api key" in lowered

    @staticmethod
    def _looks_like_model_name_error(message: str) -> bool:
        lowered = str(message or "").lower()
        return "not found" in lowered or "unknown model" in lowered or "model" in lowered and "missing" in lowered

    def _get_json(self, path: str) -> Dict[str, Any]:
        endpoint = f"{self.api_base}{path}"
        request_obj = request.Request(
            endpoint,
            headers=self._build_headers(),
            method="GET",
        )

        try:
            with request.urlopen(request_obj, timeout=self.settings.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    return parsed
                raise OllamaAPIError("Unexpected JSON payload returned by Ollama API")
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            raise OllamaAPIError(self._extract_error_message(response_text)) from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaAPIError(f"Unable to reach Ollama API: {reason}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaAPIError(f"Invalid JSON returned by Ollama API: {exc}") from exc

    def _list_remote_models(self) -> List[str]:
        if self._cached_remote_models is not None:
            return self._cached_remote_models

        payload = self._get_json("/tags")
        models = payload.get("models", []) if isinstance(payload, dict) else []

        names: List[str] = []
        seen = set()
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict):
                continue
            name = str(model.get("model") or model.get("name") or "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            names.append(name)

        self._cached_remote_models = names
        return names

    @staticmethod
    def _choose_family_model(requested_model: str, available_models: Sequence[str]) -> str | None:
        requested = str(requested_model or "").strip().lower()
        if not requested:
            return None

        base_name = requested
        for suffix in (":cloud", "-cloud"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        exact_prefix = f"{base_name}:"
        family_matches = [model for model in available_models if str(model).lower().startswith(exact_prefix)]
        if family_matches:
            def sort_key(model_name: str) -> tuple[int, int, str]:
                lowered = model_name.lower()
                size_match = re.search(r":(\d+)([bmkt])", lowered)
                if not size_match:
                    return (0, 0, lowered)

                magnitude = int(size_match.group(1))
                unit = size_match.group(2)
                unit_rank = {"b": 1, "m": 0, "k": 2, "t": 3}.get(unit, 0)
                return (unit_rank, magnitude, lowered)

            return sorted(family_matches, key=sort_key, reverse=True)[0]

        generic_matches = [model for model in available_models if str(model).lower() == base_name]
        if generic_matches:
            return generic_matches[0]

        return None

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = f"{self.api_base}{path}"
        payload_copy = dict(payload)
        original_model = str(payload_copy.get("model", "")).strip()

        if original_model:
            candidates = self._candidate_models(original_model)
        else:
            candidates = [""]

        last_error: Exception | None = None
        for candidate in candidates:
            if candidate:
                payload_copy["model"] = candidate
            elif "model" in payload_copy:
                payload_copy.pop("model", None)

            data = json.dumps(payload_copy, ensure_ascii=False).encode("utf-8")
            request_obj = request.Request(
                endpoint,
                data=data,
                headers=self._build_headers(),
                method="POST",
            )

            try:
                with request.urlopen(request_obj, timeout=self.settings.timeout_seconds) as response:
                    response_text = response.read().decode("utf-8")
                    parsed = json.loads(response_text)
                    if path == "/chat" and candidate:
                        self._resolved_chat_model = candidate
                    if path == "/embed" and candidate:
                        self._resolved_embedding_model = candidate
                    return parsed
            except error.HTTPError as exc:
                response_text = exc.read().decode("utf-8", errors="replace")
                message = self._extract_error_message(response_text)
                last_error = OllamaAPIError(message)
                if self._looks_like_model_name_error(message) and not self._is_local_host():
                    try:
                        resolved = self._choose_family_model(original_model, self._list_remote_models())
                    except OllamaAPIError:
                        resolved = None
                    if resolved and resolved.lower() not in {candidate.lower() for candidate in candidates if candidate}:
                        candidates.append(resolved)
                        continue
                if len(candidates) > 1 and self._looks_like_model_name_error(message):
                    continue
                break
            except error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                last_error = OllamaAPIError(f"Unable to reach Ollama API: {reason}")
                break
            except json.JSONDecodeError as exc:
                last_error = OllamaAPIError(f"Invalid JSON returned by Ollama API: {exc}")
                break

        if last_error is not None:
            raise last_error
        raise OllamaAPIError("Unknown Ollama API failure")

    @staticmethod
    def _parse_json_value(content: Any) -> Any:
        if isinstance(content, (dict, list)):
            return content

        text = str(content or "").strip()
        if not text:
            raise OllamaAPIError("Ollama returned an empty structured response")

        text = _CODE_FENCE_RE.sub("", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        raise OllamaAPIError("Failed to parse structured JSON from Ollama response")

    @classmethod
    def _parse_json_content(cls, content: Any) -> Dict[str, Any]:
        parsed = cls._parse_json_value(content)
        if isinstance(parsed, dict):
            return parsed
        raise OllamaAPIError("Expected a JSON object from Ollama response")

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "model": self._resolved_chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": "0",
            "options": {
                "temperature": 0.1,
            },
        }
        response = self._post_json("/chat", payload)
        content = response.get("message", {}).get("content", "")
        return self._parse_json_content(content)

    def chat_json_value(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        format_spec: Any = "json",
    ) -> Any:
        payload = {
            "model": self._resolved_chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": format_spec,
            "stream": False,
            "keep_alive": "0",
            "options": {
                "temperature": 0.1,
            },
        }
        response = self._post_json("/chat", payload)
        content = response.get("message", {}).get("content", "")
        return self._parse_json_value(content)

    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> np.ndarray:
        text_list = [str(text or "") for text in texts]
        if not text_list:
            return np.empty((0, 0), dtype=float)

        payload = {
            "model": model_name or self._resolved_embedding_model,
            "input": text_list,
            "truncate": True,
            "keep_alive": "0",
        }
        response = self._post_json("/embed", payload)
        embeddings = response.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise OllamaAPIError("Ollama embed response is missing embeddings")

        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2:
            raise OllamaAPIError("Ollama embed response returned an invalid vector shape")
        return array

    def probe(self, include_embeddings: bool = True) -> None:
        schema = {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "message": {"type": "string"},
            },
            "required": ["ok", "message"],
        }
        response = self.chat_json(
            system_prompt="You are a connectivity probe. Return concise JSON only.",
            user_prompt="Respond with {\"ok\": true, \"message\": \"ready\"}.",
            schema=schema,
        )
        if not bool(response.get("ok")):
            raise OllamaAPIError(str(response.get("message") or "Ollama chat probe failed"))

        if include_embeddings:
            self.embed_texts(["expert mode connectivity probe"])


class OllamaTextEmbedder:
    def __init__(self, client: OllamaClient, model_name: str | None = None, fallback_embedder: Any | None = None) -> None:
        self.client = client
        self.model_name = model_name or client.settings.embedding_model
        self.backend = "ollama"
        self.vectorizer = None
        self.fallback_embedder = fallback_embedder
        self._remote_embeddings_enabled = True
        self._fallback_reason = ""

    def fit(self, _texts: Iterable[str]) -> None:
        if self.fallback_embedder is not None and hasattr(self.fallback_embedder, "fit"):
            self.fallback_embedder.fit(_texts)
        return None

    def _fallback_encode(self, texts: List[str], error_message: str | None = None) -> np.ndarray:
        if error_message:
            self._fallback_reason = str(error_message)

        if self.fallback_embedder is None:
            raise OllamaAPIError(self._fallback_reason or "Ollama embeddings unavailable and no fallback embedder configured")

        return self.fallback_embedder.encode(texts)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not self._remote_embeddings_enabled:
            return self._fallback_encode(text_list)

        try:
            return self.client.embed_texts(text_list, model_name=self.model_name)
        except OllamaAPIError as exc:
            message = str(exc)
            if self.client._looks_like_auth_error(message) or self.client._looks_like_model_name_error(message):
                self._remote_embeddings_enabled = False
                return self._fallback_encode(text_list, message)
            raise