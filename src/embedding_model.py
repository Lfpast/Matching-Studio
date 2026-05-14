from __future__ import annotations

from typing import Iterable, List

import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 4) -> None:
        self.model_name = model_name
        self.batch_size = self._normalize_batch_size(batch_size)
        self.backend = "sentence-transformers"
        self.model = None
        self.vectorizer = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception:
            self.backend = "tfidf"
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(stop_words="english")

    @staticmethod
    def _normalize_batch_size(batch_size: int) -> int:
        try:
            return max(1, int(batch_size))
        except (TypeError, ValueError):
            return 4

    def _iter_batches(self, texts_list: List[str]) -> Iterable[List[str]]:
        step = max(1, self.batch_size)
        for start in range(0, len(texts_list), step):
            yield texts_list[start:start + step]

    @staticmethod
    def _stack_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        non_empty = [np.atleast_2d(batch) for batch in embeddings if getattr(batch, "size", 0) > 0]
        if not non_empty:
            return np.empty((0, 0), dtype=float)
        if len(non_empty) == 1:
            return non_empty[0]
        return np.vstack(non_empty)

    def fit(self, texts: Iterable[str]) -> None:
        if self.backend == "tfidf" and self.vectorizer is not None:
            self.vectorizer.fit(list(texts))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.empty((0, 0), dtype=float)
        if self.backend == "sentence-transformers" and self.model is not None:
            embeddings = [
                self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
                for batch_texts in self._iter_batches(texts_list)
            ]
            return self._stack_embeddings(embeddings)
        if self.backend == "tfidf" and self.vectorizer is not None:
            embeddings = [self.vectorizer.transform(batch_texts).toarray() for batch_texts in self._iter_batches(texts_list)]
            return self._stack_embeddings(embeddings)
        raise RuntimeError("No embedding backend available")
