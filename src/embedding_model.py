from __future__ import annotations

from typing import Iterable, List

import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
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

    def fit(self, texts: Iterable[str]) -> None:
        if self.backend == "tfidf" and self.vectorizer is not None:
            self.vectorizer.fit(list(texts))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if self.backend == "sentence-transformers" and self.model is not None:
            embeddings = self.model.encode(texts_list, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        if self.backend == "tfidf" and self.vectorizer is not None:
            return self.vectorizer.transform(texts_list).toarray()
        raise RuntimeError("No embedding backend available")
