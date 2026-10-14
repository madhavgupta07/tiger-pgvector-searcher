"""Embedding providers.

Every provider implements a tiny interface:

    class Embedder:
        dim: int
        name: str
        def embed(self, texts: list[str]) -> list[list[float]]: ...

Providers:
  * HashEmbedder            - deterministic, offline, zero-dependency fallback.
  * FastEmbedEmbedder       - local ONNX model via fastembed (default).
  * SentenceTransformerEmbedder - local PyTorch model.
  * OpenAIEmbedder          - hosted embeddings API.

Selection is driven by EMBEDDING_PROVIDER in the environment. `auto` prefers a
local model and degrades gracefully to the offline hash embedder so the whole
pipeline always runs, even with no network access.
"""

from __future__ import annotations

import hashlib
import math
import re

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Signed hashing of word uni/bi-grams into a fixed-size vector.

    Not a semantic model, but deterministic, fast, and fully offline. Useful as
    a demo default and as the last-resort fallback.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hash-{self.dim}d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        if not grams:
            grams = ["<empty>"]
        for gram in grams:
            h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str, dim: int):
        from fastembed import TextEmbedding  # lazy import

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    @property
    def name(self) -> str:
        return f"fastembed:{getattr(self._model, 'model_name', 'bge-small-en-v1.5')}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(texts)]


class SentenceTransformerEmbedder:
    """Local open-source sentence embeddings (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = dim

    @property
    def name(self) -> str:
        return self._model.__class__.__name__

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


class OpenAIEmbedder:
    """Hosted OpenAI embeddings (default: text-embedding-3-small, 1536-dim)."""

    def __init__(self, model_name: str, dim: int, api_key: str):
        from openai import OpenAI  # lazy import

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.dim = dim

    @property
    def name(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [list(d.embedding) for d in resp.data]


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    provider = config.EMBEDDING_PROVIDER
    dim = config.EMBEDDING_DIM

    if provider == "hash":
        _embedder = HashEmbedder(dim)
        return _embedder

    if provider == "openai":
        _embedder = OpenAIEmbedder(config.OPENAI_EMBEDDING_MODEL, dim, config.OPENAI_API_KEY)
        return _embedder

    if provider == "fastembed":
        _embedder = FastEmbedEmbedder(config.EMBEDDING_MODEL, dim)
        return _embedder

    if provider == "local":
        _embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL, dim)
        return _embedder

    # auto: prefer a local model, fall back to offline hash
    for factory in (FastEmbedEmbedder, SentenceTransformerEmbedder):
        try:
            _embedder = factory(config.EMBEDDING_MODEL, dim)
            return _embedder
        except Exception:
            continue
    _embedder = HashEmbedder(dim)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed(texts)


def embed_text(text: str) -> list[float]:
    return get_embedder().embed([text])[0]
