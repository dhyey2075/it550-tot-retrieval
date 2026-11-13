"""
Tier B dense stage: score BM25 candidates only (E2-R / E5-R).

Not full-corpus ANN search.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DocTextFn = Callable[[str], str]


class DenseReranker:
    def __init__(
        self,
        model_name: str,
        *,
        prefix_query: str = "",
        prefix_doc: str = "",
        max_seq_length: int | None = 256,
        device: str | None = None,
        batch_size: int = 64,
    ):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model_name
        self.prefix_query = prefix_query
        self.prefix_doc = prefix_doc
        self.batch_size = batch_size
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        # If offline or in sandbox, load directly from local cache without network retries
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        texts = [self.prefix_query + q for q in queries]
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        texts = [self.prefix_doc + t for t in texts]
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(emb, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidate_docids: Sequence[str],
        doc_text_fn: DocTextFn,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Re-score BM25 candidates with bi-encoder cosine similarity.

        `doc_text_fn(docid) -> text` must resolve candidate text (corpus lookup).
        """
        if not candidate_docids:
            return []
        docids = [str(d) for d in candidate_docids]
        texts = [doc_text_fn(d) for d in docids]
        q_emb = self.encode_queries([query])[0]
        d_emb = self.encode_docs(texts)
        scores = d_emb @ q_emb  # cosine if L2-normalized
        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(docids[i], float(scores[i])) for i in order]


# Suggested bake-off shortlist (D-03); freeze after Dev1.
MODEL_PRESETS = {
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "prefix_query": "",
        "prefix_doc": "",
        "max_seq_length": 256,
    },
    "e5-small": {
        "model": "intfloat/e5-small-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "max_seq_length": 256,
    },
    "e5-base": {
        "model": "intfloat/e5-base-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "max_seq_length": 256,
    },
    "bge-small": {
        "model": "BAAI/bge-small-en-v1.5",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "max_seq_length": 256,
    },
}


def build_dense_reranker(preset: str = "e5-small", **overrides: Any) -> DenseReranker:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset {preset}; choose from {sorted(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[preset], **overrides}
    return DenseReranker(
        cfg["model"],
        prefix_query=cfg.get("prefix_query", ""),
        prefix_doc=cfg.get("prefix_doc", ""),
        max_seq_length=cfg.get("max_seq_length"),
        batch_size=int(cfg.get("batch_size", 64)),
        device=cfg.get("device"),
    )
