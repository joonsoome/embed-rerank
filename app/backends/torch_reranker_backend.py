"""
PyTorch Cross-Encoder Reranker Backend

Implements cross-encoder scoring using sentence-transformers CrossEncoder.
Prefers MPS on Apple Silicon when available, falls back to CPU otherwise.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from .base import BaseBackend, EmbeddingResult


def _pick_torch_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchCrossEncoderBackend(BaseBackend):
    """Cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: Optional[int] = None):
        super().__init__(model_name=model_name, device=device)
        self.cross_encoder: Optional[CrossEncoder] = None
        self._device = _pick_torch_device(device)
        self._batch_size = batch_size or 16

    async def load_model(self) -> None:
        if self._is_loaded:
            return
        try:
            # CrossEncoder will place model on the given device automatically
            self.cross_encoder = CrossEncoder(self.model_name, device=self._device)
            self._is_loaded = True
            self._load_time = 0.0  # not measured precisely here
        except Exception as e:
            raise RuntimeError(f"Failed to load CrossEncoder model '{self.model_name}': {e}")

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Cross-encoder models are not embedding models; provide a minimal placeholder
        to satisfy interface in rare cases where this is called inadvertently.
        """
        # Produce deterministic zero vectors to avoid misuse
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        return EmbeddingResult(vectors=vectors, processing_time=0.0, device=self._device, model_info=self.model_name)

    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cross-encoder backend does not support vector similarity")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "torch",
            "type": "cross-encoder",
            "model_name": self.model_name,
            "rerank_method": "cross-encoder",
            "rerank_model_name": self.model_name,
            "device": self._device,
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
        }

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "backend": "torch",
            "device": self._device,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        if not self._is_loaded or self.cross_encoder is None:
            raise RuntimeError("Cross-encoder model not loaded")

        start = time.time()
        # Build (query, passage) pairs
        pairs = [(query, p) for p in passages]

        # sentence-transformers CrossEncoder outputs higher score = higher relevance
        scores = self.cross_encoder.predict(pairs, batch_size=self._batch_size)
        # Ensure python floats
        scores = [float(s) for s in scores]
        _ = time.time() - start
        return scores
