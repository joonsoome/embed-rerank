"""
Apple MLX Backend with Full Transformer Forward Pass

This backend uses mlx-lm to load Qwen3-style models and runs the complete
transformer forward pass to generate semantically meaningful embeddings.

Key features:
- Full transformer inference (not just embedding table lookup)
- Proper hidden state extraction from all transformer layers
- Mean pooling with L2 normalization for sentence embeddings
- Support for 4-bit quantized MLX models

Supported models:
- huynguyendbs/Qwen3-Embedding-8B-4bit-MLX
- mlx-community/Qwen3-Embedding-4B-4bit-DWQ
- Any Qwen3-style MLX model
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.logger import setup_logging
from .base import BaseBackend, EmbeddingResult

logger = setup_logging()

# Conditional MLX imports
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX core imported successfully")
except ImportError as e:
    MLX_AVAILABLE = False
    logger.warning("MLX not available", error=str(e))
    mx = None  # type: ignore

# Conditional mlx-lm import (required for proper model loading)
try:
    from mlx_lm import load as mlx_lm_load
    MLX_LM_AVAILABLE = True
    logger.info("mlx-lm imported successfully - full transformer support enabled")
except ImportError as e:
    MLX_LM_AVAILABLE = False
    logger.warning("mlx-lm not available - install with: pip install mlx-lm>=0.25.2", error=str(e))
    mlx_lm_load = None  # type: ignore


def _mx_array(x):
    """Create an MLX array in a version-compatible way."""
    if not MLX_AVAILABLE or mx is None:
        return np.array(x)

    if hasattr(mx, "array"):
        try:
            return mx.array(x)
        except Exception:
            pass

    if hasattr(mx, "asarray"):
        try:
            return mx.asarray(x)
        except Exception:
            pass

    return np.array(x)


class MLXBackend(BaseBackend):
    """
    MLX Backend with Full Transformer Forward Pass

    This backend properly loads Qwen3-style models using mlx-lm and runs
    the complete transformer forward pass to generate semantically meaningful
    embeddings. Unlike the previous implementation that only did embedding
    table lookups, this runs through all attention and feed-forward layers.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ", model_path: Optional[str] = None):
        """
        Initialize the MLX Backend.

        Args:
            model_name: MLX-optimized model identifier (e.g., huynguyendbs/Qwen3-Embedding-8B-4bit-MLX)
            model_path: Optional path to local MLX model directory

        Raises:
            RuntimeError: If MLX or mlx-lm is not available
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX Framework Required!\n"
                "MLX requires Apple Silicon (M1/M2/M3/M4) and macOS.\n"
                "Install with: pip install mlx>=0.4.0"
            )

        if not MLX_LM_AVAILABLE:
            raise RuntimeError(
                "mlx-lm Required for Full Transformer Support!\n"
                "Install with: pip install mlx-lm>=0.25.2\n"
                "This is required to run proper transformer forward passes."
            )

        super().__init__(model_name, "mlx")
        self.model_path = model_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MLX-Worker")
        self.model = None
        self.tokenizer = None
        self.config = None
        self._hidden_size = None

        logger.info(
            "Initializing MLX Backend with full transformer support",
            model_name=model_name,
            model_path=model_path,
        )

    async def load_model(self) -> None:
        """Load the model using mlx-lm for full transformer support."""
        if self._is_loaded:
            logger.info("Model already loaded", model_name=self.model_name)
            return

        logger.info("Loading MLX model with full transformer layers", model_name=self.model_name)
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            self.model, self.tokenizer, self.config = await loop.run_in_executor(
                self._executor, self._load_model_sync
            )

            self._load_time = time.time() - start_time
            self._is_loaded = True

            logger.info(
                "MLX model loaded with full transformer support",
                model_name=self.model_name,
                load_time=f"{self._load_time:.2f}s",
                hidden_size=self._hidden_size,
                num_layers=self.config.get("num_hidden_layers", "unknown") if self.config else "unknown",
            )

        except Exception as e:
            logger.error("Failed to load MLX model", model_name=self.model_name, error=str(e))
            raise RuntimeError(f"MLX model loading failed for {self.model_name}: {e}")

    def _load_model_sync(self):
        """
        Synchronous model loading using mlx-lm.

        This loads the complete model including all transformer layers,
        not just the embedding table.
        """
        try:
            # Use model_path if provided, otherwise use model_name for HuggingFace download
            model_id = self.model_path if self.model_path else self.model_name

            logger.info("Loading model via mlx-lm", model_id=model_id)

            # mlx_lm.load() returns (model, tokenizer)
            # This properly loads all transformer weights and handles quantization
            model, tokenizer = mlx_lm_load(model_id)

            # Validate model has expected structure for embedding extraction
            if not hasattr(model, 'model'):
                raise ValueError(
                    f"Model {model_id} does not have expected structure. "
                    "Expected 'model' attribute for accessing transformer layers."
                )

            inner_model = model.model

            # Check for required components
            if not hasattr(inner_model, 'embed_tokens'):
                raise ValueError("Model missing 'embed_tokens' layer")
            if not hasattr(inner_model, 'layers'):
                raise ValueError("Model missing 'layers' (transformer blocks)")
            if not hasattr(inner_model, 'norm'):
                raise ValueError("Model missing 'norm' (final layer normalization)")

            # Extract config info
            config = {}
            if hasattr(model, 'config'):
                config = model.config if isinstance(model.config, dict) else vars(model.config)
            elif hasattr(model, 'args'):
                config = model.args if isinstance(model.args, dict) else vars(model.args)

            # Determine hidden size
            self._hidden_size = (
                config.get('hidden_size') or
                config.get('dim') or
                config.get('d_model') or
                4096
            )

            num_layers = len(inner_model.layers)
            logger.info(
                "Model structure validated",
                num_layers=num_layers,
                hidden_size=self._hidden_size,
                has_embed_tokens=True,
                has_norm=True,
            )

            return model, tokenizer, config

        except Exception as e:
            logger.error("Model loading failed", error=str(e))
            raise

    def _get_hidden_states(self, input_ids: "mx.array", attention_mask: Optional["mx.array"] = None) -> "mx.array":
        """
        Run full transformer forward pass and extract hidden states.

        This is the critical fix - we run through ALL transformer layers,
        not just the embedding table lookup.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        inner_model = self.model.model

        # Step 1: Get token embeddings from embedding table
        h = inner_model.embed_tokens(input_ids)

        # Step 2: Pass through ALL transformer layers
        # This is what was missing in the original implementation!
        for layer in inner_model.layers:
            # Each layer applies: attention -> add & norm -> FFN -> add & norm
            h = layer(h, mask=None, cache=None)

        # Step 3: Apply final layer normalization
        h = inner_model.norm(h)

        return h

    def _pool_embeddings(self, hidden_states: "mx.array", attention_mask: Optional["mx.array"] = None) -> "mx.array":
        """
        Pool hidden states to get sentence-level embeddings using LAST TOKEN pooling.

        Qwen3-Embedding uses last token pooling (the EOS token position) rather than
        mean pooling. This is standard for causal LM-based embedding models.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] - required to find last real token

        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        if attention_mask is not None:
            # Find the position of the last real token (before padding)
            # Sum attention mask to get sequence lengths, subtract 1 for 0-indexing
            seq_lengths = mx.sum(attention_mask, axis=1).astype(mx.int32) - 1

            # Extract the hidden state at the last token position for each sequence
            batch_size = hidden_states.shape[0]
            pooled_list = []
            for i in range(batch_size):
                # Get the index of the last real token
                last_idx = int(seq_lengths[i].item())
                # Ensure we don't go negative
                last_idx = max(0, last_idx)
                pooled_list.append(hidden_states[i, last_idx, :])

            pooled = mx.stack(pooled_list)
        else:
            # If no attention mask, assume no padding - use actual last token
            pooled = hidden_states[:, -1, :]

        return pooled

    def _normalize_embeddings(self, embeddings: "mx.array") -> "mx.array":
        """L2 normalize embeddings for cosine similarity."""
        norm = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / mx.maximum(norm, 1e-9)

    def _tokenize_texts(self, texts: List[str], max_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Tokenize texts handling both HuggingFace tokenizers and mlx-lm TokenizerWrapper.

        The mlx-lm TokenizerWrapper doesn't implement __call__, so we need to either:
        1. Access the underlying HF tokenizer via _tokenizer attribute
        2. Fall back to manual tokenization using encode()

        Args:
            texts: List of texts to tokenize
            max_length: Maximum sequence length

        Returns:
            Dict with 'input_ids' and 'attention_mask' as numpy arrays
        """
        # Strategy 1: Try accessing underlying HuggingFace tokenizer
        if hasattr(self.tokenizer, '_tokenizer'):
            try:
                hf_tokenizer = self.tokenizer._tokenizer
                encodings = hf_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='np',
                )
                return {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask'],
                }
            except Exception as e:
                logger.warning(f"HF tokenizer call failed: {e}, trying encode method")

        # Strategy 2: Try direct __call__ (for native HF tokenizers)
        if callable(self.tokenizer):
            try:
                encodings = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='np',
                )
                return {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings.get('attention_mask', np.ones_like(encodings['input_ids'])),
                }
            except TypeError:
                pass  # Not callable, try next strategy

        # Strategy 3: Manual tokenization using encode() method
        logger.info("Using manual tokenization with encode() method")
        all_ids = []
        for text in texts:
            # mlx-lm tokenizer.encode() returns list of ints
            ids = self.tokenizer.encode(text)
            # Truncate to max_length
            ids = ids[:max_length]
            all_ids.append(ids)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_ids) if all_ids else 1
        padded_ids = []
        attention_masks = []

        # Get pad token id (default to 0 if not available)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, 'eos_token_id', 0) or 0

        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            'input_ids': np.array(padded_ids, dtype=np.int64),
            'attention_mask': np.array(attention_masks, dtype=np.int64),
        }

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Generate embeddings using full transformer forward pass.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            EmbeddingResult with semantically meaningful vectors
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.validate_inputs(texts)
        start_time = time.time()

        logger.info(
            "Generating embeddings with full transformer forward pass",
            num_texts=len(texts),
            batch_size=batch_size,
        )

        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(self._executor, self._embed_sync, texts, batch_size)

            processing_time = time.time() - start_time

            logger.info(
                "Embeddings generated successfully",
                num_texts=len(texts),
                embedding_dim=vectors.shape[1] if vectors.ndim > 1 else len(vectors),
                processing_time=f"{processing_time:.3f}s",
            )

            return EmbeddingResult(
                vectors=vectors,
                processing_time=processing_time,
                device="mlx",
                model_info=self.model_name
            )

        except Exception as e:
            logger.error("Embedding generation failed", num_texts=len(texts), error=str(e))
            raise RuntimeError(f"MLX embedding failed: {e}")

    def _embed_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Synchronous embedding generation with full transformer forward pass.

        This is the corrected implementation that runs through all transformer
        layers instead of just doing an embedding table lookup.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not loaded")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch using our compatible tokenization method
            encodings = self._tokenize_texts(batch_texts, max_length=512)

            input_ids = _mx_array(encodings['input_ids'])
            attention_mask = _mx_array(encodings['attention_mask'])

            # Run full transformer forward pass
            hidden_states = self._get_hidden_states(input_ids, attention_mask)

            # Pool to get sentence embeddings
            pooled = self._pool_embeddings(hidden_states, attention_mask)

            # Normalize for cosine similarity
            normalized = self._normalize_embeddings(pooled)

            # Force evaluation and convert to numpy
            mx.eval(normalized)
            batch_embeddings = np.array(normalized.tolist(), dtype=np.float32)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)

        logger.info(
            "Transformer forward pass completed",
            shape=embeddings_array.shape,
            dtype=str(embeddings_array.dtype),
        )

        return embeddings_array

    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores."""
        try:
            query_mx = _mx_array(query_embedding)
            passages_mx = _mx_array(passage_embeddings)

            query_norm = query_mx / mx.linalg.norm(query_mx)
            passage_norms = passages_mx / mx.linalg.norm(passages_mx, axis=1, keepdims=True)

            similarities_mx = mx.matmul(passage_norms, query_norm)
            mx.eval(similarities_mx)

            return np.array(similarities_mx.tolist())

        except Exception as e:
            logger.error("MLX similarity computation failed", error=str(e))
            # Fallback to numpy
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            return np.dot(passage_norms, query_norm)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        info = {
            "backend": "mlx",
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": "mlx",
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
            "transformer_forward_pass": True,  # Indicates we run full forward pass
        }

        if self._is_loaded and self.config:
            info.update({
                "hidden_size": self._hidden_size,
                "num_layers": self.config.get("num_hidden_layers") or len(self.model.model.layers) if self.model else None,
                "mlx_lm_version": "installed",
            })

        return info

    def get_device_info(self) -> Dict[str, Any]:
        """Return MLX device capabilities."""
        info = {
            "backend": "mlx",
            "device": "mlx",
            "apple_silicon": True,
            "mlx_available": MLX_AVAILABLE,
            "mlx_lm_available": MLX_LM_AVAILABLE,
        }

        if MLX_AVAILABLE:
            info.update({
                "mlx_version": getattr(mx, '__version__', 'unknown'),
                "unified_memory": True,
                "metal_support": True,
            })

        return info

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        """
        Rerank passages using embedding similarity.

        Note: For true cross-encoder reranking, use MLXCrossEncoderBackend.
        This method uses bi-encoder similarity which is faster but less accurate.
        """
        start_time = time.time()
        logger.info(f"Reranking {len(passages)} passages with embedding similarity")

        try:
            query_result = await self.embed_texts([query])
            passages_result = await self.embed_texts(passages)

            query_vector = query_result.vectors[0]
            passage_vectors = passages_result.vectors

            scores = await self.compute_similarity(query_vector, passage_vectors)
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

            processing_time = time.time() - start_time
            logger.info(f"Reranking completed in {processing_time:.3f}s")
            return scores_list

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return await self._fallback_rerank(query, passages)

    async def _fallback_rerank(self, query: str, passages: List[str]) -> List[float]:
        """Fallback reranking using word overlap (Jaccard similarity)."""
        logger.warning("Using fallback Jaccard similarity reranking")
        query_words = set(query.lower().split())
        scores = []
        for passage in passages:
            passage_words = set(passage.lower().split())
            overlap = len(query_words.intersection(passage_words))
            total = len(query_words.union(passage_words))
            scores.append(overlap / max(total, 1))
        return scores

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
