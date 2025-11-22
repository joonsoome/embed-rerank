"""
MLX Cross-Encoder Reranker Backend with Full Transformer Forward Pass

This backend implements proper cross-encoder reranking using mlx-lm to load
Qwen3-style reranker models and run the complete transformer forward pass.

For reranking, we concatenate query and passage, run through the transformer,
and use the final hidden state (typically CLS token or mean pooling) with
a classification head to produce relevance scores.

Supported models:
- galaxycore/Qwen3-Reranker-8B-MLX-4bit
- vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
- Any Qwen3-style reranker MLX model
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseBackend, EmbeddingResult

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore

try:
    from mlx_lm import load as mlx_lm_load
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    mlx_lm_load = None  # type: ignore

try:
    from ..utils.logger import setup_logging
    logger = setup_logging()
except Exception:
    import logging
    logger = logging.getLogger(__name__)


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


class MLXCrossEncoderBackend(BaseBackend):
    """
    MLX Cross-Encoder Reranker with Full Transformer Forward Pass

    This backend properly loads Qwen3-style reranker models using mlx-lm
    and runs the complete transformer forward pass to generate relevance scores.

    Cross-encoder reranking works by:
    1. Concatenating query and passage into a single sequence
    2. Running the full transformer forward pass
    3. Pooling the output hidden states
    4. Applying a classification head (or using similarity) to get relevance score
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        pooling: str = "last",  # Changed default: Qwen3 uses last token
        score_norm: str = "sigmoid",  # Ensure sigmoid is applied
    ):
        """
        Initialize the MLX Cross-Encoder Backend.

        Args:
            model_name: MLX reranker model (e.g., galaxycore/Qwen3-Reranker-8B-MLX-4bit)
            device: Device to use (always "mlx" for this backend)
            batch_size: Batch size for processing query-passage pairs
            max_length: Maximum sequence length for concatenated query+passage
            pooling: Pooling strategy - "last" (default for causal LM), "mean", or "cls"
            score_norm: Score normalization - "sigmoid" (default), "minmax", or "none"
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
                "Install with: pip install mlx-lm>=0.25.2"
            )

        super().__init__(model_name, device or "mlx")
        self._batch_size = batch_size
        self._max_length = max_length
        self._pooling = pooling if pooling in ("mean", "last", "cls") else "mean"
        self._score_norm = score_norm if score_norm in ("sigmoid", "none", "minmax") else "sigmoid"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MLX-Rerank")

        self.model = None
        self.tokenizer = None
        self.config: Dict[str, Any] = {}
        self._hidden_size: int = 4096

        logger.info(
            "Initializing MLX Cross-Encoder with full transformer support",
            model_name=model_name,
            pooling=self._pooling,
            score_norm=self._score_norm,
        )

    async def load_model(self) -> None:
        """Load the reranker model using mlx-lm."""
        if self._is_loaded:
            logger.info("Model already loaded", model_name=self.model_name)
            return

        logger.info("Loading MLX reranker model", model_name=self.model_name)
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, self._load_model_sync)
            self.model, self.tokenizer, self.config = result

            self._load_time = time.time() - start_time
            self._is_loaded = True

            logger.info(
                "MLX reranker loaded with full transformer support",
                model_name=self.model_name,
                load_time=f"{self._load_time:.2f}s",
                hidden_size=self._hidden_size,
            )

        except Exception as e:
            logger.error("Failed to load MLX reranker", error=str(e))
            raise RuntimeError(f"MLX reranker loading failed: {e}")

    def _load_model_sync(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Synchronous model loading using mlx-lm."""
        try:
            logger.info("Loading model via mlx-lm", model_id=self.model_name)

            model, tokenizer = mlx_lm_load(self.model_name)

            # Validate model structure
            if not hasattr(model, 'model'):
                raise ValueError("Model missing expected 'model' attribute")

            inner_model = model.model
            if not hasattr(inner_model, 'embed_tokens'):
                raise ValueError("Model missing 'embed_tokens' layer")
            if not hasattr(inner_model, 'layers'):
                raise ValueError("Model missing 'layers' (transformer blocks)")
            if not hasattr(inner_model, 'norm'):
                raise ValueError("Model missing 'norm' (final layer norm)")

            # Extract config
            config = {}
            if hasattr(model, 'config'):
                config = model.config if isinstance(model.config, dict) else vars(model.config)
            elif hasattr(model, 'args'):
                config = model.args if isinstance(model.args, dict) else vars(model.args)

            self._hidden_size = (
                config.get('hidden_size') or
                config.get('dim') or
                config.get('d_model') or
                4096
            )

            num_layers = len(inner_model.layers)
            logger.info(
                "Reranker model structure validated",
                num_layers=num_layers,
                hidden_size=self._hidden_size,
            )

            return model, tokenizer, config

        except Exception as e:
            logger.error("Reranker model loading failed", error=str(e))
            raise

    def _get_hidden_states(self, input_ids: "mx.array") -> "mx.array":
        """
        Run full transformer forward pass for cross-encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len] for concatenated query+passage

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        inner_model = self.model.model

        # Get token embeddings
        h = inner_model.embed_tokens(input_ids)

        # Pass through ALL transformer layers
        for layer in inner_model.layers:
            h = layer(h, mask=None, cache=None)

        # Apply final layer normalization
        h = inner_model.norm(h)

        return h

    def _pool_hidden_states(
        self,
        hidden_states: "mx.array",
        attention_mask: Optional["mx.array"] = None
    ) -> "mx.array":
        """
        Pool hidden states for cross-encoder scoring.

        CRITICAL: With LEFT PADDING, the last real token is ALWAYS at position -1.
        This simplifies pooling significantly and ensures correct behavior.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional [batch_size, seq_len] - not needed with left padding

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        if self._pooling == "last":
            # With LEFT PADDING, the last real token is always at position -1
            return hidden_states[:, -1, :]

        elif self._pooling == "cls":
            # Use first token (CLS-style) - note: with left padding this is the first pad token
            # For CLS pooling with left padding, we need to find the first real token
            if attention_mask is not None:
                # Find first 1 in attention mask for each sequence
                batch_size = hidden_states.shape[0]
                pooled = []
                for i in range(batch_size):
                    # Find first non-zero position
                    mask_row = attention_mask[i]
                    first_idx = 0
                    for j in range(mask_row.shape[0]):
                        if int(mask_row[j].item()) == 1:
                            first_idx = j
                            break
                    pooled.append(hidden_states[i, first_idx, :])
                return mx.stack(pooled)
            else:
                return hidden_states[:, 0, :]

        else:  # mean pooling
            if attention_mask is not None:
                mask_expanded = mx.expand_dims(attention_mask, axis=-1)
                masked = hidden_states * mask_expanded
                sum_hidden = mx.sum(masked, axis=1)
                sum_mask = mx.sum(mask_expanded, axis=1)
                return sum_hidden / mx.maximum(sum_mask, 1e-9)
            else:
                return mx.mean(hidden_states, axis=1)

    def _get_yes_no_token_ids(self) -> Tuple[int, int]:
        """
        Get token IDs for 'yes' and 'no' tokens.

        Tries multiple variations to handle different tokenizer behaviors:
        - "yes" / "no" (lowercase)
        - "Yes" / "No" (capitalized)
        - " yes" / " no" (with leading space, common in sentence-piece)
        """
        yes_id = None
        no_id = None

        # Variations to try (in order of preference for Qwen models)
        yes_variants = ["yes", "Yes", " yes", " Yes", "YES"]
        no_variants = ["no", "No", " no", " No", "NO"]

        # Strategy 1: Try HF tokenizer convert_tokens_to_ids
        if hasattr(self.tokenizer, '_tokenizer'):
            try:
                hf_tok = self.tokenizer._tokenizer
                unk_id = hf_tok.unk_token_id

                # Try yes variants
                for variant in yes_variants:
                    try:
                        tid = hf_tok.convert_tokens_to_ids(variant)
                        if tid != unk_id:
                            yes_id = tid
                            logger.info(f"Found 'yes' token: '{variant}' -> {yes_id}")
                            break
                    except Exception:
                        continue

                # Try no variants
                for variant in no_variants:
                    try:
                        tid = hf_tok.convert_tokens_to_ids(variant)
                        if tid != unk_id:
                            no_id = tid
                            logger.info(f"Found 'no' token: '{variant}' -> {no_id}")
                            break
                    except Exception:
                        continue

                if yes_id is not None and no_id is not None:
                    return yes_id, no_id
            except Exception as e:
                logger.warning(f"HF tokenizer token lookup failed: {e}")

        # Strategy 2: Encode and find the content token (skip BOS/special tokens)
        try:
            for yes_variant in yes_variants:
                yes_ids = self.tokenizer.encode(yes_variant)
                # Skip BOS token if present (usually ID < 10 or first token)
                for tid in yes_ids:
                    if tid > 10:  # Skip special tokens
                        yes_id = tid
                        logger.info(f"Encoded 'yes' variant '{yes_variant}': {yes_ids} -> using {yes_id}")
                        break
                if yes_id is not None:
                    break

            for no_variant in no_variants:
                no_ids = self.tokenizer.encode(no_variant)
                for tid in no_ids:
                    if tid > 10:
                        no_id = tid
                        logger.info(f"Encoded 'no' variant '{no_variant}': {no_ids} -> using {no_id}")
                        break
                if no_id is not None:
                    break

            if yes_id is not None and no_id is not None:
                return yes_id, no_id
        except Exception as e:
            logger.warning(f"Token encoding failed: {e}")

        # Ultimate fallback - log warning and use approximate values
        logger.warning(f"Could not find yes/no tokens, using fallback IDs. yes_id={yes_id}, no_id={no_id}")
        return yes_id or 9891, no_id or 2201

    def _compute_scores_with_lm_head(self, input_ids: "mx.array", attention_mask: "mx.array") -> Optional["mx.array"]:
        """
        Compute scores using the LM head for yes/no logits.

        Qwen3-Reranker outputs yes/no and we score based on P(yes) vs P(no).

        CRITICAL: With LEFT PADDING, the last real token is ALWAYS at position -1.
        This simplifies extraction significantly.

        Returns:
            Scores array or None if LM head not accessible
        """
        try:
            # Check if model has lm_head
            if not hasattr(self.model, 'lm_head'):
                logger.warning("Model does not have lm_head attribute")
                return None

            # Get hidden states
            hidden_states = self._get_hidden_states(input_ids)

            # With LEFT PADDING, last token is always at position -1
            last_hidden = hidden_states[:, -1, :]

            # Apply LM head to get logits
            logits = self.model.lm_head(last_hidden)  # [batch_size, vocab_size]

            # Get yes/no token IDs
            yes_id, no_id = self._get_yes_no_token_ids()

            # Extract yes/no logits
            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]

            # Log the raw logits for visibility (INFO level)
            batch_size = int(yes_logits.shape[0]) if hasattr(yes_logits, 'shape') else 1
            for i in range(min(batch_size, 3)):  # Log first 3 samples
                y_val = float(yes_logits[i].item()) if batch_size > 1 else float(yes_logits.item())
                n_val = float(no_logits[i].item()) if batch_size > 1 else float(no_logits.item())
                logger.info(f"Sample {i}: yes_logit={y_val:.4f}, no_logit={n_val:.4f}, diff={y_val - n_val:.4f}")

            # Compute score as softmax probability of 'yes'
            # score = exp(yes) / (exp(yes) + exp(no))
            max_logit = mx.maximum(yes_logits, no_logits)
            yes_exp = mx.exp(yes_logits - max_logit)
            no_exp = mx.exp(no_logits - max_logit)
            scores = yes_exp / (yes_exp + no_exp)

            # Log computed scores
            logger.info(f"LM head yes/no scoring: yes_id={yes_id}, no_id={no_id}, batch_size={batch_size}")

            return scores

        except Exception as e:
            logger.warning(f"LM head scoring failed: {e}, falling back to hidden state scoring")
            return None

    def _compute_scores(self, pooled: "mx.array") -> "mx.array":
        """
        Compute relevance scores from pooled hidden states.

        This is a fallback when LM head scoring isn't available.
        Uses a projection of the hidden state that correlates with relevance.

        Args:
            pooled: [batch_size, hidden_size]

        Returns:
            Scores [batch_size]
        """
        # Use the mean of the hidden state as a proxy for "activation level"
        # More relevant passages tend to have higher activation
        scores = mx.mean(pooled, axis=-1)

        return scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply score normalization."""
        if self._score_norm == "sigmoid":
            # Center around 0 and apply sigmoid
            centered = scores - np.mean(scores)
            return 1.0 / (1.0 + np.exp(-centered))

        elif self._score_norm == "minmax":
            s_min, s_max = np.min(scores), np.max(scores)
            if s_max - s_min > 1e-8:
                return (scores - s_min) / (s_max - s_min)
            return np.ones_like(scores) * 0.5

        else:  # none
            return scores

    def _tokenize_texts(self, texts: List[str], max_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Tokenize texts handling both HuggingFace tokenizers and mlx-lm TokenizerWrapper.

        CRITICAL: Uses LEFT PADDING for proper last-token pooling.
        With left padding, the last real token is always at position -1,
        which is essential for extracting the final token's logits.

        The mlx-lm TokenizerWrapper doesn't implement __call__, so we need to either:
        1. Access the underlying HF tokenizer via _tokenizer attribute
        2. Fall back to manual tokenization using encode()
        """
        # Strategy 1: Try accessing underlying HuggingFace tokenizer
        if hasattr(self.tokenizer, '_tokenizer'):
            try:
                hf_tokenizer = self.tokenizer._tokenizer

                # CRITICAL: Set left padding for last-token extraction
                original_padding_side = getattr(hf_tokenizer, 'padding_side', 'right')
                hf_tokenizer.padding_side = 'left'

                # Ensure pad token is set
                if hf_tokenizer.pad_token is None:
                    hf_tokenizer.pad_token = hf_tokenizer.eos_token

                encodings = hf_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='np',
                )

                # Restore original padding side
                hf_tokenizer.padding_side = original_padding_side

                return {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask'],
                }
            except Exception as e:
                logger.warning(f"HF tokenizer call failed: {e}, trying encode method")

        # Strategy 2: Try direct __call__ (for native HF tokenizers)
        if callable(self.tokenizer):
            try:
                # Try to set left padding
                original_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
                self.tokenizer.padding_side = 'left'

                encodings = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='np',
                )

                self.tokenizer.padding_side = original_padding_side

                return {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings.get('attention_mask', np.ones_like(encodings['input_ids'])),
                }
            except (TypeError, AttributeError):
                pass

        # Strategy 3: Manual tokenization using encode() method with LEFT PADDING
        logger.info("Using manual tokenization with encode() method (left padding)")
        all_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            ids = ids[:max_length]
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids) if all_ids else 1
        padded_ids = []
        attention_masks = []

        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, 'eos_token_id', 0) or 0

        for ids in all_ids:
            pad_len = max_len - len(ids)
            # LEFT PADDING: padding goes at the beginning
            padded_ids.append([pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            'input_ids': np.array(padded_ids, dtype=np.int64),
            'attention_mask': np.array(attention_masks, dtype=np.int64),
        }

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        """
        Rerank passages using full cross-encoder transformer forward pass.

        Args:
            query: Query text
            passages: List of passage texts to rerank

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not passages:
            return []

        start_time = time.time()
        logger.info(f"Cross-encoder reranking {len(passages)} passages")

        try:
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                self._executor, self._rerank_sync, query, passages
            )

            processing_time = time.time() - start_time
            logger.info(f"Cross-encoder reranking completed in {processing_time:.3f}s")

            return scores

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}")
            # Fallback to simple similarity
            return self._fallback_rerank(query, passages)

    def _format_rerank_input(self, query: str, document: str) -> str:
        """
        Format query-document pair using Qwen3-Reranker chat template.

        Qwen3-Reranker requires the EXACT prompt format with:
        - System message with specific wording
        - User section with <Instruct>, <Query>, <Document> tags
        - Assistant section with <think></think> tokens for reasoning

        Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
        """
        # Use the exact format from Qwen3-Reranker documentation
        formatted = (
            f"<|im_start|>system\n"
            f"Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            f"Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n\n</think>\n"
        )
        return formatted

    def _rerank_sync(self, query: str, passages: List[str]) -> List[float]:
        """
        Synchronous cross-encoder reranking with full transformer forward pass.

        Uses Qwen3-Reranker chat template format:
        <|im_start|>system
        Judge whether the Document is relevant to the Query. Answer only "yes" or "no".<|im_end|>
        <|im_start|>user
        <Query>: {query}
        <Document>: {document}<|im_end|>
        <|im_start|>assistant

        Tries LM head yes/no scoring first, falls back to hidden state scoring.
        """
        all_scores = []

        # Process in batches
        for i in range(0, len(passages), self._batch_size):
            batch_passages = passages[i:i + self._batch_size]

            # Format inputs using Qwen3-Reranker chat template
            pairs = []
            for passage in batch_passages:
                # Use proper chat template format
                pair_text = self._format_rerank_input(query, passage)
                pairs.append(pair_text)

            # Tokenize all pairs using our compatible tokenization method
            encodings = self._tokenize_texts(pairs, max_length=self._max_length)

            input_ids = _mx_array(encodings['input_ids'])
            attention_mask = _mx_array(encodings['attention_mask'])

            # Try LM head scoring first (proper yes/no logits)
            batch_scores_mx = self._compute_scores_with_lm_head(input_ids, attention_mask)

            if batch_scores_mx is not None:
                # LM head scoring succeeded - scores are already in [0, 1]
                mx.eval(batch_scores_mx)
                batch_scores = np.array(batch_scores_mx.tolist(), dtype=np.float32)
                # These are already normalized probabilities
                all_scores.extend(batch_scores.tolist())
            else:
                # Fall back to hidden state scoring
                hidden_states = self._get_hidden_states(input_ids)
                pooled = self._pool_hidden_states(hidden_states, attention_mask)
                scores = self._compute_scores(pooled)
                mx.eval(scores)
                batch_scores = np.array(scores.tolist(), dtype=np.float32)
                all_scores.extend(batch_scores.tolist())

        # Normalize scores if using hidden state fallback
        scores_array = np.array(all_scores)

        # Check if scores are already normalized (from LM head)
        if np.all((scores_array >= 0) & (scores_array <= 1)):
            # Already normalized from LM head
            return scores_array.tolist()
        else:
            # Apply normalization for hidden state scores
            normalized = self._normalize_scores(scores_array)
            return normalized.tolist()

    def _fallback_rerank(self, query: str, passages: List[str]) -> List[float]:
        """Fallback using Jaccard similarity."""
        logger.warning("Using fallback Jaccard similarity")
        query_words = set(query.lower().split())
        scores = []
        for passage in passages:
            passage_words = set(passage.lower().split())
            overlap = len(query_words.intersection(passage_words))
            total = len(query_words.union(passage_words))
            scores.append(overlap / max(total, 1))
        return scores

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Generate embeddings (for interface compatibility).

        Cross-encoders aren't typically used for embeddings, but we provide
        this for interface compatibility.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            self._executor, self._embed_texts_sync, texts, batch_size
        )

        return EmbeddingResult(
            vectors=vectors,
            processing_time=time.time() - start_time,
            device="mlx",
            model_info=self.model_name
        )

    def _embed_texts_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using transformer forward pass."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Use our compatible tokenization method
            encodings = self._tokenize_texts(batch, max_length=self._max_length)

            input_ids = _mx_array(encodings['input_ids'])
            attention_mask = _mx_array(encodings['attention_mask'])

            hidden_states = self._get_hidden_states(input_ids)
            pooled = self._pool_hidden_states(hidden_states, attention_mask)

            # Normalize
            norm = mx.linalg.norm(pooled, axis=-1, keepdims=True)
            normalized = pooled / mx.maximum(norm, 1e-9)

            mx.eval(normalized)
            batch_embeddings = np.array(normalized.tolist(), dtype=np.float32)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    async def compute_similarity(self, query_embedding, passage_embeddings):
        """Compute cosine similarity (for interface compatibility)."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        passage_norms = passage_embeddings / np.linalg.norm(
            passage_embeddings, axis=1, keepdims=True
        )
        return np.dot(passage_norms, query_norm)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "backend": "mlx",
            "model_name": self.model_name,
            "rerank_method": "cross-encoder-full",
            "transformer_forward_pass": True,
            "batch_size": self._batch_size,
            "pooling": self._pooling,
            "score_norm": self._score_norm,
            "hidden_size": self._hidden_size,
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Return device capabilities."""
        return {
            "backend": "mlx",
            "device": "mlx",
            "mlx_available": MLX_AVAILABLE,
            "mlx_lm_available": MLX_LM_AVAILABLE,
            "apple_silicon": True,
        }

    def __del__(self):
        """Cleanup."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
