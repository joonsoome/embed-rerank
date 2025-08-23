"""
MLX-based embedding backend for Apple Silicon.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import BaseBackend, EmbeddingResult
from ..utils.logger import setup_logging

logger = setup_logging()

# Conditional MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from transformers import AutoTokenizer
    MLX_AVAILABLE = True
    logger.info("MLX modules successfully imported")
except ImportError as e:
    MLX_AVAILABLE = False
    logger.warning("MLX not available", error=str(e))


class MLXBackend(BaseBackend):
    """MLX-based embedding backend optimized for Apple Silicon."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None):
        """
        Initialize MLXBackend.
        
        Args:
            model_name: HuggingFace model identifier
            model_path: Optional path to MLX converted model
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. MLX requires macOS and Apple Silicon. "
                "Install with: pip install mlx>=0.4.0 mlx-lm>=0.2.0"
            )
        
        super().__init__(model_name, "mlx")
        self.model_path = model_path
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(
            "Initializing MLXBackend",
            model_name=model_name,
            model_path=model_path,
            device="mlx"
        )
    
    async def load_model(self) -> None:
        """Load the MLX model asynchronously."""
        if self._is_loaded:
            logger.info("Model already loaded", model_name=self.model_name)
            return
        
        logger.info("Loading MLX model", model_name=self.model_name)
        start_time = time.time()
        
        try:
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            self.model, self.tokenizer = await loop.run_in_executor(
                self._executor,
                self._load_model_sync
            )
            
            self._load_time = time.time() - start_time
            self._is_loaded = True
            
            logger.info(
                "MLX model loaded successfully",
                model_name=self.model_name,
                load_time=self._load_time,
                device="mlx"
            )
            
        except Exception as e:
            logger.error(
                "Failed to load MLX model",
                model_name=self.model_name,
                error=str(e)
            )
            raise RuntimeError(f"Failed to load MLX model {self.model_name}: {e}")
    
    def _load_model_sync(self):
        """Synchronous MLX model loading."""
        try:
            if self.model_path:
                # Load from local MLX converted model
                logger.info("Loading MLX model from path", path=self.model_path)
                model, tokenizer = load(self.model_path)
            else:
                # Load from HuggingFace and convert to MLX on-the-fly
                logger.info("Loading model from HuggingFace", model_name=self.model_name)
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # For now, we'll use a placeholder approach since MLX model conversion
                # is complex and model-specific. In a real implementation, you'd need
                # to implement proper MLX model conversion for the specific embedding model.
                logger.warning(
                    "Direct MLX conversion not implemented for this model type. "
                    "Consider using pre-converted MLX models or the TorchBackend."
                )
                
                # Fallback: we'll simulate MLX loading but actually use a simplified approach
                # In production, you'd implement proper MLX embedding model loading here
                model = None  # Placeholder
                
            return model, tokenizer
            
        except Exception as e:
            logger.error("MLX model loading failed", error=str(e))
            raise
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Generate embeddings using MLX.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
        
        Returns:
            EmbeddingResult with vectors and metadata
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.validate_inputs(texts)
        start_time = time.time()
        
        logger.info(
            "Generating embeddings with MLX",
            num_texts=len(texts),
            batch_size=batch_size,
            device="mlx"
        )
        
        try:
            # Run embedding in thread pool
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                self._executor,
                self._embed_sync,
                texts,
                batch_size
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                "MLX embeddings generated",
                num_texts=len(texts),
                embedding_dim=vectors.shape[1] if vectors.ndim > 1 else len(vectors),
                processing_time=processing_time,
                device="mlx"
            )
            
            return EmbeddingResult(
                vectors=vectors,
                processing_time=processing_time,
                device="mlx",
                model_info=self.model_name
            )
            
        except Exception as e:
            logger.error(
                "MLX embedding generation failed",
                num_texts=len(texts),
                error=str(e)
            )
            raise RuntimeError(f"MLX embedding failed: {e}")
    
    def _embed_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Synchronous MLX embedding generation."""
        try:
            # This is a simplified implementation placeholder
            # In a real MLX embedding implementation, you would:
            # 1. Tokenize the texts using the tokenizer
            # 2. Run forward pass through the MLX model
            # 3. Extract embeddings from the model output
            # 4. Apply pooling (mean, cls, etc.)
            # 5. Normalize the embeddings
            
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                batch_embeddings = self._process_batch_mlx(batch_texts)
                embeddings_list.extend(batch_embeddings)
            
            embeddings_array = np.array(embeddings_list)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            return embeddings_array
            
        except Exception as e:
            logger.error("MLX sync embedding failed", error=str(e))
            raise
    
    def _process_batch_mlx(self, batch_texts: List[str]) -> List[List[float]]:
        """
        Process a batch of texts using MLX.
        
        This is a placeholder implementation. In a real scenario, you would:
        1. Tokenize the batch
        2. Run MLX model forward pass
        3. Extract and pool embeddings
        """
        try:
            if not self.tokenizer:
                raise RuntimeError("Tokenizer not loaded")
            
            # For now, return placeholder embeddings
            # In a real implementation, this would use the actual MLX model
            embedding_dim = 384  # Common dimension for many embedding models
            batch_embeddings = []
            
            for text in batch_texts:
                # Placeholder: generate random normalized embeddings
                # Replace this with actual MLX model inference
                embedding = np.random.randn(embedding_dim).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                batch_embeddings.append(embedding.tolist())
            
            logger.warning(
                "Using placeholder embeddings. Implement actual MLX model inference."
            )
            
            return batch_embeddings
            
        except Exception as e:
            logger.error("MLX batch processing failed", error=str(e))
            raise
    
    async def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        passage_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity using MLX operations.
        
        Args:
            query_embedding: Query embedding vector
            passage_embeddings: Passage embedding matrix
        
        Returns:
            Array of similarity scores
        """
        try:
            # Convert to MLX arrays for potential acceleration
            query_mx = mx.array(query_embedding)
            passages_mx = mx.array(passage_embeddings)
            
            # Normalize embeddings
            query_norm = query_mx / mx.linalg.norm(query_mx)
            passage_norms = passages_mx / mx.linalg.norm(passages_mx, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities_mx = mx.matmul(passage_norms, query_norm)
            
            # Convert back to numpy
            similarities = np.array(similarities_mx)
            
            return similarities
            
        except Exception as e:
            logger.error("MLX similarity computation failed", error=str(e))
            # Fallback to numpy computation
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(
                passage_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(passage_norms, query_norm)
            return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return MLX model metadata."""
        info = {
            "backend": "mlx",
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": "mlx",
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
        }
        
        if self._is_loaded:
            try:
                # Add MLX-specific model info
                info.update({
                    "mlx_device": "apple_silicon",
                    "memory_usage": "unified_memory",
                })
            except Exception as e:
                logger.warning("Could not get MLX model info", error=str(e))
        
        return info
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return MLX device capabilities."""
        info = {
            "backend": "mlx",
            "device": "mlx",
            "apple_silicon": True,
        }
        
        try:
            if MLX_AVAILABLE:
                info.update({
                    "mlx_version": getattr(mx, '__version__', 'unknown'),
                    "unified_memory": True,
                    "metal_support": True,
                })
            else:
                info["mlx_available"] = False
        
        except Exception as e:
            logger.warning("Could not get MLX device info", error=str(e))
        
        return info
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
