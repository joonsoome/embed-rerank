"""
PyTorch-based embedding backend with MPS support.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import torch
from sentence_transformers import SentenceTransformer

from .base import BaseBackend, EmbeddingResult
from ..utils.device import get_optimal_torch_device
from ..utils.logger import setup_logging

logger = setup_logging()


class TorchBackend(BaseBackend):
    """PyTorch-based embedding backend with Apple Silicon MPS support."""
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize TorchBackend.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (auto-detect if None)
        """
        super().__init__(model_name, device)
        self.device = device or get_optimal_torch_device()
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(
            "Initializing TorchBackend",
            model_name=model_name,
            device=self.device
        )
    
    def _detect_device(self) -> str:
        """Detect the best available PyTorch device."""
        return get_optimal_torch_device()
    
    async def load_model(self) -> None:
        """Load the SentenceTransformer model asynchronously."""
        if self._is_loaded:
            logger.info("Model already loaded", model_name=self.model_name)
            return
        
        logger.info("Loading model", model_name=self.model_name, device=self.device)
        start_time = time.time()
        
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self._executor, 
                self._load_model_sync
            )
            
            self._load_time = time.time() - start_time
            self._is_loaded = True
            
            logger.info(
                "Model loaded successfully",
                model_name=self.model_name,
                device=self.device,
                load_time=self._load_time
            )
            
        except Exception as e:
            logger.error(
                "Failed to load model",
                model_name=self.model_name,
                device=self.device,
                error=str(e)
            )
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _load_model_sync(self) -> SentenceTransformer:
        """Synchronous model loading."""
        try:
            # Load model with specified device
            model = SentenceTransformer(self.model_name, device=self.device)
            
            # Optimize for inference
            model.eval()
            
            # Use FP16 for memory efficiency on compatible devices
            if self.device in ['cuda', 'mps'] and hasattr(model, 'half'):
                try:
                    model.half()
                    logger.info("Using FP16 precision", device=self.device)
                except Exception as e:
                    logger.warning("Could not enable FP16", device=self.device, error=str(e))
            
            return model
            
        except Exception as e:
            logger.error("Model loading failed in sync method", error=str(e))
            raise
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Generate embeddings using SentenceTransformer.
        
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
            "Generating embeddings",
            num_texts=len(texts),
            batch_size=batch_size,
            device=self.device
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
                "Embeddings generated",
                num_texts=len(texts),
                embedding_dim=vectors.shape[1],
                processing_time=processing_time,
                device=self.device
            )
            
            return EmbeddingResult(
                vectors=vectors,
                processing_time=processing_time,
                device=self.device,
                model_info=self.model_name
            )
            
        except Exception as e:
            logger.error(
                "Embedding generation failed",
                num_texts=len(texts),
                error=str(e)
            )
            raise RuntimeError(f"Embedding failed: {e}")
    
    def _embed_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Synchronous embedding generation."""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalize for cosine similarity
                )
            return embeddings
            
        except Exception as e:
            logger.error("Sync embedding failed", error=str(e))
            raise
    
    async def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        passage_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity scores.
        
        Args:
            query_embedding: Query embedding vector
            passage_embeddings: Passage embedding matrix
        
        Returns:
            Array of similarity scores
        """
        try:
            # Ensure embeddings are normalized (should already be from encode)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(
                passage_embeddings, axis=1, keepdims=True
            )
            
            # Compute cosine similarity
            similarities = np.dot(passage_norms, query_norm)
            
            return similarities
            
        except Exception as e:
            logger.error("Similarity computation failed", error=str(e))
            raise RuntimeError(f"Similarity computation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        info = {
            "backend": "torch",
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
        }
        
        if self._is_loaded and self.model:
            try:
                # Get model-specific info
                info.update({
                    "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
                    "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                })
            except Exception as e:
                logger.warning("Could not get model info", error=str(e))
        
        return info
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return PyTorch device capabilities."""
        info = {
            "backend": "torch",
            "device": self.device,
            "torch_version": torch.__version__,
        }
        
        try:
            # MPS info
            if hasattr(torch.backends, 'mps'):
                info.update({
                    "mps_available": torch.backends.mps.is_available(),
                    "mps_built": torch.backends.mps.is_built(),
                })
            
            # CUDA info
            if torch.cuda.is_available():
                info.update({
                    "cuda_available": True,
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                })
                
                if self.device.startswith('cuda'):
                    device_props = torch.cuda.get_device_properties(self.device)
                    info.update({
                        "cuda_device_name": device_props.name,
                        "cuda_memory_total": device_props.total_memory,
                    })
            else:
                info["cuda_available"] = False
        
        except Exception as e:
            logger.warning("Could not get device info", error=str(e))
        
        return info
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
