"""
Embedding service for text vectorization operations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.backends.base import BackendManager
from app.config import Settings
from app.models.requests import EmbedRequest
from app.models.responses import EmbeddingVector, EmbedResponse
from app.utils.model_metadata import ModelMetadataExtractor
from app.utils.text_utils import TextSummarizer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for handling embedding operations with backend abstraction."""

    def __init__(self, backend_manager: BackendManager, config: Optional[Settings] = None):
        """
        Initialize the embedding service.

        Args:
            backend_manager: Backend management instance
            config: Application settings (optional, will use global if not provided)
        """
        self.backend_manager = backend_manager
        self.config = config or Settings()
        self._request_counter = 0

    async def embed_texts(self, request: EmbedRequest, request_id: Optional[str] = None) -> EmbedResponse:
        """
        Generate embeddings for the provided texts.

        Args:
            request: Embedding request with texts and parameters
            request_id: Optional request identifier for tracking

        Returns:
            EmbedResponse with embeddings and metadata

        Raises:
            ValueError: If backend is not available or texts are invalid
            RuntimeError: If embedding generation fails
        """
        start_time = time.time()
        self._request_counter += 1

        if request_id is None:
            request_id = f"embed_{self._request_counter}_{int(time.time())}"

        logger.info(f"Processing embedding request {request_id} with {len(request.texts)} texts")

        try:
            # Validate backend availability
            if not self.backend_manager.is_available():
                raise ValueError("No backend available for embedding generation")

            # 🚀 텍스트 전처리 및 처리 (설정 기본값 적용)
            processed_texts, processing_info_dicts = await self._preprocess_texts(request)

            # Get embeddings from backend
            # Pass is_query for asymmetric retrieval (queries get instruction prefix, documents don't)
            is_query = request.is_query if request.is_query is not None else False
            embeddings_array = await self._generate_embeddings(
                texts=processed_texts, batch_size=request.batch_size, normalize=request.normalize, is_query=is_query
            )

            # Process results
            processing_time = time.time() - start_time

            # Create embedding objects with metadata (including processing info)
            embeddings = []
            for i, (original_text, processed_text, embedding) in enumerate(
                zip(request.texts, processed_texts, embeddings_array)
            ):
                # Get processing info for this text if available
                processing_info = processing_info_dicts[i] if processing_info_dicts else None

                embeddings.append(
                    EmbeddingVector(
                        embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        index=i,
                        text=(
                            processed_text if len(processed_text) <= 100 else f"{processed_text[:97]}..."
                        ),  # Truncate for response
                        processing_info=processing_info,  # Include processing metadata
                    )
                )

            # Get backend info
            backend_info = self.backend_manager.get_current_backend_info()

            # Create response
            response = EmbedResponse(
                embeddings=embeddings,
                vectors=[emb.embedding for emb in embeddings],  # Legacy format
                dim=len(embeddings[0].embedding) if embeddings else 0,
                backend=backend_info.get("name", "unknown"),
                device=backend_info.get("device", "unknown"),
                processing_time=processing_time,
                model_info=backend_info.get("model_name", "unknown"),
                usage={
                    "total_texts": len(request.texts),
                    "total_tokens": sum(len(text.split()) for text in request.texts),
                    "processing_time_ms": processing_time * 1000,
                    "backend": backend_info.get("name", "unknown"),
                    "batch_size": request.batch_size,
                    "normalize": request.normalize,
                },
                timestamp=datetime.now(),
                num_texts=len(request.texts),  # Add this field for test compatibility
            )

            logger.info(f"Successfully processed embedding request {request_id} in {processing_time:.3f}s")
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(f"Embedding request {request_id} failed after {processing_time:.3f}s: {error_msg}")

            raise RuntimeError(error_msg) from e

    async def _preprocess_texts(self, request: EmbedRequest) -> tuple[List[str], Optional[List[Dict[str, Any]]]]:
        """
        텍스트 전처리 및 길이 관리.

        Args:
            request: 임베딩 요청

        Returns:
            (처리된 텍스트 리스트, 처리 정보 딕셔너리 리스트)
        """
        # 모델 메타데이터에서 토큰 제한 가져오기
        try:
            backend_info = self.backend_manager.get_current_backend_info()
            model_name = backend_info.get("model_name", self.config.model_name)

            # ModelMetadataExtractor 사용하여 메타데이터 추출
            extractor = ModelMetadataExtractor()
            model_path = extractor.get_model_cache_path(model_name)

            if model_path:
                metadata = extractor.extract_metadata_from_path(model_path)
                recommended_max_tokens = metadata.get("recommended_max_tokens", 2048)
                absolute_max_tokens = metadata.get("absolute_max_tokens", 8192)
            else:
                # 기본값 사용
                recommended_max_tokens = 2048
                absolute_max_tokens = 8192

            logger.info(f"📏 Model limits: recommended={recommended_max_tokens}, absolute={absolute_max_tokens} tokens")
        except Exception as e:
            logger.warning(f"⚠️  Failed to get model metadata, using defaults: {e}")
            recommended_max_tokens = 2048
            absolute_max_tokens = 8192

        # 사용자 오버라이드 적용
        max_tokens = recommended_max_tokens
        if request.max_tokens_override:
            if request.max_tokens_override > absolute_max_tokens:
                logger.warning(
                    f"⚠️  max_tokens_override ({request.max_tokens_override}) exceeds absolute limit "
                    f"({absolute_max_tokens}), using absolute limit"
                )
                max_tokens = absolute_max_tokens
            else:
                max_tokens = request.max_tokens_override
                logger.info(
                    f"🔧 Using user-specified max_tokens_override: {max_tokens} "
                    f"(recommended: {recommended_max_tokens})"
                )

        # 사용자 오버라이드가 절대 한도를 초과하는 경우 오류 발생
        if request.max_tokens_override and request.max_tokens_override > absolute_max_tokens:
            raise ValueError(
                f"max_tokens_override ({request.max_tokens_override}) exceeds absolute maximum "
                f"({absolute_max_tokens}) for model {model_name}"
            )
            max_tokens = absolute_max_tokens

        try:
            # 🚀 개선된 텍스트 처리 엔진 사용 (설정 기본값 적용)
            auto_truncate = (
                request.auto_truncate if request.auto_truncate is not None else self.config.default_auto_truncate
            )
            truncation_strategy = request.truncation_strategy or self.config.default_truncation_strategy
            return_processing_info = (
                request.return_processing_info
                if request.return_processing_info is not None
                else self.config.default_return_processing_info
            )

            processed_texts, processing_infos = TextSummarizer.process_texts_with_options(
                texts=request.texts,
                max_tokens=max_tokens,
                absolute_max_tokens=absolute_max_tokens,
                strategy=truncation_strategy,
                auto_truncate=auto_truncate,
                return_processing_info=return_processing_info,
            )

            # 처리 통계 로깅
            truncated_count = sum(1 for info in (processing_infos or []) if info.truncated)
            if truncated_count > 0:
                logger.info(
                    f"📊 Text processing summary: {truncated_count}/{len(request.texts)} texts truncated, "
                    f"strategy='{truncation_strategy}', max_tokens={max_tokens}"
                )

            # 처리 정보를 딕셔너리로 변환 (응답에 포함하기 위해)
            processing_info_dicts = None
            if return_processing_info and processing_infos:
                processing_info_dicts = [info.to_dict() for info in processing_infos]

            return processed_texts, processing_info_dicts

        except Exception as e:
            logger.error(f"💥 Text preprocessing failed: {e}")
            raise ValueError(f"Text preprocessing failed: {str(e)}") from e

    async def _generate_embeddings(
        self, texts: List[str], batch_size: int, normalize: bool = True, is_query: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings using the current backend.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            is_query: Whether texts are queries (True) or documents (False).
                     Queries get instruction prefix for asymmetric retrieval.

        Returns:
            List of embedding vectors
        """
        # Get current backend
        backend = self.backend_manager.get_current_backend()
        if backend is None:
            raise ValueError("No backend available")

        # Generate embeddings using backend
        # Pass is_query for backends that support asymmetric embedding (e.g., MLX with Qwen3)
        try:
            result = await backend.embed_texts(texts=texts, batch_size=batch_size, is_query=is_query)
        except TypeError:
            # Fallback for backends that don't support is_query parameter
            result = await backend.embed_texts(texts=texts, batch_size=batch_size)

        # Extract vectors from result (as numpy array for processing)
        import numpy as np

        embeddings_array = (
            np.array(result.vectors) if not hasattr(result.vectors, 'tolist') else np.array(result.vectors)
        )

        # Determine target dimension based on configuration
        target_dim: Optional[int] = None
        chosen_reason: str = "as_is"

        try:
            if self.config.dimension_strategy == "hidden_size":
                backend_info = self.backend_manager.get_current_backend_info()
                hs = backend_info.get("hidden_size")

                # Try to get HF metadata embedding dimension as the authoritative spec
                md_dim: Optional[int] = None
                try:
                    model_name = backend_info.get("model_name", "")
                    from app.utils.model_metadata import ModelMetadataExtractor

                    extractor = ModelMetadataExtractor()
                    path = extractor.get_model_cache_path(model_name) if model_name else None
                    if path:
                        md = extractor.extract_metadata_from_path(str(path))
                        maybe = md.get("embedding_dimension")
                        if isinstance(maybe, int) and maybe > 0:
                            md_dim = maybe
                except Exception:
                    md_dim = None

                # Prefer metadata dimension when available; otherwise use hidden_size from backend
                if isinstance(md_dim, int) and md_dim > 0:
                    target_dim = md_dim
                    chosen_reason = "hf_metadata_embedding_dimension"
                elif isinstance(hs, int) and hs > 0:
                    target_dim = hs
                    chosen_reason = "backend_hidden_size"
            elif self.config.dimension_strategy == "pad_or_truncate":
                if self.config.output_embedding_dimension and self.config.output_embedding_dimension > 0:
                    target_dim = int(self.config.output_embedding_dimension)
                    chosen_reason = "pad_or_truncate_env"
        except Exception:
            # Fall back gracefully if any metadata missing
            target_dim = None
            chosen_reason = "as_is_exception"

        # If target dimension is specified and differs, adjust BEFORE normalization
        if target_dim and embeddings_array.shape[1] != target_dim:
            before = embeddings_array.shape[1]
            if embeddings_array.shape[1] > target_dim:
                embeddings_array = embeddings_array[:, :target_dim]
            else:
                pad = target_dim - embeddings_array.shape[1]
                embeddings_array = np.pad(embeddings_array, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)
            try:
                msg = (
                    f"🔧 Adjusted embedding dimension from {before} to {target_dim} "
                    f"(strategy={getattr(self.config, 'dimension_strategy', 'unknown')}, reason={chosen_reason})"
                )
                logger.info(msg)
            except Exception:
                # Never let logging break the request
                pass

        # Apply normalization if requested
        if normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms

        return embeddings_array.tolist()

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and status.

        Returns:
            Dictionary with service metadata
        """
        backend_info = self.backend_manager.get_current_backend_info()

        return {
            "service": "EmbeddingService",
            "version": "1.0.0",
            "backend": backend_info,
            "requests_processed": self._request_counter,
            "available": self.backend_manager.is_available(),
            "supported_operations": ["embed_texts"],
            "max_batch_size": 128,
            "max_text_length": 8192,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.

        Returns:
            Health status information with model metadata
        """
        try:
            # Test with a simple embedding
            test_texts = ["Health check test"]
            test_request = EmbedRequest(texts=test_texts, batch_size=1)

            start_time = time.time()
            result = await self.embed_texts(test_request, request_id="health_check")
            response_time = time.time() - start_time

            # 🚀 Extract model metadata for health response
            model_metadata = {}
            try:
                backend_info = self.backend_manager.get_current_backend_info()
                model_name = backend_info.get("model_name", "")

                if model_name:
                    from app.utils.model_metadata import ModelMetadataExtractor

                    metadata_extractor = ModelMetadataExtractor()

                    # Try to get metadata from model cache
                    cache_path = metadata_extractor.get_model_cache_path(model_name)
                    if cache_path:
                        model_metadata = metadata_extractor.extract_metadata_from_path(cache_path)

                    # Add service configuration
                    model_metadata.update(
                        {
                            "embedding_dimension": result.dim,
                            "max_tokens": getattr(self.config, 'max_sequence_length', 8192),
                            "recommended_max_tokens": 2048,  # Model specific
                            "warning_threshold": 4096,
                            "optimal_batch_size": getattr(self.config, 'batch_size', 32),
                            "default_auto_truncate": getattr(self.config, 'default_auto_truncate', False),
                            "default_truncation_strategy": getattr(
                                self.config, 'default_truncation_strategy', 'smart_truncate'
                            ),
                            "default_return_processing_info": getattr(
                                self.config, 'default_return_processing_info', False
                            ),
                        }
                    )
            except Exception as meta_error:
                logger.warning(f"Failed to extract model metadata: {meta_error}")

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "backend_available": True,
                "test_embedding_dim": result.dim,
                "service_info": self.get_service_info(),
                "model_metadata": model_metadata,  # 🚀 Include model metadata
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_available": self.backend_manager.is_available(),
                "service_info": self.get_service_info(),
                "model_metadata": {},  # Empty metadata on error
            }
