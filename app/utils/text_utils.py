"""
🚀 Apple MLX 최적화된 텍스트 처리 유틸리티

빠른 추출적 요약과 텍스트 전처리로 embedding 성능 극대화! 🔥
Apple Silicon의 unified memory architecture를 활용한 초고속 처리!
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextProcessingResult:
    """텍스트 처리 결과 정보"""

    original_text: str
    processed_text: str
    original_tokens: int
    processed_tokens: int
    truncated: bool
    strategy_used: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "original_tokens": self.original_tokens,
            "processed_tokens": self.processed_tokens,
            "truncated": self.truncated,
            "strategy": self.strategy_used,
            "warnings": self.warnings,
            "token_reduction": self.original_tokens - self.processed_tokens,
            "reduction_percentage": (
                round((self.original_tokens - self.processed_tokens) / self.original_tokens * 100, 1)
                if self.original_tokens > 0
                else 0
            ),
        }


class TextSummarizer:
    """🚀 Apple MLX 최적화된 텍스트 요약 유틸리티

    빠른 추출적 요약으로 embedding 성능 극대화! 🔥
    Apple Silicon의 강력한 텍스트 처리 성능을 활용합니다.
    """

    @staticmethod
    def truncate_by_tokens(text: str, max_tokens: int = 512) -> str:
        """토큰 기준으로 텍스트 자르기 - 초고속 처리! ⚡

        Args:
            text: 처리할 텍스트
            max_tokens: 최대 토큰 수

        Returns:
            자른 텍스트 (단어 경계 보존)
        """
        # 대략적인 토큰 계산 (1 토큰 ≈ 4 글자)
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # 단어 경계에서 자르기
        truncated = text[:max_chars]
        # 마지막 공백에서 자르기 (단어 중간에서 끊어지지 않도록)
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # 너무 많이 자르지 않도록
            truncated = truncated[:last_space]

        return truncated + "..."

    @staticmethod
    def extract_key_sentences(text: str, max_sentences: int = 3) -> str:
        """핵심 문장 추출 - MLX 백엔드 친화적! 🎯

        Args:
            text: 처리할 텍스트
            max_sentences: 최대 문장 수

        Returns:
            추출된 핵심 문장들
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return text

        # 첫 문장 + 중간 문장들 선택
        selected = [sentences[0]]
        if max_sentences > 1:
            mid_start = len(sentences) // 3
            selected.extend(sentences[mid_start : mid_start + max_sentences - 1])

        return '. '.join(selected) + '.'

    @staticmethod
    def smart_truncate(text: str, max_tokens: int = 512) -> str:
        """🧠 스마트 요약 - 문장 경계 보존하며 자르기

        Apple MLX 성능을 위해 최적화된 스마트 텍스트 자르기!
        문장 완성도를 보장하면서 토큰 제한을 준수합니다.

        Args:
            text: 처리할 텍스트
            max_tokens: 최대 토큰 수

        Returns:
            스마트하게 자른 텍스트
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # 문장 단위로 자르기 시도
        sentences = re.split(r'[.!?]+', text)
        result = ""
        char_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 문장을 추가했을 때 길이 체크
            if char_count + len(sentence) + 2 > max_chars:  # +2 for ". "
                break

            result += sentence + ". "
            char_count += len(sentence) + 2

        # 결과가 비어있다면 강제로 자르기
        if not result.strip():
            return TextSummarizer.truncate_by_tokens(text, max_tokens)

        return result.strip()

    @staticmethod
    def validate_and_process_text(
        text: str, max_tokens: int, strategy: str = "smart_truncate", return_processing_info: bool = False
    ) -> str | Tuple[str, TextProcessingResult]:
        """📝 텍스트 검증 및 처리 - 개선된 All-in-One 솔루션! 🚀

        Args:
            text: 처리할 텍스트
            max_tokens: 최대 토큰 수
            strategy: 처리 전략 ("truncate", "extract", "smart_truncate", "error")
            return_processing_info: 처리 정보 반환 여부

        Returns:
            처리된 텍스트 또는 (처리된 텍스트, 처리 정보) 튜플

        Raises:
            ValueError: 빈 텍스트이거나 strategy가 "error"이고 토큰 한계 초과인 경우
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()

        # 토큰 길이 추정 (1 토큰 ≈ 4 글자)
        original_tokens = len(text) // 4
        warnings = []

        # 처리 결과 초기화
        result = TextProcessingResult(
            original_text=text,
            processed_text=text,
            original_tokens=original_tokens,
            processed_tokens=original_tokens,
            truncated=False,
            strategy_used=strategy,
            warnings=warnings,
        )

        if original_tokens <= max_tokens:
            # 토큰 제한 내 - 그대로 반환
            if return_processing_info:
                return text, result
            return text

        # 토큰 제한 초과 처리
        logger.info(f"🔧 Text too long ({original_tokens} > {max_tokens} tokens), applying {strategy}")

        if strategy == "error":
            raise ValueError(f"Text exceeds maximum token limit: {original_tokens} > {max_tokens} tokens")

        # 전략에 따른 처리
        if strategy == "truncate":
            processed_text = TextSummarizer.truncate_by_tokens(text, max_tokens)
        elif strategy == "extract":
            # 문장 수를 토큰 길이에 맞게 조정
            max_sentences = max(1, max_tokens // 100)
            processed_text = TextSummarizer.extract_key_sentences(text, max_sentences)
        else:  # smart_truncate (default)
            processed_text = TextSummarizer.smart_truncate(text, max_tokens)

        # 처리 결과 업데이트
        result.processed_text = processed_text
        result.processed_tokens = len(processed_text) // 4
        result.truncated = True

        # 경고 메시지 추가
        if original_tokens > max_tokens * 2:
            warnings.append(f"Text was significantly longer than recommended ({original_tokens} tokens)")

        if result.processed_tokens < original_tokens * 0.5:
            warnings.append("More than 50% of original text was removed")

        if return_processing_info:
            return processed_text, result
        return processed_text

    @staticmethod
    def process_texts_with_options(
        texts: List[str],
        max_tokens: int,
        absolute_max_tokens: int,
        strategy: str = "smart_truncate",
        auto_truncate: bool = True,
        return_processing_info: bool = False,
    ) -> Tuple[List[str], Optional[List[TextProcessingResult]]]:
        """🚀 여러 텍스트 일괄 처리 - 고급 옵션 지원! ⚡

        Args:
            texts: 처리할 텍스트 리스트
            max_tokens: 권장 최대 토큰 수
            absolute_max_tokens: 절대 최대 토큰 수 (초과시 에러)
            strategy: 처리 전략
            auto_truncate: 자동 축소 활성화 여부
            return_processing_info: 처리 정보 반환 여부

        Returns:
            (처리된 텍스트 리스트, 처리 정보 리스트)

        Raises:
            ValueError: 절대 최대 토큰 초과 또는 auto_truncate=False인데 토큰 한계 초과
        """
        processed_texts = []
        processing_infos = [] if return_processing_info else None

        for i, text in enumerate(texts):
            try:
                # 절대 최대 길이 체크
                estimated_tokens = len(text) // 4

                if estimated_tokens > absolute_max_tokens:
                    raise ValueError(
                        f"Text at index {i} exceeds absolute maximum token limit: "
                        f"{estimated_tokens} > {absolute_max_tokens} tokens. "
                        f"Please split the text into smaller chunks."
                    )

                # 권장 길이 체크
                if estimated_tokens > max_tokens:
                    if not auto_truncate:
                        raise ValueError(
                            f"Text at index {i} exceeds recommended token limit: "
                            f"{estimated_tokens} > {max_tokens} tokens. "
                            f"Enable auto_truncate or reduce text length."
                        )

                    # 자동 축소 처리
                    if return_processing_info:
                        processed_text, processing_info = TextSummarizer.validate_and_process_text(
                            text, max_tokens, strategy, return_processing_info=True
                        )
                        processing_infos.append(processing_info)
                    else:
                        processed_text = TextSummarizer.validate_and_process_text(
                            text, max_tokens, strategy, return_processing_info=False
                        )

                    processed_texts.append(processed_text)
                else:
                    # 토큰 제한 내 - 그대로 처리
                    processed_texts.append(text)

                    if return_processing_info:
                        processing_infos.append(
                            TextProcessingResult(
                                original_text=text,
                                processed_text=text,
                                original_tokens=estimated_tokens,
                                processed_tokens=estimated_tokens,
                                truncated=False,
                                strategy_used="none",
                                warnings=[],
                            )
                        )

            except Exception as e:
                logger.error(f"💥 Text processing failed for text {i}: {e}")
                raise

        return processed_texts, processing_infos
