"""
Utilities for document chunking and score aggregation for reranking.

This is intentionally dependency-free and uses the project's existing heuristic:
1 token ~= 4 characters.
"""

from __future__ import annotations

from typing import List, Literal, Tuple


def chunk_documents(
    documents: List[str],
    max_tokens: int,
    overlap_tokens: int = 32,
) -> Tuple[List[str], List[int]]:
    """Chunk documents into overlapping windows based on an approximate token budget.

    Args:
        documents: List of document strings to chunk.
        max_tokens: Maximum tokens per chunk (approximate).
        overlap_tokens: Overlap between chunks in tokens (approximate).

    Returns:
        (chunked_documents, doc_indices)
        - chunked_documents: the produced chunks (may be longer than input list)
        - doc_indices: maps each chunk index back to its original document index
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be >= 1")

    # Clamp overlap to guarantee progress. If overlap >= max_tokens, the loop can hang.
    if overlap_tokens >= max_tokens:
        overlap_tokens = max(0, max_tokens - 1)

    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunked_docs: List[str] = []
    doc_indices: List[int] = []

    for doc_idx, doc in enumerate(documents):
        if doc is None:
            doc = ""

        if len(doc) <= max_chars:
            chunked_docs.append(doc)
            doc_indices.append(doc_idx)
            continue

        start = 0
        doc_len = len(doc)
        while start < doc_len:
            end = min(start + max_chars, doc_len)
            slice_ = doc[start:end]

            actual_end = end
            # Try to preserve word boundary when we're not at the end of the document.
            if end < doc_len:
                last_space = slice_.rfind(" ")
                # Only cut on a boundary if it doesn't shrink too aggressively.
                if last_space >= 0 and last_space > int(max_chars * 0.8):
                    actual_end = start + last_space
                    slice_ = doc[start:actual_end]

            if actual_end <= start:
                # Defensive: ensure progress even in pathological cases.
                actual_end = end
                slice_ = doc[start:actual_end]

            chunked_docs.append(slice_)
            doc_indices.append(doc_idx)

            if actual_end >= doc_len:
                break

            next_start = actual_end - overlap_chars
            if next_start <= start:
                next_start = actual_end
            start = next_start

    return chunked_docs, doc_indices


def aggregate_scores(
    scores: List[float],
    doc_indices: List[int],
    num_docs: int,
    aggregation: Literal["max"] = "max",
) -> List[tuple[int, float]]:
    """Aggregate chunk-level scores back to original document-level scores.

    Args:
        scores: Chunk-level scores aligned by chunk index.
        doc_indices: Maps each chunk index to original document index.
        num_docs: Number of original documents.
        aggregation: Aggregation strategy. Only "max" is supported.

    Returns:
        List of (doc_index, aggregated_score) sorted by score desc then index asc.
    """
    if aggregation != "max":
        raise ValueError("Only aggregation='max' is supported")
    if num_docs < 0:
        raise ValueError("num_docs must be >= 0")

    best: List[float | None] = [None] * num_docs

    limit = min(len(scores), len(doc_indices))
    for chunk_idx in range(limit):
        doc_idx = doc_indices[chunk_idx]
        if doc_idx < 0 or doc_idx >= num_docs:
            continue
        s = float(scores[chunk_idx])
        cur = best[doc_idx]
        if cur is None or s > cur:
            best[doc_idx] = s

    aggregated: List[tuple[int, float]] = [(i, s) for i, s in enumerate(best) if s is not None]
    aggregated.sort(key=lambda x: (-x[1], x[0]))
    return aggregated

