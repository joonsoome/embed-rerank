from app.utils.rerank_chunking import aggregate_scores, chunk_documents


def test_chunk_documents_splits_and_tracks_indices():
    docs = ["a" * 5000]
    chunks, indices = chunk_documents(docs, max_tokens=480)  # ~1920 chars per chunk
    assert len(chunks) > 1
    assert len(chunks) == len(indices)
    assert all(i == 0 for i in indices)


def test_chunk_documents_overlap_clamp_does_not_hang():
    docs = ["hello world " * 200]
    chunks, indices = chunk_documents(docs, max_tokens=10, overlap_tokens=10)
    assert len(chunks) >= 1
    assert len(chunks) == len(indices)


def test_aggregate_scores_max_and_deterministic_sort():
    scores = [0.1, 0.9, 0.5, 0.5]
    doc_indices = [0, 0, 1, 2]
    aggregated = aggregate_scores(scores, doc_indices, num_docs=3, aggregation="max")
    assert aggregated == [(0, 0.9), (1, 0.5), (2, 0.5)]

