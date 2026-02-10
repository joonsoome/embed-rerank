def test_embedding_token_limit_alias(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("EMBEDDING_TOKEN_LIMIT", "1234")
    s = Settings()
    assert s.default_max_tokens_override == 1234


def test_embedding_send_dim_alias(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("EMBEDDING_SEND_DIM", "false")
    s = Settings()
    assert s.embedding_send_dim is False


def test_rerank_chunking_envs(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("RERANK_ENABLE_CHUNKING", "true")
    monkeypatch.setenv("RERANK_MAX_TOKENS_PER_DOC", "480")
    s = Settings()
    assert s.rerank_enable_chunking is True
    assert s.rerank_max_tokens_per_doc == 480
