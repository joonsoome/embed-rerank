# Deployment Profiles

This page details recommended ways to run embed-rerank depending on your goals.

## Profiles

### 1) Embeddings only (simple)
- No reranker model configured (CROSS_ENCODER_MODEL/RERANKER_MODEL_ID unset)
- Endpoints: `/api/v1/embed`, `/v1/embeddings`, `/embed`
- Rerank endpoint exists but falls back to embedding similarity.

Quick start:
```bash
pip install embed-rerank
cat > .env <<'ENV'
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
PORT=9000
HOST=0.0.0.0
ENV
embed-rerank
```

### 2) Embeddings + Reranking (single embedding model)
- No dedicated reranker. Reranking uses embedding similarity.
- Endpoints: Native `/api/v1/rerank`, Cohere `/v1/rerank`, `/v2/rerank`.

### 3) Embeddings + Dedicated Reranker (two models)
- Configure a dedicated cross-encoder reranker.
- Torch example:
```env
RERANKER_BACKEND=torch
RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
```
- MLX experimental v1 example:
```env
RERANKER_BACKEND=mlx
RERANKER_MODEL_ID=vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
# Optional MLX tuning
RERANK_POOLING=mean     # mean|cls
RERANK_SCORE_NORM=none  # none|sigmoid|minmax
```
- Endpoints: Native `/api/v1/rerank`, OpenAI `/v1/rerank`, Cohere `/v1/rerank` `/v2/rerank`.

## OpenAI /v1/rerank behavior
- Prefers the dedicated reranker if configured; else falls back to embedding similarity.
- Scores are sigmoid-normalized to [0,1] by default; disable with `OPENAI_RERANK_AUTO_SIGMOID=false`.

## Native /api/v1/rerank OpenAI auto-sigmoid
- If the client looks like OpenAI (user-agent contains `openai`) or header `x-openai-compat: true`, scores are auto sigmoid-normalized by default.
- Disable globally with `OPENAI_RERANK_AUTO_SIGMOID=false`.

## Links
- OpenAI usage: `docs/ENHANCED_OPENAI_API.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- MLX/Backend specs: `docs/BACKEND_TECHNICAL_SPECS.md`
- Performance: `docs/PERFORMANCE_COMPARISON_CHARTS.md`