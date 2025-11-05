# 🔥 Embeddings + Reranking on your Mac (MLX‑first)

<p>
  <a href="docs/ENHANCED_OPENAI_API.md">
    <img src="https://img.shields.io/badge/OpenAI%20rerank-supported-2ea44f" alt="OpenAI rerank supported (/v1/openai/rerank)" />
  </a>
  <a href="docs/DEPLOYMENT_PROFILES.md">
    <img src="https://img.shields.io/badge/auto--sigmoid-default%20on-blue" alt="auto-sigmoid default on" />
  </a>
  <a href="https://pypi.org/project/embed-rerank/">
    <img src="https://img.shields.io/pypi/v/embed-rerank?logo=pypi&logoColor=white" alt="PyPI Version" />
  </a>
</p>

Blazing‑fast local embeddings and true cross‑encoder reranking on Apple Silicon. Works with Native, OpenAI, TEI, and Cohere APIs.

This page is a beginner‑friendly quick start. Detailed guides live in docs/.

## 🚀 Start here (60 seconds)

1) Install and run (embeddings only)

```bash
pip install embed-rerank

# Minimal .env
cat > .env <<'ENV'
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
PORT=9000
HOST=0.0.0.0
ENV

embed-rerank  # http://localhost:9000
```

Want 2560‑D vectors by default? Add this to .env and restart:

```bash
cat >> .env <<'ENV'
# Use the model hidden_size (2560 for Qwen3-Embedding-4B) as output dimension
DIMENSION_STRATEGY=hidden_size
# Or enforce a fixed size (pads/truncates as needed):
# OUTPUT_EMBEDDING_DIMENSION=2560
# DIMENSION_STRATEGY=pad_or_truncate
ENV

# Verify
curl -s http://localhost:9000/api/v1/embed/ \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello"],"normalize":true}' | jq '.vectors[0] | length'
```

2) Try it (embeddings + simple rerank)

```bash
# Embeddings (Native)
curl -s http://localhost:9000/api/v1/embed/ \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Hello MLX","Apple Silicon rocks"]}' | jq '.embeddings | length'

# Rerank fallback (no dedicated reranker yet)
curl -s http://localhost:9000/api/v1/rerank/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"capital of france","documents":["Paris is the capital of France","Berlin is in Germany"],"top_n":2}' | jq '.results[0]'
```

3) Add a dedicated reranker (better quality)

```bash
cat >> .env <<'ENV'
RERANKER_BACKEND=auto
RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2  # Torch (stable)
# MLX experimental v1 also available: vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
ENV

# Restart server, then call Native or OpenAI-compatible rerank
curl -s http://localhost:9000/api/v1/rerank/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"capital of france","documents":["Paris is the capital of France","Berlin is in Germany"],"top_n":2}' | jq '.results[0]'
```

4) (Optional) Run as a macOS service

```bash
# Uses your .env to generate a LaunchAgent and start the service
./tools/setup-macos-service.sh

# Check status and health
launchctl list | grep com.embed-rerank.server
open http://localhost:9000/health/
```

Notes
- OpenAI drop-in supported for both embeddings and rerank (/v1/embeddings, /v1/rerank). See docs for a tiny SDK example.
- Scores may be auto‑sigmoid‑normalized for OpenAI clients by default (disable via `OPENAI_RERANK_AUTO_SIGMOID=false`).
- The root endpoint `/` shows both `embedding_dimension` (served) and `hidden_size` (model config) for clarity.

Quick endpoints reference
- Native: `/api/v1/embed`, `/api/v1/rerank`
- OpenAI: `/v1/embeddings`, `/v1/openai/rerank` (alias: `/v1/rerank_openai`)
- TEI: `/embed`, `/rerank`, `/info`
- Cohere: `/v1/rerank`, `/v2/rerank`

Run the full validation suite
```bash
./tools/server-tests.sh --full
```

## 🧭 Pick your path

- Deployment profiles (Embeddings‑only, Fallback rerank, Dedicated reranker): docs/DEPLOYMENT_PROFILES.md
- OpenAI usage (tiny example + options): docs/ENHANCED_OPENAI_API.md
- Quality benchmarks (JSONL/CSV judgments): docs/QUALITY_BENCHMARKS.md
- Troubleshooting: docs/TROUBLESHOOTING.md
- Backend specs and performance: docs/BACKEND_TECHNICAL_SPECS.md, docs/PERFORMANCE_COMPARISON_CHARTS.md

### Try it with OpenAI SDK (tiny)

```python
import openai

client = openai.OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")

# Embeddings
res = client.embeddings.create(model="text-embedding-ada-002", input=["hello world"]) 
print(len(res.data[0].embedding))

# Rerank (OpenAI-compatible)
rr = client._request(
  "post",
  "/v1/openai/rerank",
  json={
    "query": "capital of france",
    "documents": [
      {"id": "a", "text": "Paris is the capital of France"},
      {"id": "b", "text": "Berlin is in Germany"},
    ],
    "top_n": 2,
  },
)
print(rr.get("results", rr))
```

## 📄 License

MIT License – build amazing things locally.
 
