# Rerank Quality Benchmarks

Use these lightweight tools to compare reranker configurations (Torch vs MLX, pooling, normalization).

## Quick check (tiny dataset)
```bash
./tools/server-tests.sh --quality-rerank-only
```

## Custom judgments
The script accepts JSONL or CSV via `--judgments`.

- JSONL: each line is an object with fields
```json
{"query":"...","passages":["...","..."],"judgments":[1,0,1]}
```
  - Aliases: `documents` for `passages`, `relevances` for `judgments`.

- CSV: columns `query`, `passages`, `judgments`
  - `passages` and `judgments` are JSON arrays.

Run:
```bash
python tools/tests/quality-rerank-only.py --host http://localhost:9000 --k 3 --judgments /path/to/data.jsonl
```

### Metrics
- MRR@k – reciprocal rank of the first relevant result
- nDCG@k – graded relevance quality of the top-k ranking

## Tips
- If your client expects scores in [0,1], set `RERANK_SCORE_NORM=sigmoid` for MLX reranker or rely on OpenAI auto-sigmoid features.
- Prefer the dedicated reranker for best quality. MLX v1 is experimental: try `RERANK_POOLING=cls` and `RERANK_SCORE_NORM=sigmoid` as a baseline.
