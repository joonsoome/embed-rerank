#!/usr/bin/env python3
"""
Quick rerank quality metrics against a running server.

- Computes MRR@k and nDCG@k on a tiny synthetic dataset
- Targets: http://localhost:${PORT:-9000}/api/v1/rerank

Usage:
  python tools/tests/quality-rerank-only.py --host http://localhost:9000 --k 3

Notes:
- This is a lightweight smoke metric to compare backend configs (torch vs mlx)
- For reliable evaluation, plug in your own dataset/judgments
"""
import argparse
import json
import os
from typing import List, Dict, Any

import requests


def mrr_at_k(relevances: List[int], k: int) -> float:
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def dcg_at_k(relevances: List[int], k: int) -> float:
    import math
    return sum((rel / math.log2(i + 1)) for i, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def fetch_rerank_scores(host: str, query: str, passages: List[str]) -> List[float]:
    url = host.rstrip('/') + '/api/v1/rerank'
    payload = {
        "query": query,
        "documents": passages,
        "top_n": len(passages),
        "return_documents": False,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Expect scores in the same order as input; our API typically returns sorted documents
    # We reconstruct by mapping back to original indices when return_documents=False
    if isinstance(data, dict) and "results" in data:
        # results is a list of {index, score, document?}
        # Sort by index to realign to original order
        ordered = sorted(data["results"], key=lambda x: x.get("index", 0))
        return [float(item.get("score", 0.0)) for item in ordered]
    # Fallback: direct list
    if isinstance(data, list):
        return [float(x) for x in data]
    raise RuntimeError(f"Unexpected rerank response schema: {data}")


def load_dataset_from_file(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    data: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect {query: str, passages: [str], judgments: [int]}
                q = obj.get("query")
                passages = obj.get("passages") or obj.get("documents")
                judgments = obj.get("judgments") or obj.get("relevances")
                if not isinstance(q, str) or not isinstance(passages, list) or not isinstance(judgments, list):
                    continue
                data.append({"query": q, "passages": passages, "judgments": judgments})
    elif ext == ".csv":
        import csv
        # Expect columns: query, passages, judgments where passages/judgments are JSON arrays
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    q = row.get("query")
                    passages = json.loads(row.get("passages")) if row.get("passages") else []
                    judgments = json.loads(row.get("judgments")) if row.get("judgments") else []
                    if isinstance(q, str) and isinstance(passages, list) and isinstance(judgments, list):
                        data.append({"query": q, "passages": passages, "judgments": judgments})
                except Exception:
                    continue
    else:
        raise ValueError("Unsupported file extension. Use .jsonl or .csv")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST_URL", "http://localhost:9000"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--judgments", help="Path to JSONL or CSV with fields: query, passages, judgments")
    args = parser.parse_args()

    # Tiny demo dataset with known relevances (1=rel, 0=non-rel)
    if args.judgments:
        dataset = load_dataset_from_file(args.judgments)
        if not dataset:
            print("No valid records found in the judgments file; falling back to built-in tiny dataset.")
    else:
        dataset = []
    if not dataset:
        dataset = [
            {
                "query": "capital of france",
                "passages": [
                    "Paris is the capital and most populous city of France.",
                    "Berlin is the capital of Germany.",
                    "The Eiffel Tower is in Paris.",
                    "France is a country in Western Europe.",
                ],
                "judgments": [1, 0, 1, 0],
            },
            {
                "query": "fastapi framework",
                "passages": [
                    "FastAPI is a modern, fast web framework for building APIs with Python.",
                    "Django is a high-level Python Web framework that encourages rapid development.",
                    "Flask is a lightweight WSGI web application framework.",
                    "Starlette is a lightweight ASGI framework/toolkit, ideal for building async services.",
                ],
                "judgments": [1, 0, 0, 1],
            },
        ]

    metrics: List[Dict[str, Any]] = []
    for item in dataset:
        scores = fetch_rerank_scores(args.host, item["query"], item["passages"])
        # Sort passages by predicted score (descending) and collect relevances in that order
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_rels = [item["judgments"][i] for i in order]
        metrics.append(
            {
                "query": item["query"],
                "MRR@k": mrr_at_k(ranked_rels, args.k),
                "nDCG@k": ndcg_at_k(ranked_rels, args.k),
            }
        )

    avg_mrr = sum(m["MRR@k"] for m in metrics) / len(metrics)
    avg_ndcg = sum(m["nDCG@k"] for m in metrics) / len(metrics)

    summary = {
        "k": args.k,
        "samples": len(metrics),
        "average_MRR@k": avg_mrr,
        "average_nDCG@k": avg_ndcg,
        "details": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
