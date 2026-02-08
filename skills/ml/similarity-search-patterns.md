---
name: similarity-search-patterns
description: Use when implementing similarity search for document, code, or entity retrieval with robust normalization, filtering, and ranking behavior.
trigger_keywords: [similarity search, cosine similarity, nearest neighbors, semantic similarity, vector retrieval patterns]
---

# Similarity Search Patterns

Use this skill to build predictable similarity retrieval that supports production debugging and tuning.

## When to Use This Skill

- Designing nearest-neighbor retrieval endpoints
- Implementing semantic search APIs for product features
- Debugging poor ranking quality in vector retrieval
- Adding metadata-aware retrieval filters

## Core Concepts

- **Normalize embeddings consistently** at ingest and query time.
- **Use distance metric intentionally** (cosine, dot, L2) per model assumptions.
- **Apply metadata filters before ranking** where available.
- **Log top-k candidates with scores** for debugging.

## Implementation Patterns

```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k(query_vec, candidates, k=5):
    scored = [(doc_id, cosine(query_vec, vec)) for doc_id, vec in candidates]
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:k]
```

## Validation Checklist

- Distance metric matches embedding model guidance
- Index and query normalization are consistent
- Score thresholds are documented for downstream usage
- Retrieval logs support offline quality analysis
