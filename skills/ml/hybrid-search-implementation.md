---
name: hybrid-search-implementation
description: Use when combining semantic vector search with keyword or lexical retrieval to improve relevance and robustness for production RAG systems.
trigger_keywords: [hybrid search, lexical plus vector search, bm25 plus embeddings, retrieval fusion, rag hybrid retrieval]
---

# Hybrid Search Implementation

Use this skill to combine lexical precision and semantic recall in one retrieval pipeline.

## When to Use This Skill

- Pure vector search misses exact terms or IDs
- Keyword search misses paraphrased intent
- High-stakes retrieval requires robust fallback behavior
- Designing retrieval for mixed technical and natural-language queries

## Core Concepts

- **Lexical branch** captures exact-match intent.
- **Vector branch** captures semantic intent.
- **Rank fusion** combines both signals consistently.
- **Reranking** should happen after candidate merge.

## Implementation Patterns

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

# usage:
# lexical_ids = bm25_search(query)
# vector_ids = vector_search(query_embedding)
# fused = reciprocal_rank_fusion([lexical_ids, vector_ids])
```

## Validation Checklist

- Fusion strategy is deterministic and tested
- Candidate pools from both branches are logged
- Relevance evaluation includes exact and paraphrase queries
- Failure mode fallback (lexical-only/vector-only) is defined
