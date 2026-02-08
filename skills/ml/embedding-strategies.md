---
name: embedding-strategies
description: Use when designing embedding pipelines for RAG and semantic search, including chunking, model selection, metadata strategy, and re-indexing workflows.
trigger_keywords: [embedding strategies, embeddings pipeline, chunking strategy, semantic indexing, vector embeddings, rag embeddings]
---

# Embedding Strategies

Use this skill to design robust embedding pipelines with clear quality and cost tradeoffs.

## When to Use This Skill

- Building or refactoring RAG retrieval systems
- Choosing embedding models for domain corpora
- Defining chunking and metadata policies
- Planning re-index and drift management

## Core Concepts

- **Chunk for retrieval intent**, not arbitrary token length.
- **Track embedding provenance** (model, version, chunker settings).
- **Use metadata filtering before vector search** where possible.
- **Measure retrieval quality continuously** with benchmark sets.

## Implementation Patterns

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    chunk_size: int
    chunk_overlap: int
    normalize: bool

CONFIG = EmbeddingConfig(
    model="text-embedding-3-large",
    chunk_size=800,
    chunk_overlap=120,
    normalize=True,
)

def chunk_document(text: str, size: int = CONFIG.chunk_size, overlap: int = CONFIG.chunk_overlap):
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        yield text[i:i + size]
```

## Validation Checklist

- Embedding config is versioned and reproducible
- Retrieval benchmark set is maintained
- Re-index strategy exists for model upgrades
- Metadata filters are defined for high-cardinality fields
