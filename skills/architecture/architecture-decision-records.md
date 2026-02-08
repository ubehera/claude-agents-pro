---
name: architecture-decision-records
description: Use when creating or reviewing ADRs to capture architecture decisions, alternatives, tradeoffs, and implementation consequences.
trigger_keywords: [architecture decision records, adr, architecture decision log, technical decision record, design decision documentation]
---

# Architecture Decision Records

Use this skill to keep major architecture decisions explicit, reviewable, and traceable over time.

## When to Use This Skill

- Introducing new infrastructure or platform patterns
- Choosing between competing architecture options
- Documenting migrations, deprecations, and reversals
- Reviewing historical decisions during incidents or rewrites

## Core Concepts

- **One decision per ADR** keeps scope clear.
- **Alternatives matter**: capture why options were rejected.
- **Consequences must be concrete** for operations and delivery.
- **Status lifecycle** (proposed, accepted, superseded) prevents stale docs.

## Implementation Patterns

```markdown
# ADR-012: Adopt Hybrid Search for Support RAG

## Status
Accepted (2026-02-08)

## Context
Support search missed exact ticket IDs with vector-only retrieval.

## Decision
Adopt lexical + vector retrieval with reciprocal rank fusion and reranking.

## Alternatives Considered
1. Vector-only with larger context window
2. Lexical-only BM25 with synonym expansion

## Consequences
- Positive: Better recall for mixed query types
- Negative: Extra retrieval complexity and tuning overhead
```

## Validation Checklist

- ADR has status, context, decision, alternatives, consequences
- Decision can be implemented from document alone
- Ownership and review date are present
- Superseded ADRs reference replacement documents
