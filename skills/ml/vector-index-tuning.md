---
name: vector-index-tuning
description: Use when optimizing vector index recall, latency, and cost across HNSW/IVF index types and production retrieval workloads.
trigger_keywords: [vector index tuning, hnsw tuning, ivf tuning, ANN performance, vector retrieval latency, recall optimization]
---

# Vector Index Tuning

Use this skill to tune ANN indexes for stable recall/latency under production traffic.

## When to Use This Skill

- Retrieval latency exceeds SLOs
- Recall quality drops after corpus growth
- Sizing index parameters for new workloads
- Planning cost-performance tradeoffs for vector stores

## Core Concepts

- **Tune against workload shape** (query volume, corpus size, filter selectivity).
- **Measure recall against exact-search baseline**.
- **Separate offline tuning from online serving rollout**.
- **Use staged rollout with shadow queries**.

## Implementation Patterns

```yaml
tuning_cycle:
  baseline:
    - build exact-search subset
    - capture p50/p95 latency and recall@k
  candidate_configs:
    hnsw:
      M: [16, 24, 32]
      efConstruction: [100, 200, 400]
      efSearch: [40, 80, 120]
    ivf:
      nlist: [256, 512, 1024]
      nprobe: [8, 16, 32]
  rollout:
    - shadow_traffic
    - canary_5_percent
    - full_rollout_if_slo_passes
```

## Validation Checklist

- Recall@k and latency targets are both met
- Index parameter changes are tracked by version
- Canary and rollback strategy is documented
- Capacity estimates are updated after tuning
