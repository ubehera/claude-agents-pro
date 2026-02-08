---
name: context-driven-development
description: Use when planning feature delivery with explicit context artifacts so requirements, architecture, and implementation stay aligned across sessions.
trigger_keywords: [context driven development, project context, implementation context, planning artifacts, product context, track planning]
---

# Context-Driven Development

Use this skill to keep delivery aligned by treating context as an explicit artifact.

## When to Use This Skill

- Starting a new initiative or large refactor
- Resuming complex work after context switches
- Aligning product, architecture, and implementation decisions
- Coordinating multi-agent or multi-engineer execution

## Core Concepts

- **Context before code**: define why and what before implementation details.
- **One source per concern**: product goals, technical stack, and workflow rules stay separated.
- **Continuously maintained context**: update artifacts when scope or architecture changes.
- **Traceability**: every implementation task maps back to acceptance criteria.

## Implementation Patterns

```yaml
artifacts:
  product_context:
    fields: [problem, goals, users, success_metrics]
  technical_context:
    fields: [stack, constraints, interfaces, quality_targets]
  workflow_context:
    fields: [delivery_phases, quality_gates, ownership, release_policy]

execution_loop:
  - confirm_context
  - define_spec
  - plan_work
  - implement_slice
  - validate_gate
  - update_context
```

## Validation Checklist

- Context artifacts exist and are versioned in repo
- Acceptance criteria are explicit before coding
- Implementation decisions reference context artifacts
- Context documents are updated after scope changes
