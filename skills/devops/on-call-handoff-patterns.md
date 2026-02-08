---
name: on-call-handoff-patterns
description: Use when handing off active incidents or operational context between on-call rotations to preserve continuity and reduce repeated diagnosis.
trigger_keywords: [on-call handoff, shift handoff, incident handover, pager handoff, operational continuity]
---

# On-Call Handoff Patterns

Use this skill to transfer live operational context without losing progress or increasing risk.

## When to Use This Skill

- End-of-shift handoffs with active incidents
- Escalations between support tiers
- Weekend/weeknight rotation changes
- Follow-up transitions after temporary mitigations

## Core Concepts

- **State snapshot first**: summarize current service health and incident state.
- **Decisions log**: record actions taken and why.
- **Pending risk visibility**: highlight unresolved failure modes.
- **Next actions**: provide concrete, ordered tasks for incoming responder.

## Implementation Patterns

```markdown
## Handoff Packet
- Incident: INC-2026-019
- Current Status: Mitigated, monitoring elevated error baseline
- What Changed:
  - Rolled back checkout service to build 24f9a1
  - Increased read replica capacity
- Risks:
  - Retry storm may return under peak load
- Next Actions:
  1. Validate retry budget patch in staging
  2. Re-enable feature flag only after saturation <70%
- Escalation Condition:
  - If error rate >3% for 5 min, escalate to incident commander
```

## Validation Checklist

- Handoff includes status, decisions, risks, and next actions
- Every pending action has owner or assignee role
- Escalation thresholds are explicit
- Links to dashboards/runbooks are included
