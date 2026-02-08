---
name: track-management
description: Use when decomposing large delivery efforts into execution tracks with clear ownership, sequencing, dependencies, and quality gates.
trigger_keywords: [track management, execution track, delivery track, implementation sequencing, dependency mapping, phased rollout]
---

# Track Management

Use this skill to organize complex initiatives into predictable, independently executable tracks.

## When to Use This Skill

- Managing multi-phase projects across teams or agents
- Breaking epics into bounded, testable delivery streams
- Coordinating dependencies and parallel work safely
- Monitoring progress with objective quality gates

## Core Concepts

- **Track by outcome**, not by technology layer alone.
- **Explicit dependencies** avoid hidden blockers.
- **Quality gates per track** keep merge and release safe.
- **Short feedback loops** keep plans realistic.

## Implementation Patterns

```markdown
## Track Template

### Track: AUTH-FOUNDATION
- Goal: Ship authentication baseline used by all clients
- Scope: JWT issuance, refresh, RBAC middleware
- Dependencies: DB schema migration complete
- Done When:
  - [ ] API contract approved
  - [ ] Integration tests pass
  - [ ] Security review complete

### Track: CLIENT-INTEGRATION
- Goal: Integrate auth flows in web and mobile clients
- Depends On: AUTH-FOUNDATION
```

```yaml
status_model:
  - planned
  - in_progress
  - blocked
  - in_review
  - done
```

## Validation Checklist

- Every track has a measurable success definition
- Dependencies are directional and acyclic
- Track owners and reviewers are assigned
- Blockers are captured with mitigation actions
