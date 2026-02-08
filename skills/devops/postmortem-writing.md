---
name: postmortem-writing
description: Use when documenting incidents with blameless analysis, concrete timelines, root causes, and prevention actions that improve reliability.
trigger_keywords: [postmortem writing, blameless postmortem, incident report, root cause analysis, five whys, reliability review]
---

# Postmortem Writing

Use this skill to produce clear, blameless postmortems that convert incidents into lasting system improvements.

## When to Use This Skill

- After production incidents and near-miss events
- During reliability review cycles
- When aligning engineering and product on follow-up actions
- When building an incident knowledge base

## Core Concepts

- **Blameless language** focuses on systems, not individuals.
- **Timeline fidelity** improves root-cause precision.
- **Contributing factors** are separated from direct trigger.
- **Action items require owners and deadlines**.

## Implementation Patterns

```markdown
# Incident Postmortem: INC-2026-014
## Summary
- Impact: Checkout failures for 23% of requests for 18 minutes
- Customer impact window: 14:03 UTC to 14:21 UTC

## Timeline (UTC)
- 14:03 Alert fired (error budget burn-rate)
- 14:08 Incident declared, rollback initiated
- 14:21 Service restored

## Root Cause
Connection pool exhaustion due to unbounded retry storm in downstream client.

## Corrective Actions
1. Add retry budget + jitter (Owner: Platform, Due: 2026-02-15)
2. Add saturation alert for pool usage >85% (Owner: SRE, Due: 2026-02-12)
```

## Validation Checklist

- Timeline includes detection, mitigation, and resolution events
- Root cause and contributing factors are distinct
- Every action item has owner, due date, and success criteria
- Learning is reflected in runbooks or automation updates
