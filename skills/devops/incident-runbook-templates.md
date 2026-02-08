---
name: incident-runbook-templates
description: Use when creating or standardizing runbooks for recurring incidents with clear diagnosis paths, mitigation steps, and escalation rules.
trigger_keywords: [incident runbook, runbook template, on-call runbook, operational playbook, incident mitigation steps]
---

# Incident Runbook Templates

Use this skill to create actionable runbooks that reduce MTTR and improve on-call consistency.

## When to Use This Skill

- Defining first-response procedures for common failures
- Standardizing diagnostics and mitigations across services
- Training new on-call engineers
- Preparing operational controls before major releases

## Core Concepts

- **Symptom-first navigation** for fast triage.
- **Deterministic diagnostics** with exact commands and expected outputs.
- **Safe mitigation order** prioritizing reversible actions.
- **Escalation clarity** with handoff thresholds.

## Implementation Patterns

```markdown
# Runbook: API Latency Spike
## Trigger
- p95 latency > 800ms for 10 minutes

## Diagnose
1. Check error and saturation dashboards
2. Compare current deploy hash vs last known good
3. Verify dependency health

## Mitigate
1. Enable traffic shaping / canary rollback
2. Scale service replicas
3. Disable non-critical background jobs

## Escalate
- Escalate to database owner if pool saturation >90% for 15 minutes
```

## Validation Checklist

- Trigger conditions are objective and measurable
- Steps contain concrete command-level guidance
- Rollback and fallback actions are explicit
- Escalation contacts and thresholds are current
