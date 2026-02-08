# Tier 00: Meta (Orchestration)

Meta agents form the control plane of the agent ecosystem. They handle multi-agent workflow coordination, intelligent task routing, quality gate enforcement, and standards validation across all tiers.

## When to Use Meta Agents

Use these agents when you need to:
- **Orchestrate complex workflows** requiring multiple specialized agents
- **Route tasks** to the optimal agent based on domain and complexity
- **Validate phase completions** against quality gate criteria
- **Decompose large projects** into agent-specific work streams
- **Enforce standards** across deliverables before phase transitions

## Available Agents

### [orchestration-coordinator](orchestration-coordinator.md)
Multi-agent orchestration master for complex workflows requiring coordination between specialized agents. Handles task decomposition, agent routing, dependency management, and quality orchestration.

**Use when:** Complex multi-domain projects, coordinating 3+ agents, managing task dependencies, sequencing parallel and sequential work streams.

### [agent-organizer](agent-organizer.md)
Intelligent agent dispatch and routing meta-layer for determining optimal agent selection. Analyzes task requirements, maps them to agent capabilities, and resolves routing ambiguity.

**Use when:** Uncertain which agent to use, multi-domain requests needing triage, optimizing agent selection, resolving routing conflicts.

### [workflow-validator](workflow-validator.md)
Quality gate enforcement and standards validation specialist. Ensures deliverables meet defined criteria before phase transitions, verifies acceptance criteria, and prevents technical debt accumulation.

**Use when:** Validating phase completion, enforcing quality standards, verifying acceptance criteria, checking compliance with architectural decisions.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Coordinate multi-agent workflows | **orchestration-coordinator** |
| Route a task to the right agent | **agent-organizer** |
| Validate quality gates | **workflow-validator** |

## Common Combinations

**Complex Project Kickoff:**
1. `agent-organizer` --> Triage requirements and identify agents
2. `orchestration-coordinator` --> Decompose and sequence work
3. Foundation/Specialist agents --> Execute tasks
4. `workflow-validator` --> Validate phase completion

**Phase Transition:**
1. `workflow-validator` --> Validate current phase deliverables
2. `orchestration-coordinator` --> Coordinate next phase agents
3. `workflow-validator` --> Verify readiness for next phase

## Best Practices

- **Start with routing**: Use `agent-organizer` when unsure which agent fits the task
- **Orchestrate, don't micromanage**: Let `orchestration-coordinator` handle complex dependencies
- **Gate every phase**: Use `workflow-validator` at phase transitions to prevent quality drift
- **Escalate complexity**: Simple tasks go directly to specialists; multi-domain tasks go through meta tier
