---
description: Invoke a specific agent with context and requirements
args: <agent-name> <task-description>
tools: Task, Read, TodoWrite
model: sonnet
---

# Direct Agent Invocation

Invoke the **$1** agent to handle: **$2**

## Agent Resolution

First, identify the appropriate agent based on the request:
- If $1 matches an exact agent name, use it directly
- If $1 is a domain keyword, select the best matching specialist
- Verify agent exists in the tier structure (00-meta through 09-enterprise)

## Context Preparation

Before delegating to the agent:
1. Gather relevant project context using Read tool
2. Check for existing related work in TodoWrite
3. Identify dependencies and constraints
4. Prepare comprehensive task description

## Agent Invocation

Use Task tool with:
- **subagent_type**: The resolved agent identifier
- **prompt**: Enriched context including:
  - Original task description
  - Project context and constraints
  - Expected deliverables
  - Quality criteria
  - Integration points

## Common Agent Mappings

### Foundation Tier (01)
- `api` → api-platform-engineer
- `review` → code-reviewer
- `debug` → error-diagnostician
- `test` → test-engineer
- `perf` → performance-optimization-specialist
- `design` → system-design-specialist

### Development Tier (02)
- `frontend` → frontend-expert
- `mobile` → mobile-specialist
- `python` → python-expert
- `typescript` → typescript-architect

### Specialist Tier (03)
- `aws`/`cloud` → aws-cloud-architect
- `backend` → backend-architect
- `database`/`db` → database-architect
- `devops` → devops-automation-expert
- `observability` → observability-engineer
- `sre` → sre-incident-responder

### Expert Tier (04+)
- `ml`/`ai` → machine-learning-engineer
- `security` → security-architect
- `research` → research-librarian

## Progress Tracking

Create TodoWrite entry for:
- Agent task delegation
- Expected deliverables
- Follow-up actions

## Examples

```
/agent api-platform-engineer "Design REST API for user management"
/agent python "Optimize data processing pipeline"
/agent security "Audit authentication implementation"
/agent frontend "Create responsive dashboard with React"
```

## Error Handling

If agent not found:
1. Suggest closest matches
2. Recommend using `/team` for complex multi-agent tasks
3. Fall back to orchestration-coordinator for guidance