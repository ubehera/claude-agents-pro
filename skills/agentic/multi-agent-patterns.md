---
name: multi-agent-patterns
description: Load when designing multi-agent workflows, coordinating parallel agents, or implementing agent communication patterns
trigger_keywords: [multi-agent, agent coordination, parallel agents, agent pipeline, fan-out, agent orchestration, workflow pattern, agent composition]
---

# Multi-Agent Patterns Skill

Production patterns for coordinating multiple agents in complex workflows. Covers sequential pipelines, parallel fan-out, review chains, and error recovery.

## Overview

Multi-agent coordination enables complex tasks to be decomposed and handled by domain specialists working in concert. This skill provides the patterns, anti-patterns, and decision frameworks for effective agent orchestration.

**When to Use**:
- Task touches 3+ domains (e.g., frontend + backend + database)
- Multiple independent subtasks can run in parallel
- Sequential quality gates required (implement → review → test)
- Complex features requiring iterative refinement across specialists

## Core Patterns

### 1. Sequential Pipeline

Agents execute in order, each building on the previous output.

```
Trigger: Feature with clear phases (design → implement → test → deploy)

Flow:
  product-owner          → Requirements & acceptance criteria
  domain-modeling-expert → Domain model & bounded contexts
  api-platform-engineer  → API contracts (OpenAPI spec)
  backend-architect      → Implementation
  test-engineer          → Test suite
  code-reviewer          → Quality validation

Context Passing: Each agent receives summary of prior agent output
Error Handling: If any agent fails, halt pipeline and report
```

**When to Use**: Well-understood workflows with clear dependencies.

**Anti-Pattern**: Don't pipeline when subtasks are independent — use fan-out instead.

### 2. Parallel Fan-Out / Fan-In

Independent subtasks dispatched simultaneously, results synthesized.

```
Trigger: Full-stack feature, multi-service change, comprehensive audit

Fan-Out (parallel):
  ├── frontend-expert     → UI components
  ├── backend-architect   → Service logic
  ├── database-architect  → Schema design
  └── test-engineer       → Test strategy

Fan-In (synthesize):
  orchestration-coordinator → Integrate outputs, resolve conflicts
```

**When to Use**: Tasks with 3+ independent work streams.

**Anti-Pattern**: Don't fan-out when agents need each other's output — sequence instead.

### 3. Review Chain

Primary agent implements, review agents validate.

```
Trigger: Production code changes, security-sensitive features

Chain:
  [specialist]           → Implementation
  code-reviewer          → Quality review (bugs, patterns, style)
  security-architect     → Security review (OWASP, auth, injection)
  test-engineer          → Coverage review (gaps, edge cases)

Iteration: If review finds issues, loop back to specialist
Max Iterations: 3 before escalating to user
```

**When to Use**: Any production-bound change.

**Anti-Pattern**: Don't skip security review for auth/payment/PII features.

### 4. Expert Consultation

Specialist consulted for specific decisions within larger workflow.

```
Trigger: Architecture decision, technology selection, risk assessment

Flow:
  backend-architect      → "Should we use event sourcing here?"
  event-driven-architect → Consultation response with recommendation
  backend-architect      → Continues with informed decision

Pattern: Delegate specific question, receive answer, resume
```

**When to Use**: Point decisions requiring deep domain knowledge.

### 5. Competitive Evaluation

Multiple agents propose solutions, best selected.

```
Trigger: Multiple valid approaches, architecture decision records

Proposals:
  backend-architect      → Monolith approach
  event-driven-architect → Event-sourced approach
  system-design-specialist → Hybrid approach

Selection:
  agent-organizer        → Evaluate trade-offs, recommend
  [user]                 → Final decision
```

**When to Use**: High-impact architectural decisions with multiple valid paths.

### 6. Iterative Refinement

Agent refines output through multiple passes with feedback.

```
Trigger: Complex implementation requiring progressive improvement

Loop:
  [specialist]           → Draft implementation
  code-reviewer          → Feedback (issues, improvements)
  [specialist]           → Revised implementation
  test-engineer          → Validation

Exit Condition: All reviews pass OR max 3 iterations
```

## Coordination Mechanisms

### Context Passing
```yaml
Between Agents:
  - TodoWrite: Track task state and handoff context
  - Memory: Store decisions in knowledge graph for persistence
  - Summary: Brief context paragraph at delegation start

Template:
  "Context: [1-2 line situation summary]
   Prior Work: [What previous agent delivered]
   Your Task: [Specific deliverable expected]
   Constraints: [Non-negotiable requirements]
   Output Format: [Expected deliverable structure]"
```

### Conflict Resolution
```yaml
Priority Order:
  1. Domain specialist recommendation (highest authority in their domain)
  2. Security architect (veto power on security concerns)
  3. User decision (final arbiter)
  4. Most recent implementation (when no conflict resolution available)

Resolution Steps:
  1. Identify conflicting recommendations
  2. Consult domain specialist for the conflict area
  3. Present trade-offs to user if unresolvable
  4. Document decision in memory as ArchitecturalDecision
```

### Error Recovery
```yaml
Agent Failure:
  1. Don't mark task completed
  2. Capture partial output
  3. Create TodoWrite describing failure and partial state
  4. Try alternative agent OR handle directly with context

Timeout:
  1. Check if agent produced partial output
  2. Capture what exists
  3. Re-delegate with narrower scope OR handle directly

Conflict:
  1. Document both perspectives
  2. Escalate to user with trade-off analysis
  3. Store resolution in memory for future reference
```

## Anti-Patterns

### Over-Orchestration
```
❌ Simple bug fix → agent-organizer → error-diagnostician → code-reviewer → test-engineer
✅ Simple bug fix → error-diagnostician (done)

Rule: If <3 files and single domain, skip orchestration.
```

### Agent Ping-Pong
```
❌ Agent A delegates to B, B delegates back to A
✅ Clear ownership: one agent owns deliverable, others consult

Rule: Every task has exactly one owner agent.
```

### Context Loss
```
❌ Fan-out 5 agents, synthesize without tracking
✅ TodoWrite tracks each agent's deliverable and status

Rule: Always use TodoWrite for multi-agent workflows.
```

### Premature Parallelism
```
❌ Fan-out frontend + backend when API contract not defined
✅ Sequential: API contract first, then parallel implementation

Rule: Establish contracts before parallelizing implementation.
```

## Decision Framework

```
Task Complexity Assessment:
  Single domain, <3 files     → Direct (no orchestration)
  Single domain, 3-10 files   → Sequential pipeline
  Multi-domain, independent    → Parallel fan-out
  Multi-domain, dependent      → Sequential then parallel
  High-risk (security/payment) → Review chain mandatory
  Architecture decision        → Expert consultation or competitive evaluation
```

---

**Skill Type**: Agentic — Orchestration
**Complexity**: Complex
**Typical Usage**: Multi-agent workflow design, agent coordination patterns
