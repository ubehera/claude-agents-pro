---
description: Initialize or continue DDD workflow with quality gates and agent coordination
tools: TodoWrite, Task, Read, Write, MultiEdit
model: opus
args: [phase] [context]
---

# DDD Workflow Orchestration Command

Orchestrates Domain-Driven Design workflows through the agent ecosystem with quality gates and progress tracking.

## Usage Patterns

### Initialize New Feature Workflow
```
/workflow init "user authentication system"
/workflow requirements "JWT-based auth with role-based permissions"
/workflow domain "auth bounded context with user aggregates"
```

### Continue Existing Workflow
```
/workflow continue api-design
/workflow advance testing-phase
/workflow validate security-review
```

### Quality Gate Validation
```
/workflow gate architecture-review
/workflow gate performance-testing
/workflow gate security-audit
```

## Command Logic

1. **Phase Detection**: Analyze current project state and determine workflow phase
2. **Agent Coordination**: Route to appropriate specialist agents based on phase
3. **Progress Tracking**: Use TodoWrite to maintain workflow state and dependencies
4. **Quality Gates**: Enforce completion criteria before phase transitions
5. **Context Management**: Maintain domain context across agent handoffs

## Integration Points

- **orchestration-coordinator**: Primary orchestration for complex workflows
- **TodoWrite**: Progress tracking and task management
- **MCP Memory**: Persistent workflow state and decision history
- **Quality Agents**: Automated validation at each gate

## Workflow Phases

1. **Requirements**: Extract business goals, constraints, acceptance criteria
2. **Domain Modeling**: Bounded contexts, aggregates, events, invariants
   - **Agent**: `domain-modeling-expert` for event storming, context mapping, ubiquitous language
   - **Command**: `/domain-model` for strategic DDD activities
3. **Architecture**: System design, NFRs, technology choices
4. **API Design**: Contract specifications with validation
   - **Agent**: `api-platform-engineer` for API contracts
   - **Command**: `/api` for OpenAPI/GraphQL specification
5. **Implementation**: Code generation with testing
6. **Quality Gates**: Review, testing, security validation
   - **Agents**: `code-reviewer`, `test-engineer`, `security-architect`
   - **Commands**: `/review`, `/test`, `/security-audit`
7. **Deployment**: Infrastructure and monitoring setup
   - **Agents**: `devops-automation-expert`, `observability-engineer`

Each phase maintains 85-95% completion thresholds before advancement.