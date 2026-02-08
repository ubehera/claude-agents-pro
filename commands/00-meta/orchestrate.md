---
description: Multi-agent workflow orchestration for complex tasks
args: <task-description> [--agents list] [--strategy sequential|parallel]
tools: Task, TodoWrite, Read, Write
model: opus
---

# /orchestrate - Multi-Agent Workflow Orchestration

Orchestrates complex workflows by decomposing tasks and coordinating multiple specialized agents with dependency management and quality gates.

## Usage

```bash
/orchestrate <workflow_type> [options]
```

## Core Workflow Types

### Sequential Workflows
- `/orchestrate feature --sequential` - Complete feature development pipeline
- `/orchestrate api --sequential` - API design → implementation → testing → docs
- `/orchestrate migration --sequential` - Research → plan → implement → validate → deploy

### Parallel Workflows
- `/orchestrate fullstack --parallel` - Frontend + backend development concurrently
- `/orchestrate review --parallel` - Code + security + performance reviews simultaneously
- `/orchestrate deployment --parallel` - Infrastructure + application + monitoring setup

### Iterative Workflows
- `/orchestrate design --iterative` - Design → feedback → refinement cycles
- `/orchestrate optimization --iterative` - Profile → optimize → validate loops
- `/orchestrate research --iterative` - Research → analysis → synthesis iterations

## Team Composition Commands

### Foundation Team (Tier 1)
```bash
/orchestrate foundation-review
# Coordinates: api-platform-engineer, code-reviewer, test-engineer, performance-optimization-specialist
```

### Full-Stack Team (Tiers 1-3)
```bash
/orchestrate fullstack-team
# Coordinates: system-design-specialist, frontend-expert, backend-architect, database-architect
```

### Platform Team (Tier 3)
```bash
/orchestrate platform-team
# Coordinates: aws-cloud-architect, devops-automation-expert, observability-engineer, sre-incident-responder
```

### Quality Gate Team
```bash
/orchestrate quality-gates
# Coordinates: code-reviewer, test-engineer, performance-optimization-specialist, security-architect
```

## Agent Delegation Patterns

### Primary → Review Pattern
```bash
/orchestrate delegate-review <domain>
# 1. Primary specialist implements
# 2. Domain reviewer validates
# 3. Cross-domain expert checks integration
# 4. Quality gate enforcement
```

### Research → Implementation Pattern
```bash
/orchestrate research-implement <topic>
# 1. research-librarian gathers requirements
# 2. Appropriate specialist implements
# 3. test-engineer validates
# 4. Documentation update
```

### Design → Build → Deploy Pattern
```bash
/orchestrate design-build-deploy
# 1. system-design-specialist creates architecture
# 2. Domain specialists implement
# 3. devops-automation-expert handles deployment
# 4. observability-engineer adds monitoring
```

## Context Management

### Context Passing Between Agents
The orchestrator maintains shared context through:
- Task dependency mapping
- Inter-agent communication protocols
- Shared artifact repositories
- Quality gate checkpoints

### Progress Tracking Integration
- TodoWrite integration for multi-step workflows
- Agent performance monitoring
- Quality metric tracking
- Dependency resolution status

## Quality Gate Integration

### Automated Quality Gates
- **Code Quality**: >80% coverage, linting compliance
- **Security**: Vulnerability scanning, threat model review
- **Performance**: Latency targets, load testing validation
- **Architecture**: Design pattern compliance, dependency analysis

### Quality Gate Enforcement
```bash
/orchestrate with-gates <workflow>
# Enforces quality thresholds at each stage
# Blocks progression until gates pass
# Escalates to human review for gate failures
```

## Example Complex Workflows

### Complete Feature Development
```bash
/orchestrate feature "user authentication system"
# 1. research-librarian: Security standards research
# 2. system-design-specialist: Architecture design
# 3. api-platform-engineer: API specification
# 4. backend-architect: Service implementation
# 5. frontend-expert: UI implementation
# 6. test-engineer: Comprehensive test suite
# 7. security-architect: Security review
# 8. devops-automation-expert: CI/CD pipeline
# 9. observability-engineer: Monitoring setup
# Quality gates at steps 6, 7, 9
```

### Performance Optimization Campaign
```bash
/orchestrate optimize "application performance"
# 1. performance-optimization-specialist: Profiling analysis
# 2. database-architect: Query optimization (if needed)
# 3. backend-architect: Service optimization (if needed)
# 4. frontend-expert: Client optimization (if needed)
# 5. aws-cloud-architect: Infrastructure optimization
# 6. observability-engineer: Monitoring improvements
# 7. test-engineer: Performance test validation
```

### Security Audit & Remediation
```bash
/orchestrate security-audit
# 1. security-architect: Threat model analysis
# 2. code-reviewer: Security-focused code review
# 3. devops-automation-expert: Infrastructure security scan
# 4. backend-architect: API security validation
# 5. database-architect: Data protection review
# 6. test-engineer: Security test implementation
# Quality gates: Security scan pass, penetration test validation
```

## Advanced Orchestration Features

### Load Balancing
- Distribute tasks based on agent expertise and availability
- Prevent agent overloading through intelligent routing
- Dynamic workload redistribution based on performance metrics

### Error Recovery
- Automatic failover to secondary agents for critical tasks
- Error pattern learning for improved routing decisions
- Escalation protocols for unresolvable issues

### Performance Optimization
- Agent performance monitoring and optimization
- Workflow efficiency analysis and improvement
- Resource utilization optimization across agent teams

## Configuration

### Agent Priority Matrix
```yaml
primary_agents:
  api_design: api-platform-engineer
  system_architecture: system-design-specialist
  security_review: security-architect
  performance_analysis: performance-optimization-specialist

review_agents:
  code_quality: code-reviewer
  test_strategy: test-engineer
  infrastructure: devops-automation-expert

fallback_agents:
  general_implementation: backend-architect
  research_support: research-librarian
```

### Quality Thresholds
```yaml
quality_gates:
  code_coverage: 80%
  security_scan: PASS
  performance_p95: <200ms
  architecture_review: APPROVED
```

The orchestration system ensures optimal agent collaboration while maintaining quality standards and efficient resource utilization.