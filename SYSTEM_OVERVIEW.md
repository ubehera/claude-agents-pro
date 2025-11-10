# System Overview

## Architecture Philosophy

Claude Agents Pro is a **tiered agent collection** designed for production-grade software engineering with specialized expertise in algorithmic trading and quantitative finance. The system balances breadth (33 agents across 8 domains) with depth (9 finance-specific specialists forming complete trading workflows).

### Design Principles

1. **Hierarchical Specialization**: Agents organized by complexity and domain expertise (Meta → Foundation → Development → Specialists → Experts)
2. **Tool Inheritance**: All agents access full toolset for maximum flexibility (no explicit `tools:` restrictions)
3. **Quality-First**: Automated validation (verify-agents.sh) and scoring (quality-scorer.py) with 70/100 minimum, 85/100 production threshold
4. **Multi-Agent Coordination**: Complex workflows decomposed through orchestration-coordinator with specialist delegation
5. **MCP Integration**: Persistent memory and sequential-thinking for session continuity and complex reasoning

---

## Tier Architecture

### Tier 0: Meta (Orchestration)
**Purpose**: Multi-agent workflow coordination and task decomposition

- **orchestration-coordinator**: Breaks complex problems into specialist tasks, manages delegation patterns (sequential, parallel, review, iterative)

### Tier 1: Foundation (Core Engineering)
**Purpose**: Fundamental software engineering capabilities applicable across domains

- **api-platform-engineer**: REST/GraphQL design, OpenAPI, API governance
- **domain-modeling-expert**: Strategic DDD, event storming, bounded contexts
- **code-reviewer**: Quality gates, OWASP security, architecture assessment
- **error-diagnostician**: Production debugging, stack trace analysis, root cause identification
- **performance-optimization-specialist**: Profiling, bottleneck identification, Core Web Vitals
- **system-design-specialist**: Distributed systems, scalability, CAP theorem
- **test-engineer**: Test automation, coverage analysis, test pyramid

### Tier 2: Development (Language/Platform Specialists)
**Purpose**: Deep expertise in specific languages and frameworks

- **frontend-expert**: React 18+, Vue 3, Angular 17+, accessibility, responsive design
- **mobile-specialist**: iOS (SwiftUI), Android (Kotlin), React Native, Flutter
- **python-expert**: Python 3.11+, FastAPI, async workers, type safety
- **typescript-architect**: Node.js 20+, Bun, Deno, monorepo design

### Tier 3: Specialists (Infrastructure & Data)
**Purpose**: Domain-specific technical expertise

- **aws-cloud-architect**: CloudFormation, CDK, Lambda, ECS/EKS, Well-Architected Framework
- **backend-architect**: Microservices, event-driven systems, API design
- **data-pipeline-engineer**: Apache Spark, Airflow, Kafka, ETL/ELT
- **database-architect**: PostgreSQL, MySQL, MongoDB, migrations, query optimization
- **devops-automation-expert**: CI/CD (GitHub Actions, GitLab CI), Terraform, Ansible
- **full-stack-architect**: End-to-end web applications, React/Next.js + Node.js/FastAPI
- **observability-engineer**: Prometheus, Grafana, OpenTelemetry, SLO design
- **sre-incident-responder**: High-severity incident response, runbook automation

### Tier 4: Experts (ML & AI)
**Purpose**: Machine learning and advanced AI systems

- **machine-learning-engineer**: PyTorch, TensorFlow, MLOps, model serving, distributed training

### Tier 6: Integration (Research & Documentation)
**Purpose**: Knowledge discovery and technical communication

- **research-librarian**: Primary-source research, RFC analysis, comparative studies
- **technical-documentation-specialist**: ADRs, API docs, C4 diagrams, documentation quality

### Tier 7: Quality (Security & Compliance)
**Purpose**: Security architecture and regulatory compliance

- **security-architect**: Threat modeling, OWASP Top 10, GDPR/PCI-DSS/SOC2, penetration testing

### Tier 8: Finance (Trading & Quantitative Analysis)
**Purpose**: Complete systematic trading system from data to execution

**Data Pipeline**:
- **market-data-engineer**: Real-time/historical data (Alpaca, Fidelity, E*TRADE), TimescaleDB, QuestDB, data quality

**Analysis & Strategy**:
- **quantitative-analyst**: Technical indicators (RSI, MACD, Bollinger), options Greeks, statistical models
- **equity-research-analyst**: Fundamental analysis, DCF, P/E, financial statement analysis
- **trading-ml-specialist**: ML for trading, walk-forward validation, overfitting detection

**Risk & Execution**:
- **trading-strategy-architect**: Backtesting (vectorbt, backtrader), walk-forward analysis, Sharpe >1.5
- **trading-risk-manager**: Position sizing (Kelly criterion), portfolio optimization, VaR/CVaR
- **algorithmic-trading-engineer**: Order execution (TWAP, VWAP), multi-broker integration, OMS
- **portfolio-manager**: Multi-strategy allocation, rebalancing, performance attribution

**Compliance**:
- **trading-compliance-officer**: PDT rules, wash sales, FINRA/SEC compliance, 1099-B reporting

---

## Multi-Agent Coordination Patterns

### 1. Direct Delegation (Simple Tasks)
```
User Request → Single Specialist Agent → Result
Example: "Design a REST API" → api-platform-engineer
```

### 2. Sequential Workflow (Phased Execution)
```
orchestration-coordinator → Agent 1 → Agent 2 → Agent 3
Example: DDD Workflow
  domain-modeling-expert (bounded contexts)
  ↓
  api-platform-engineer (API contracts)
  ↓
  database-architect (data models)
  ↓
  test-engineer (test suite)
```

### 3. Parallel Delegation (Independent Tasks)
```
orchestration-coordinator
  ├─→ Agent A (parallel)
  ├─→ Agent B (parallel)
  └─→ Agent C (parallel)

Example: Feature Development
  ├─→ backend-architect (API implementation)
  ├─→ frontend-expert (UI components)
  └─→ database-architect (schema design)
```

### 4. Review Pattern (Quality Gates)
```
Implementation Agent → Review Agent → Approval/Feedback Loop

Example: Code Quality Workflow
  python-expert (implementation)
  ↓
  code-reviewer (quality gate)
  ↓
  test-engineer (test validation)
  ↓
  security-architect (security scan)
```

### 5. Iterative Refinement (Complex Problems)
```
orchestration-coordinator → Specialist → Review → Refine → Repeat

Example: Performance Optimization
  performance-optimization-specialist (analyze)
  ↓
  code-reviewer (validate changes)
  ↓
  performance-optimization-specialist (refine)
  ↓
  Repeat until threshold met
```

### 6. Finance Multi-Agent Workflow (Complete Trading System)
```
market-data-engineer (data pipeline)
  ↓
quantitative-analyst (technical analysis) + equity-research-analyst (fundamentals)
  ↓
trading-ml-specialist (ML signals)
  ↓
trading-strategy-architect (backtest validation)
  ↓
trading-risk-manager (position sizing, portfolio optimization)
  ↓
algorithmic-trading-engineer (order execution)
  ↓
trading-compliance-officer (regulatory validation)
  ↓
portfolio-manager (capital allocation)
```

---

## Quality Framework

### Automated Validation

**verify-agents.sh** (Structural Validation):
- YAML frontmatter structure
- Required fields: `name`, `description`
- Name-filename matching
- Tool declaration warnings

**quality-scorer.py** (Comprehensive Scoring):
- **Completeness (25%)**: Content coverage, thoroughness
- **Accuracy (25%)**: Technical correctness, best practices
- **Usability (20%)**: Clear instructions, invocation patterns
- **Performance (15%)**: Tool optimization, efficiency
- **Maintainability (15%)**: Documentation quality, structure

**Thresholds**:
- Minimum new agents: 70/100
- Production-ready: 85/100
- Tier-specific targets: Meta ≥9.0, Foundation ≥8.0, Specialists ≥7.5

### CI/CD Integration

**GitHub Actions Workflows**:
- `test-agents.yml`: Automated agent testing
- `validate-agents.yml`: Frontmatter validation on PRs
- `update-registry.yml`: Auto-generate agent metadata

---

## Tool Access Model

### Philosophy: Full Inheritance

**Design Choice**: Agents inherit all Claude Code tools without explicit `tools:` field restrictions.

**Rationale**:
- **Flexibility**: No permission errors during complex workflows
- **Maintenance**: No tool list updates when capabilities expand
- **Trust**: Claude selects appropriate tools based on task context

**Alternative**: For least-privilege deployments, fork agents and add explicit `tools:` fields (pattern used in VoltAgent repository).

### Typical Tool Patterns

**Foundation Agents**: Read, Write, MultiEdit, Bash, Grep, Task, WebSearch (6-7 tools)
**Development Agents**: Read, Write, MultiEdit, Bash, Task, WebSearch (5-6 tools)
**Specialist Agents**: Read, Write, MultiEdit, Bash, Task (4-5 tools)
**Finance Agents**: Read, Write, MultiEdit, Bash, Task (lean 4-6 tools despite complexity)

---

## MCP Integration

### Configured Servers

**.mcp.json** (Project-Level):
- **memory**: Persistent knowledge graph for session continuity, project context
- **sequential-thinking**: Complex problem decomposition, multi-step reasoning

### Optional Servers (Disabled by Default)

Users can enable additional servers by editing `.mcp.json`:
- **aws-docs**: AWS documentation retrieval
- **codex**: Advanced code generation (when available)
- **playwright**: Browser automation
- **context7**: Library documentation

### Memory Usage Patterns

**Entity Types**:
- **ArchitecturalDecision**: Major technical choices with rationale
- **Project**: Project context, tech stack, goals
- **WorkflowPhase**: Phase outcomes from multi-agent workflows
- **CodePattern**: Reusable patterns, anti-patterns, best practices

**Relations**:
- `depends_on`: Dependency relationships
- `implements`: Implementation of spec/contract
- `follows`: Sequential workflow connections
- `contributes_to`: Parallel work feeding into larger features
- `validates`: Review/testing relationships

---

## Slash Commands (35 Workflows)

### Command Organization

**00-meta/**: Orchestration workflows (multi-agent coordination, team formation, parallel execution)
**01-foundation/**: Core engineering (domain modeling, API design, debugging)
**03-agents/**: Agent management (`/agent`, `/agents`)
**05-utilities/**: Developer tools (search, documentation, quick fixes)
**automation/**: Deploy pipelines, batch updates, health checks
**git/**: Workflow automation (branch strategies, PR automation)
**quality/**: Test, review, security audit, performance, verify/score agents
**workflows/**: Feature development, full-stack workflows, review-deploy pipelines

### Key Commands

- `/00-meta:orchestrate`: Multi-agent workflow orchestration
- `/01-foundation:api`: API design with OpenAPI
- `/01-foundation:domain-model`: Strategic DDD with event storming
- `/quality:test`: Comprehensive testing with coverage gates
- `/quality:review`: Code review with quality gates
- `/quality:security-audit`: OWASP compliance and vulnerability assessment
- `/quality:verify-agents`: Run agent validation suite
- `/quality:score-agents`: Execute quality scoring
- `/workflows:feature-development`: Complete DDD workflow
- `/workflows:full-stack-feature`: End-to-end feature with testing

---

## Extension Points

### Adding New Agents

1. **Create agent file** in appropriate tier directory (`agents/XX-tier/agent-name.md`)
2. **Add frontmatter**:
   ```yaml
   ---
   name: agent-name
   description: Detailed description with technologies and "Use for..." triggers
   category: [orchestrator|foundation|development|specialist|expert|integration|quality|finance]
   complexity: [simple|moderate|complex|expert]
   model: claude-sonnet-4-5-20250929
   model_rationale: Brief justification for model selection
   capabilities: [list, of, key, capabilities]
   auto_activate:
     keywords: [trigger, words]
     conditions: [when, to, activate]
   ---
   ```
3. **Do NOT add `tools:` field** - agents inherit all tools automatically
4. **Update catalogs**:
   - `agents/README.md`: Add row to Active Agents table
   - `configs/agent-metadata.json`: Add metadata entry
   - `AGENTS.md`: Update top-level catalog
5. **Validate**: Run `./scripts/verify-agents.sh` and `python3 scripts/quality-scorer.py --agent agents/XX-tier/agent-name.md`
6. **Test**: Follow `agents/TESTING.md` procedures

### Adding Skills (Progressive Disclosure)

1. **Create skills directory**: `skills/domain/`
2. **Extract knowledge** from dense agents into skill files:
   ```yaml
   ---
   name: skill-name
   description: When to activate this skill
   trigger_keywords: [async, await, asyncio]
   ---

   ## Core Concepts
   [Instructions loaded when skill activated]

   ## Examples
   [Detailed examples loaded on-demand]
   ```
3. **Update agent frontmatter**:
   ```yaml
   skills:
     - skill-name-1
     - skill-name-2
   ```
4. **Token Savings**: Baseline agent stays lean, skills load progressively

---

## Success Metrics

### Quality Indicators
- Agent validation pass rate: 100% (all agents pass verify-agents.sh)
- Average quality score: 8.2/10 (exceeds 7.5 specialist threshold)
- Production-ready agents: 33/33 (100% meet 85/100 threshold)

### Coverage Metrics
- Foundation tier: 7 agents (core engineering)
- Development tier: 4 agents (major language ecosystems)
- Specialist tier: 8 agents (infrastructure & data)
- Finance tier: 9 agents (complete trading system)

### Automation Metrics
- CI/CD validation: GitHub Actions on all PRs
- Quality scoring: Automated rubric-based evaluation
- Agent installation: One-command deployment (`install-agents.sh`)

---

## References

- **Agent Catalog**: `agents/README.md`
- **Testing Procedures**: `agents/TESTING.md`
- **Pre-Flight Checklist**: `agents/AGENT_CHECKLIST.md`
- **Contribution Guide**: `CONTRIBUTING.md`
- **Changelog**: `CHANGELOG.md`
- **Delegation Patterns**: `patterns/orchestration/delegation-patterns.md`
- **Quality Scoring**: `scripts/VALIDATION_README.md`

---

**Last Updated**: 2025-01-09
**Agent Count**: 33
**Quality Framework**: Validated & Production-Ready
