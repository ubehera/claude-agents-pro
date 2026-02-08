---
name: subagent-catalog
description: Load when routing tasks to agents, discovering agent capabilities, or composing multi-agent workflows
trigger_keywords: [agent catalog, which agent, find agent, agent capabilities, route task, delegate task, agent discovery, subagent, multi-agent, team formation]
---

# Subagent Catalog Skill

Dynamic agent discovery and routing intelligence for composing multi-agent workflows. Provides the complete agent inventory with capabilities, triggers, and optimal delegation patterns.

## Overview

This skill enables intelligent task routing by providing a searchable catalog of all available agents organized by tier, domain, and capability. Use it to determine which agent(s) should handle a given task, compose parallel work streams, and resolve routing conflicts.

**When to Use**:
- Task requires specialist knowledge beyond current agent
- Multi-domain tasks needing coordinated delegation
- Ambiguous routing where multiple agents could qualify
- Composing parallel work streams for complex features

## Agent Inventory

### Tier 0 — Meta (Orchestration)

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `agent-organizer` | Routing | route, dispatch, which agent, triage | Task routing, agent selection, conflict resolution |
| `orchestration-coordinator` | Orchestration | orchestrate, coordinate, parallel, workflow | Multi-agent coordination, parallel work streams |
| `workflow-validator` | Quality | validate, quality gate, acceptance criteria | Phase transitions, quality gate enforcement |

### Tier 1 — Foundation (Core Engineering)

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `api-platform-engineer` | API | REST, GraphQL, OpenAPI, gateway, API design | API design, gateway setup, developer portals |
| `code-reviewer` | Quality | review, code quality, PR review, bugs | Code review, quality audits, PR analysis |
| `dependency-manager` | Dependencies | dependency, CVE, update, vulnerability | Dependency updates, security remediation |
| `domain-modeling-expert` | DDD | bounded context, event storming, aggregate | Strategic DDD, domain modeling, context mapping |
| `error-diagnostician` | Debugging | error, stack trace, debug, crash, diagnose | Runtime errors, test failures, system issues |
| `performance-optimization-specialist` | Performance | slow, optimize, Core Web Vitals, latency | Performance profiling, bottleneck identification |
| `refactoring-specialist` | Refactoring | refactor, technical debt, extract, simplify | Safe refactoring, code modernization |
| `system-design-specialist` | Architecture | system design, distributed, scalability | Large-scale architecture, distributed systems |
| `test-engineer` | Testing | test, coverage, TDD, integration test | Test suites, testing strategies, coverage |

### Tier 2 — Development (Language/Platform Specialists)

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `angular-expert` | Frontend | Angular, signals, standalone, NgRx | Angular 17+ apps, signal migration |
| `bash-expert` | Shell | bash, shell script, sed, awk, cron | Shell scripting, automation, CLI tools |
| `csharp-expert` | .NET | C#, .NET, ASP.NET, Entity Framework, Blazor | .NET 8+ apps, minimal APIs, EF Core |
| `django-expert` | Python Web | Django, DRF, Celery, Django ORM | Django 5+ apps, DRF APIs, async views |
| `frontend-expert` | Frontend | React, Vue, CSS, responsive, accessibility | React/Vue/Angular apps, Web Vitals, a11y |
| `go-expert` | Go | Go, goroutine, channel, Kubernetes operator | Go services, concurrency, cloud-native |
| `java-expert` | Java | Java, Spring, Maven, Gradle, virtual threads | Enterprise Java 21+, Spring Boot 3.x |
| `kotlin-expert` | Kotlin | Kotlin, coroutines, Compose, KMP, Ktor | Kotlin 2.0+, multiplatform, Android |
| `laravel-expert` | PHP | Laravel, Eloquent, Livewire, PHP | Laravel 11+, PHP 8.3+, Horizon queues |
| `mobile-specialist` | Mobile | iOS, Android, React Native, Flutter, SwiftUI | Cross-platform mobile, native modules |
| `python-expert` | Python | Python, FastAPI, Django, pandas, asyncio | Python 3.11+ services, data workflows |
| `rails-expert` | Ruby | Rails, Hotwire, Turbo, Active Record | Rails 7.1+, Turbo Streams, Stimulus |
| `rust-expert` | Rust | Rust, ownership, async Tokio, WebAssembly | Systems programming, performance-critical |
| `spring-boot-expert` | Java | Spring Boot, Spring Cloud, WebFlux, Reactor | Spring Boot 3.x microservices, resilience |
| `typescript-architect` | TypeScript | TypeScript, type safety, monorepo, Bun | TS platform upgrades, type sharing, DX |

### Tier 3 — Specialists (Domain Experts)

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `aws-cloud-architect` | AWS | AWS, CloudFormation, CDK, Lambda, ECS | AWS infrastructure, Well-Architected |
| `azure-cloud-architect` | Azure | Azure, Bicep, AKS, Azure Functions, Entra | Azure infrastructure, Landing Zones |
| `backend-architect` | Backend | service design, microservices, messaging | End-to-end backend, modernization |
| `chaos-engineer` | Resilience | chaos, failure injection, game day | Resilience testing, disaster recovery |
| `data-pipeline-engineer` | Data | Spark, Airflow, ETL, streaming, dbt | Data pipelines, warehousing, quality |
| `database-architect` | Database | PostgreSQL, schema, migration, sharding | Schema design, query optimization, DR |
| `devops-automation-expert` | DevOps | CI/CD, GitHub Actions, GitOps, ArgoCD | Pipeline setup, deployment strategies |
| `event-driven-architect` | Events | event sourcing, CQRS, Kafka, saga | Event-driven systems, message-driven |
| `full-stack-architect` | Full-Stack | full-stack, Next.js, state management | Full-stack apps, API integration |
| `gcp-cloud-architect` | GCP | GCP, GKE, Cloud Run, BigQuery, Spanner | GCP infrastructure, SRE principles |
| `kubernetes-architect` | K8s | Kubernetes, Helm, Istio, service mesh | Cluster architecture, operator dev |
| `observability-engineer` | Observability | metrics, Prometheus, tracing, SLO | Instrumentation, dashboards, alerting |
| `sre-incident-responder` | SRE | incident, on-call, postmortem, outage | Incident response, runbook automation |
| `terraform-expert` | IaC | Terraform, OpenTofu, state, module | Infrastructure as Code, multi-cloud |

### Tier 4 — Experts (Advanced Specialists)

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `llm-architect` | AI/ML | RAG, LLM, agent framework, AI-native | AI system design, RAG pipelines |
| `machine-learning-engineer` | ML | PyTorch, model training, MLOps, MLflow | ML pipelines, model deployment |
| `prompt-engineer` | AI | prompt design, few-shot, chain-of-thought | Prompt optimization, evaluation |

### Tier 5 — Platform

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `dx-optimizer` | DX | developer experience, onboarding, tooling | DX optimization, workflow friction |
| `git-workflow-manager` | Git | branch strategy, GitFlow, conventional commits | Git workflows, release management |

### Tier 6 — Integration

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `product-owner` | Requirements | user story, backlog, acceptance criteria | Requirements, story writing, prioritization |
| `research-librarian` | Research | RFC, specification, vendor docs, compare | Source discovery, comparative analysis |
| `tech-writer` | Documentation | user guide, tutorial, API docs, quickstart | End-user docs, developer onboarding |
| `technical-documentation-specialist` | Docs Quality | ADR, README, architecture diagram, runbook | Doc review, structure improvement |

### Tier 7 — Quality

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `security-architect` | Security | OWASP, threat model, encryption, compliance | Security review, threat analysis |

### Tier 8 — Finance

| Agent | Domain | Triggers | Best For |
|-------|--------|----------|----------|
| `algorithmic-trading-engineer` | Execution | broker API, order management, TWAP, VWAP | Live trading, broker integration |
| `equity-research-analyst` | Research | DCF, P/E, financial statement, valuation | Fundamental analysis, stock screening |
| `finance-glossary` | Reference | finance term, trading term, definition | Canonical terminology, domain boundaries |
| `market-data-engineer` | Data | market data, OHLCV, WebSocket feed | Data feeds, time-series storage |
| `portfolio-manager` | Portfolio | portfolio, allocation, rebalance, attribution | Multi-strategy portfolios, capital allocation |
| `quantitative-analyst` | Quant | RSI, MACD, Greeks, volatility, GARCH | Technical analysis, quant research |
| `trading-compliance-officer` | Compliance | PDT, wash sale, FINRA, SEC, tax | Regulatory compliance, audit trails |
| `trading-ml-specialist` | ML Trading | ML trading, feature engineering, walk-forward | ML-enhanced strategies, price prediction |
| `trading-risk-manager` | Risk | position sizing, Kelly, VaR, drawdown | Risk assessment, portfolio optimization |
| `trading-strategy-architect` | Strategy | backtest, strategy design, Sharpe, Sortino | Strategy design, backtesting frameworks |

## Routing Decision Tree

```
1. Is this a multi-domain task?
   YES → orchestration-coordinator (parallel streams)
   NO  → continue

2. Identify primary domain:
   - Language-specific → Tier 2 specialist
   - Infrastructure/Cloud → Tier 3 specialist
   - AI/ML → Tier 4 expert
   - Finance/Trading → Tier 8 specialist
   - Architecture → system-design-specialist or backend-architect
   - API → api-platform-engineer
   - Security → security-architect
   - Testing → test-engineer
   - Performance → performance-optimization-specialist

3. Does it need review after implementation?
   YES → Chain: implementer → code-reviewer
   NO  → Single agent dispatch

4. Does it cross bounded contexts?
   YES → domain-modeling-expert first, then specialists
   NO  → Direct to specialist
```

## Composition Patterns

### Sequential Chain
```
Requirements → Implementation → Review → Deploy
product-owner → [specialist] → code-reviewer → devops-automation-expert
```

### Parallel Fan-Out
```
Full-Stack Feature:
  ├── frontend-expert (UI components)
  ├── backend-architect (API + services)
  ├── database-architect (schema + migrations)
  └── test-engineer (test suites)
  → orchestration-coordinator (synthesize)
```

### Review Pipeline
```
Implementation → code-reviewer → security-architect → test-engineer
```

### Domain-Specific Teams

**Trading System**: trading-strategy-architect + quantitative-analyst + market-data-engineer + algorithmic-trading-engineer + trading-risk-manager

**Cloud Migration**: aws-cloud-architect (or azure/gcp) + terraform-expert + kubernetes-architect + devops-automation-expert + observability-engineer

**New Microservice**: domain-modeling-expert + backend-architect + api-platform-engineer + database-architect + test-engineer

## Best Practices

1. **Least-privilege routing**: Choose the most specific agent, not the most powerful
2. **Chain reviews**: Always include code-reviewer for production changes
3. **Domain first**: Start with domain-modeling-expert for new bounded contexts
4. **Parallel when independent**: Fan out to multiple specialists when tasks don't depend on each other
5. **Single source of truth**: One agent owns each deliverable; others review

---

**Skill Type**: Agentic — Discovery
**Complexity**: Moderate
**Typical Usage**: Task routing, agent selection, workflow composition
**Agent Count**: 61 across 9 tiers
