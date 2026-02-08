# Claude Code Agent Collection

Central catalog for Claude Agents Pro. Every agent complies with Claude Code frontmatter requirements and ships with pragmatic guidance.

## Tool Access Philosophy

All agents in this collection **inherit the full Claude Code toolset** for maximum flexibility and developer experience. This intentional design choice prioritizes:

- **No permission errors** during complex workflows
- **Simplified maintenance** - no tool list updates required when new capabilities are added
- **Trust in Claude's judgment** for appropriate tool selection based on task context

The tool sets documented in the table below represent **typical usage patterns** for each agent, not access restrictions. Agents can access any tool when the task requires it.

**For least-privilege deployments**: Fork agents and add explicit `tools:` fields following the pattern used in other repositories (e.g., VoltAgent). This collection optimizes for flexibility over strict access control.

## Active Agents
| Agent | Tier | Domain Focus | Tool Set |
|-------|------|--------------|----------|
| `orchestration-coordinator` | 00-meta | Multi-agent orchestration | Task, Read, Write, MultiEdit |
| `agent-organizer` | 00-meta | Agent dispatch & routing | Task, Read, Grep, Glob |
| `workflow-validator` | 00-meta | Quality gate enforcement | Read, Grep, Glob, Bash, Task |
| `api-platform-engineer` | 01-foundation | API design & governance | Read, Write, MultiEdit, Bash, Grep, WebFetch, Task |
| `code-reviewer` | 01-foundation | Code review & quality gates | Read, Grep, Glob, Task, WebSearch |
| `domain-modeling-expert` | 01-foundation | Strategic DDD & context mapping | Read, Write, MultiEdit, Task, WebSearch |
| `error-diagnostician` | 01-foundation | Production debugging & triage | Read, Grep, Bash, Glob, WebSearch, Task |
| `performance-optimization-specialist` | 01-foundation | End-to-end performance tuning | Read, Write, MultiEdit, Bash, Grep, Task |
| `system-design-specialist` | 01-foundation | Distributed systems architecture | Read, Write, MultiEdit, WebSearch, Task |
| `dependency-manager` | 01-foundation | Dependency analysis & security | Read, Write, Bash, Grep, Glob, Task |
| `refactoring-specialist` | 01-foundation | Safe code refactoring | Read, Write, MultiEdit, Bash, Grep, Task |
| `test-engineer` | 01-foundation | Test strategy & automation | Read, Write, MultiEdit, Bash, Grep, Task |
| `bash-expert` | 02-development | Shell scripting & CLI automation | Read, Write, MultiEdit, Bash, Grep |
| `frontend-expert` | 02-development | Modern web UI engineering | Read, Write, MultiEdit, WebFetch |
| `mobile-specialist` | 02-development | Native + cross-platform mobile | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `python-expert` | 02-development | Python services & libraries | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `go-expert` | 02-development | Go microservices & cloud-native | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `rust-expert` | 02-development | Rust systems programming | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `typescript-architect` | 02-development | TypeScript platforms & tooling | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `angular-expert` | 02-development | Enterprise Angular architecture and performance | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `csharp-expert` | 02-development | C#/.NET platform engineering and ASP.NET services | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `django-expert` | 02-development | Django and DRF web platform delivery | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `java-expert` | 02-development | Enterprise Java architecture and JVM optimization | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `kotlin-expert` | 02-development | Kotlin multiplatform, Android, and Ktor delivery | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `laravel-expert` | 02-development | Laravel architecture, APIs, and Eloquent patterns | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `rails-expert` | 02-development | Ruby on Rails full-stack product delivery | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `spring-boot-expert` | 02-development | Spring Boot microservices and cloud-native Java | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `aws-cloud-architect` | 03-specialists | Cloud architecture on AWS | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `azure-cloud-architect` | 03-specialists | Cloud architecture, governance, and landing zones on Azure | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `gcp-cloud-architect` | 03-specialists | Data-first and AI-enabled architecture on Google Cloud | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `backend-architect` | 03-specialists | Service architecture & APIs | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `data-pipeline-engineer` | 03-specialists | ETL and streaming pipelines | Read, Write, MultiEdit, Bash, Task |
| `database-architect` | 03-specialists | Data modelling & performance | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `devops-automation-expert` | 03-specialists | CI/CD and platform automation | Read, Write, MultiEdit, Bash, Task, Grep |
| `full-stack-architect` | 03-specialists | End-to-end web application delivery | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `observability-engineer` | 03-specialists | Metrics, logging, tracing, SLOs | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `event-driven-architect` | 03-specialists | Event sourcing, CQRS, message systems | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `kubernetes-architect` | 03-specialists | K8s orchestration & cloud-native | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `sre-incident-responder` | 03-specialists | Incident response & reliability | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `terraform-expert` | 03-specialists | Infrastructure as Code & Terraform | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `chaos-engineer` | 03-specialists | Chaos engineering & resilience testing | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `machine-learning-engineer` | 04-experts | MLOps and production ML systems | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `llm-architect` | 04-experts | LLM systems, RAG pipelines, AI architecture | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `prompt-engineer` | 04-experts | Prompt design, optimization, security | Read, Write, MultiEdit, Task, WebSearch |
| `git-workflow-manager` | 05-platform | Git workflows, branch strategies, PR automation | Read, Write, MultiEdit, Bash, Task |
| `dx-optimizer` | 05-platform | Developer experience & productivity | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `research-librarian` | 06-integration | Primary-source research & synthesis | Read, Write, MultiEdit, WebSearch |
| `product-owner` | 06-integration | Requirements & user story creation | Read, Write, MultiEdit, Task, WebSearch |
| `tech-writer` | 06-integration | End-user guides & API documentation | Read, Write, MultiEdit, Grep, WebSearch |
| `technical-documentation-specialist` | 06-integration | Technical writing & doc quality | Read, Write, MultiEdit, Grep, WebFetch |
| `security-architect` | 07-quality | Threat modelling & secure design | Read, Write, MultiEdit, Bash, Grep, WebSearch, Task |
| `market-data-engineer` | 08-finance | Market data pipelines & quality | Read, Write, MultiEdit, Bash, WebFetch, Task |
| `quantitative-analyst` | 08-finance | Technical indicators & Greeks | Read, Write, MultiEdit, Bash, Task |
| `trading-strategy-architect` | 08-finance | Backtesting & strategy validation | Read, Write, MultiEdit, Bash, Task |
| `trading-risk-manager` | 08-finance | Position sizing & portfolio optimization | Read, Write, MultiEdit, Bash, Task |
| `algorithmic-trading-engineer` | 08-finance | Order execution & broker integration | Read, Write, MultiEdit, Bash, Task |
| `equity-research-analyst` | 08-finance | Fundamental analysis & valuation | Read, Write, MultiEdit, WebSearch, Task |
| `trading-ml-specialist` | 08-finance | ML for trading with walk-forward validation | Read, Write, MultiEdit, Bash, Task |
| `trading-compliance-officer` | 08-finance | PDT rules & regulatory compliance | Read, Write, MultiEdit, WebSearch, Task |
| `portfolio-manager` | 08-finance | Multi-strategy portfolio construction & allocation | Read, Write, MultiEdit, Bash, Task |
| `finance-glossary` | 08-finance | Finance trading terminology & domain boundaries | Read, Grep |

## Invocation Cheatsheet
Use natural language triggers that mirror the descriptions above. Examples:
- **APIs** → `api-platform-engineer`
- **Cloud (AWS)** → `aws-cloud-architect`
- **Cloud (Azure)** → `azure-cloud-architect`
- **Cloud (GCP)** → `gcp-cloud-architect`
- **CI/CD or Infra-as-code** → `devops-automation-expert`
- **MLOps / ML pipelines** → `machine-learning-engineer`
- **Frontend UI / React / Next.js** → `frontend-expert`
- **Mobile (iOS/Android/cross-platform)** → `mobile-specialist`
- **TypeScript platforms / toolchains** → `typescript-architect`
- **Angular enterprise apps / NgRx / signals** → `angular-expert`
- **Django / DRF / async Python web platforms** → `django-expert`
- **Ruby on Rails / Hotwire / Sidekiq** → `rails-expert`
- **Laravel / Eloquent / PHP web APIs** → `laravel-expert`
- **Java 21+ / JVM architecture** → `java-expert`
- **Spring Boot 3.x / Spring Cloud / WebFlux** → `spring-boot-expert`
- **Kotlin / coroutines / KMP / Ktor** → `kotlin-expert`
- **C# / .NET 8 / ASP.NET Core** → `csharp-expert`
- **Python services / libraries** → `python-expert`
- **Backend architecture / microservices** → `backend-architect`
- **Database design / migrations** → `database-architect`
- **Observability / SLOs / telemetry** → `observability-engineer`
- **Incident response / on-call** → `sre-incident-responder`
- **Chaos engineering / resilience testing / game days** → `chaos-engineer`
- **Performance regressions** → `performance-optimization-specialist`
- **Security reviews** → `security-architect`
- **LLM systems / RAG / AI agents** → `llm-architect`
- **Prompt design / optimization / security** → `prompt-engineer`
- **Git workflows / branch strategies / PR automation** → `git-workflow-manager`
- **Developer experience / onboarding / productivity** → `dx-optimizer`
- **Complex architecture** → `system-design-specialist`
- **Research and sourcing** → `research-librarian`
- **Agent routing / dispatch** → `agent-organizer`
- **Quality gates / standards validation** → `workflow-validator`
- **Dependency updates / CVE remediation** → `dependency-manager`
- **Safe refactoring / technical debt** → `refactoring-specialist`
- **Shell scripting / Bash automation** → `bash-expert`
- **Go microservices / cloud-native Go** → `go-expert`
- **Rust systems / performance-critical code** → `rust-expert`
- **Event sourcing / CQRS / Kafka** → `event-driven-architect`
- **Kubernetes / container orchestration** → `kubernetes-architect`
- **Terraform / Infrastructure as Code** → `terraform-expert`
- **Requirements / user stories / backlog** → `product-owner`
- **User-facing documentation / tutorials** → `tech-writer`
- **Code review / triage** → `code-reviewer` or `error-diagnostician`
- **Domain modeling / event storming / DDD** → `domain-modeling-expert`
- **Documentation review / ADRs / README** → `technical-documentation-specialist`
- **Market data pipelines / stock data** → `market-data-engineer`
- **Technical indicators / RSI / MACD / options Greeks** → `quantitative-analyst`
- **Backtesting / strategy validation / walk-forward** → `trading-strategy-architect`
- **Position sizing / portfolio optimization / VaR** → `trading-risk-manager`
- **Order execution / broker API / live trading** → `algorithmic-trading-engineer`
- **Fundamental analysis / DCF / financial statements** → `equity-research-analyst`
- **Machine learning for trading / price prediction** → `trading-ml-specialist`
- **PDT rules / wash sales / trade compliance** → `trading-compliance-officer`
- **Multi-strategy portfolio / capital allocation / rebalancing** → `portfolio-manager`

## Installation & Validation
```bash
# Install or refresh all agents for the current user
./scripts/install-agents.sh --user

# Validate structure, frontmatter, and tool declarations
./scripts/verify-agents.sh

# Generate a quality snapshot (optional)
python3 scripts/quality-scorer.py --agents-dir agents --output quality-report.json
```
- Restart Claude Code after installation.
- Follow `agents/TESTING.md` to exercise automatic invocation and cross-agent workflows.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Design a new API | `api-platform-engineer` |
| Review code quality | `code-reviewer` |
| Debug production issues | `error-diagnostician` |
| Model domain boundaries | `domain-modeling-expert` |
| Optimize performance | `performance-optimization-specialist` |
| Design distributed systems | `system-design-specialist` |
| Write comprehensive tests | `test-engineer` |
| Build React/Next.js UI | `frontend-expert` |
| Build Angular applications | `angular-expert` |
| Develop mobile apps | `mobile-specialist` |
| Build Python services | `python-expert` |
| Build Django applications and DRF APIs | `django-expert` |
| Build Rails applications | `rails-expert` |
| Build Laravel applications | `laravel-expert` |
| Build Java services | `java-expert` |
| Build Spring Boot microservices | `spring-boot-expert` |
| Build Kotlin applications | `kotlin-expert` |
| Build C#/.NET services | `csharp-expert` |
| Design cloud architecture | `aws-cloud-architect` |
| Design Azure cloud architecture | `azure-cloud-architect` |
| Design GCP cloud architecture | `gcp-cloud-architect` |
| Set up CI/CD pipelines | `devops-automation-expert` |
| Conduct security review | `security-architect` |
| Design LLM/RAG systems | `llm-architect` |
| Optimize prompts | `prompt-engineer` |
| Test system resilience | `chaos-engineer` |
| Design Git workflows | `git-workflow-manager` |
| Improve developer experience | `dx-optimizer` |
| Manage dependencies & CVEs | `dependency-manager` |
| Refactor code safely | `refactoring-specialist` |
| Write shell scripts | `bash-expert` |
| Build Go services | `go-expert` |
| Build Rust applications | `rust-expert` |
| Design event-driven systems | `event-driven-architect` |
| Manage Kubernetes clusters | `kubernetes-architect` |
| Write Terraform IaC | `terraform-expert` |
| Define requirements & user stories | `product-owner` |
| Write user documentation | `tech-writer` |
| Develop trading strategies | `trading-strategy-architect` |
| Orchestrate multi-agent work | `orchestration-coordinator` |

## Common Agent Combinations

### Full-Stack Feature Development
1. `domain-modeling-expert` → Define bounded contexts
2. `api-platform-engineer` → Design API contracts
3. `database-architect` → Create data models
4. `backend-architect` → Implement services
5. `frontend-expert` → Build UI components
6. `test-engineer` → Add test coverage

### Production Incident Response
1. `error-diagnostician` → Initial triage
2. `sre-incident-responder` → Coordinate response
3. `observability-engineer` → Analyze metrics
4. `performance-optimization-specialist` → Fix bottlenecks

### Security Review Workflow
1. `security-architect` → Threat modeling
2. `code-reviewer` → Code-level issues
3. `api-platform-engineer` → API security
4. `devops-automation-expert` → Infrastructure security

### Trading System Development
1. `market-data-engineer` → Data pipeline setup
2. `quantitative-analyst` → Strategy research
3. `trading-strategy-architect` → Backtest framework
4. `trading-risk-manager` → Risk controls
5. `algorithmic-trading-engineer` → Execution layer
6. `trading-compliance-officer` → Regulatory validation

### Developer Experience Improvement
1. `dx-optimizer` → Assess friction points
2. `git-workflow-manager` → Streamline Git workflows
3. `devops-automation-expert` → Automate CI/CD
4. `technical-documentation-specialist` → Improve docs

## Troubleshooting

### Agents not loading
1. Check installation directory:
   ```bash
   ls ~/.claude/agents/    # User agents
   ls .claude/agents/      # Project agents
   ```
2. Ensure files have `.md` extension
3. Restart Claude Code after installation
4. Check file permissions (`chmod 644`)

### Agent not being invoked
1. Use explicit invocation: `"Use the [agent-name] to..."`
2. Check the agent's description matches your task
3. Verify frontmatter is valid YAML
4. Run `./scripts/verify-agents.sh` for structural issues

### Wrong agent selected
1. Be more specific in your request
2. Use @ mentions: `@agent-code-reviewer`
3. Check for overlapping descriptions between agents
4. Use `orchestration-coordinator` to explicitly route

### Performance issues
- Install only needed agents for projects (use project-level installation)
- Keep system prompts focused
- Avoid overlapping agent descriptions

## Contribution Notes
- Place new agents in the appropriate tier directory (`agents/00-meta`, `agents/01-foundation`, ...).
- **Do not add `tools:` field** to agent frontmatter - agents inherit all tools automatically.
- Document typical tool usage in this catalog's table for reference purposes.
- Update this catalog and `configs/agent-metadata.json` whenever agents are added, renamed, or removed.
