# Claude Code Agent Collection

Central catalog for Claude Agents Pro. Every agent complies with Claude Code frontmatter requirements and ships with pragmatic guidance.

## Active Agents
| Agent | Tier | Domain Focus | Tool Set |
|-------|------|--------------|----------|
| `agent-coordinator` | 00-meta | Multi-agent orchestration | Task, Read, Write, MultiEdit |
| `api-platform-engineer` | 01-foundation | API design & governance | Read, Write, MultiEdit, Bash, Grep, WebFetch, Task |
| `code-reviewer` | 01-foundation | Code review & quality gates | Read, Grep, Glob, Task, WebSearch |
| `domain-modeling-expert` | 01-foundation | Strategic DDD & context mapping | Read, Write, MultiEdit, Task, WebSearch |
| `error-diagnostician` | 01-foundation | Production debugging & triage | Read, Grep, Bash, Glob, WebSearch, Task |
| `performance-optimization-specialist` | 01-foundation | End-to-end performance tuning | Read, Write, MultiEdit, Bash, Grep, Task |
| `system-design-specialist` | 01-foundation | Distributed systems architecture | Read, Write, MultiEdit, WebSearch, Task |
| `test-engineer` | 01-foundation | Test strategy & automation | Read, Write, MultiEdit, Bash, Grep, Task |
| `frontend-expert` | 02-development | Modern web UI engineering | Read, Write, MultiEdit, WebFetch |
| `mobile-specialist` | 02-development | Native + cross-platform mobile | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `python-expert` | 02-development | Python services & libraries | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `typescript-architect` | 02-development | TypeScript platforms & tooling | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `aws-cloud-architect` | 03-specialists | Cloud architecture on AWS | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `backend-architect` | 03-specialists | Service architecture & APIs | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `data-pipeline-engineer` | 03-specialists | ETL and streaming pipelines | Read, Write, MultiEdit, Bash, Task |
| `database-architect` | 03-specialists | Data modelling & performance | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `devops-automation-expert` | 03-specialists | CI/CD and platform automation | Read, Write, MultiEdit, Bash, Task, Grep |
| `full-stack-architect` | 03-specialists | End-to-end web application delivery | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `observability-engineer` | 03-specialists | Metrics, logging, tracing, SLOs | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `sre-incident-responder` | 03-specialists | Incident response & reliability | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `machine-learning-engineer` | 04-experts | MLOps and production ML systems | Read, Write, MultiEdit, Bash, Task, WebSearch |
| `research-librarian` | 06-integration | Primary-source research & synthesis | Read, Write, MultiEdit, WebSearch |
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

## Invocation Cheatsheet
Use natural language triggers that mirror the descriptions above. Examples:
- **APIs** → `api-platform-engineer`
- **Cloud (AWS)** → `aws-cloud-architect`
- **CI/CD or Infra-as-code** → `devops-automation-expert`
- **MLOps / ML pipelines** → `machine-learning-engineer`
- **Frontend UI / React / Next.js** → `frontend-expert`
- **Mobile (iOS/Android/cross-platform)** → `mobile-specialist`
- **TypeScript platforms / toolchains** → `typescript-architect`
- **Python services / libraries** → `python-expert`
- **Backend architecture / microservices** → `backend-architect`
- **Database design / migrations** → `database-architect`
- **Observability / SLOs / telemetry** → `observability-engineer`
- **Incident response / on-call** → `sre-incident-responder`
- **Performance regressions** → `performance-optimization-specialist`
- **Security reviews** → `security-architect`
- **Complex architecture** → `system-design-specialist`
- **Research and sourcing** → `research-librarian`
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

## Contribution Notes
- Place new agents in the appropriate tier directory (`agents/00-meta`, `agents/01-foundation`, ...).
- Keep `tools` lists minimal to match least-privilege guidance.
- Update this catalog, `../AGENTS.md`, and `configs/agent-metadata.json` whenever agents are added, renamed, or removed.
