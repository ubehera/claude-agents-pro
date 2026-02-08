# Skills Directory

Progressive disclosure knowledge modules that activate on-demand to reduce baseline token usage while providing deep expertise when needed.

## What are Skills?

Skills are **modular knowledge packages** that extend agent capabilities without loading all details into every conversation. Instead of baking 2000+ lines of technical content into agent prompts, skills load progressively:

1. **Tier 1 (Metadata)**: Always loaded (~10 tokens) - name, description, trigger keywords
2. **Tier 2 (Instructions)**: Loaded when activated (~200-400 tokens) - core concepts, patterns, best practices
3. **Tier 3 (Resources)**: Loaded on-demand (~1000+ tokens) - complete code examples, detailed implementations

## Benefits

**Token Efficiency**:
- Agent baseline: Lean prompt without detailed implementations
- Skills activate only when needed
- Average conversation: ~70% token reduction vs monolithic agents

**Maintainability**:
- Update skills independently without touching agent prompts
- Share skills across multiple agents
- Version skills separately

**Clarity**:
- Agents focus on orchestration and high-level strategy
- Skills provide deep technical knowledge
- Clear separation of concerns

## Skills Catalog

### Agentic Domain

**Location**: `skills/agentic/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `hook-development` | hooks, event hooks, pre-tool, post-tool, stop hooks | Build and configure Claude Code hooks for policy enforcement and workflow automation | ~196 |
| `mcp-server-development` | mcp server, model context protocol, stdio server, tool server | Design and implement MCP servers with robust tool interfaces | ~205 |
| `memory-taxonomy` | memory taxonomy, memory graph, session memory, knowledge memory | Structure agent memory systems for retrieval, continuity, and context hygiene | ~212 |
| `multi-agent-patterns` | multi-agent, orchestration, agent collaboration, delegation | Coordinate specialized agents with clear handoffs and execution patterns | ~241 |
| `plugin-architecture` | plugin architecture, commands, skills, hooks, plugin layout | Design Claude Code plugin structure with composable commands/skills/agents | ~248 |
| `subagent-catalog` | subagent catalog, agent routing, skill routing, task routing | Route tasks to the right subagents using capability-driven selection | ~200 |

**Total**: ~1,302 lines of agentic expertise

### API Domain

**Location**: `skills/api/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `api-versioning` | api versioning, deprecation, breaking change, backwards compatibility, semantic versioning | URL/header versioning, deprecation strategies, and breaking change management | ~595 |
| `caching-strategies` | caching, etag, cache control, cdn, cache invalidation, redis cache, stale while revalidate | HTTP cache headers, ETags, CDN, Redis/Memcached, and cache invalidation patterns | ~734 |
| `graphql-patterns` | graphql, schema, resolver, mutation, subscription, n+1 problem, dataloader, apollo | GraphQL schema design, resolvers, N+1 solutions, and real-time subscriptions | ~721 |
| `openapi-spec-generation` | openapi, swagger, api schema, contract first, operation schema, api examples | OpenAPI 3.1 generation and normalization with reusable schemas and error contracts | ~71 |
| `rate-limiting` | rate limiting, throttling, token bucket, sliding window, fixed window, leaky bucket, quota | Token bucket, sliding window, fixed window algorithms with Redis | ~740 |
| `rest-best-practices` | rest, restful, http methods, status code, api design, pagination, jwt, oauth | REST API design patterns following industry standards | ~699 |

**Total**: ~3,560 lines of API expertise

### Architecture Domain

**Location**: `skills/architecture/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `architecture-decision-records` | adr, architecture decision record, design decision, architecture rationale, tradeoff analysis | Capture architecture decisions, alternatives, tradeoffs, and implementation consequences | ~53 |
| `distributed-tracing` | distributed tracing, jaeger, tempo, opentelemetry, span, trace, observability, zipkin | Track requests across distributed systems for latency and failure analysis | ~76 |
| `microservices-patterns` | microservices, saga pattern, circuit breaker, service mesh, distributed system, event driven, strangler fig | Service boundaries, event-driven communication, and resilience patterns | ~90 |
| `event-sourcing-cqrs` | event sourcing, cqrs, event store, command query, event stream, projection, aggregate event, domain event | Event sourcing and CQRS patterns for audit trails and temporal queries | ~209 |

**Total**: ~428 lines of architecture expertise

### Backend Domain

**Location**: `skills/backend/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `defense-in-depth-validation` | validation, defense in depth, multi-layer validation, boundary validation, fail fast | Multi-layer validation making bugs structurally impossible | ~502 |
| `error-handling-patterns` | error handling, exception, try catch, result type, retry, circuit breaker, exponential backoff, graceful degradation | Resilient error handling strategies across Python, TypeScript, Rust, and Go | ~758 |
| `nodejs-backend-patterns` | nodejs, express, fastify, middleware, dependency injection, repository pattern, service layer, postgresql | Node.js backend architecture with Express/Fastify and layered design | ~810 |
| `stripe-integration` | stripe, payment, checkout, subscription, webhook, payment intent, pci, billing | PCI-compliant Stripe payment processing with checkout, subscriptions, and webhooks | ~223 |
| `websocket-patterns` | websocket, ws, socket.io, real-time, sse, server sent events, pub sub, live update | WebSocket, SSE, and real-time communication patterns | ~234 |

**Total**: ~2,527 lines of backend expertise

### Cloud Domain

**Location**: `skills/cloud/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `multi-cloud-patterns` | aws, azure, gcp, google cloud, multi-cloud, cloud migration, cloud agnostic, s3, lambda | Multi-cloud architecture across AWS, GCP, and Azure with service mapping | ~686 |
| `terraform-state-management` | terraform, terraform state, remote backend, s3 backend, terraform workspace, state locking, terraform cloud | Advanced Terraform state management, remote backends, and collaboration patterns | ~703 |

**Total**: ~1,389 lines of cloud expertise

### Data Domain

**Location**: `skills/data/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `data-quality-frameworks` | data quality, data validation, great expectations, data testing, schema validation, quality gates | Data quality validation using Great Expectations and custom validators | ~623 |
| `database-migration` | database migration, schema migration, sequelize, typeorm, prisma migrate, zero downtime, rollback | Database migrations across ORMs with zero-downtime strategies | ~144 |
| `etl-pipeline-patterns` | etl, elt, data pipeline, airflow, dagster, prefect, batch processing, incremental load, dag | ETL/ELT pipelines with orchestration, incremental processing, and error handling | ~705 |
| `pandas-polars-patterns` | pandas, polars, dataframe, aggregation, groupby, merge, time series, window functions, vectorization | Advanced DataFrame manipulation for high-performance data processing | ~627 |
| `streaming-data-patterns` | kafka, stream processing, spark streaming, flink, real-time data, cdc, change data capture, kinesis | Real-time data streaming with Kafka, Spark Streaming, and Flink | ~631 |
| `warehouse-design-patterns` | data warehouse, snowflake, bigquery, redshift, dimensional modeling, star schema, dbt, medallion architecture | Modern data warehouse design with Snowflake, BigQuery, and Redshift | ~688 |

**Total**: ~3,418 lines of data expertise

### Database Domain

**Location**: `skills/database/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `query-optimization` | query optimization, sql optimization, index, execution plan, explain, n+1 query, database performance, slow query | SQL query optimization, indexing strategies, and performance tuning for PostgreSQL/MySQL | ~680 |

**Total**: ~680 lines of database expertise

### Debugging Domain

**Location**: `skills/debugging/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `debugging-strategies` | debug, debugger, breakpoint, stack trace, profiling, memory leak, git bisect, root cause, crash dump | Systematic debugging techniques, profiling tools, and root cause analysis | ~178 |

**Total**: ~178 lines of debugging expertise

### DevOps Domain

**Location**: `skills/devops/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `ci-cd-patterns` | ci/cd, continuous integration, continuous deployment, github actions, pipeline, blue-green, canary, rolling deployment | CI/CD pipeline patterns using GitHub Actions, Docker, and Kubernetes | ~836 |
| `docker-security-patterns` | docker, dockerfile, multi-stage build, buildkit, docker security, container security, distroless, trivy, hadolint | Docker security, optimization, and best practices with multi-stage builds | ~660 |
| `github-actions-patterns` | github actions, workflow, ci/cd pipeline, yaml workflow, reusable workflow, matrix build | GitHub Actions CI/CD patterns, reusable workflows, and deployment strategies | ~208 |
| `incident-runbook-templates` | incident runbook, on-call runbook, outage procedure, response playbook, escalation path | Incident runbook templates for triage, mitigation, escalation, and recovery | ~51 |
| `kubernetes-advanced-patterns` | kubernetes, k8s, statefulset, operator, crd, daemonset, service mesh, istio, ingress, network policy, helm | Advanced K8s patterns for complex workloads, operators, and enterprise clusters | ~835 |
| `monorepo-management` | monorepo, turborepo, nx workspace, pnpm workspace, changesets, lerna, multi-package | Monorepo management with Turborepo, Nx, and pnpm workspaces | ~110 |
| `on-call-handoff-patterns` | on-call handoff, shift handoff, incident handover, operational continuity | On-call handoff patterns that preserve incident context across rotations | ~48 |
| `opentelemetry-observability` | opentelemetry, otel, distributed tracing, jaeger, tempo, spans, metrics, observability, prometheus, grafana | OpenTelemetry observability for distributed tracing, metrics, and logging | ~646 |
| `postmortem-writing` | postmortem, blameless postmortem, incident analysis, root cause timeline | Blameless incident postmortems with timeline, RCA, and prevention actions | ~51 |

**Total**: ~3,445 lines of DevOps expertise

### Finance Domain

**Location**: `skills/finance/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `algo-trading-systems` | algo trading, algorithmic trading, trading system, trading bot, automated trading, systematic trading, trading engine | Complete algorithmic trading system design from data ingestion to live execution | ~834 |
| `backtesting-patterns` | backtest, backtesting, walk forward, overfitting, out of sample, transaction cost, vectorbt, backtrader, zipline | Walk-forward analysis, overfitting detection, and strategy validation | ~1053 |
| `factor-investing` | factor investing, fama french, alpha research, factor model, momentum factor, value factor, smart beta, risk premia | Systematic factor-based investing with multi-factor models and alpha decomposition | ~457 |
| `financial-modeling` | dcf, financial model, monte carlo, sensitivity analysis, valuation, wacc, lbo, terminal value, irr | DCF analysis, sensitivity testing, Monte Carlo simulations, and scenario planning | ~88 |
| `futures-strategies` | futures, contango, backwardation, roll, basis, margin, settlement, calendar spread, cost of carry | Futures contract mechanics, term structure analysis, and trading strategies | ~696 |
| `market-microstructure` | order book, bid ask spread, market depth, liquidity, price discovery, market maker, microprice, order flow | Order book modeling, liquidity metrics, price discovery, and market impact | ~860 |
| `ml-trading` | machine learning trading, ml trading, feature engineering, time series ml, xgboost trading, walk forward validation | ML pipelines for alpha generation with proper time-series validation | ~654 |
| `options-greeks` | greeks, delta, gamma, theta, vega, black-scholes, implied volatility, option pricing, iv | Options pricing and Greeks calculations using Black-Scholes model | ~607 |
| `options-strategies` | options strategy, spread, straddle, strangle, iron condor, iron butterfly, covered call, protective put, butterfly spread | Multi-leg options strategies with payoff analysis and Greeks profiles | ~607 |
| `order-execution` | twap, vwap, iceberg, execution algorithm, order execution, slippage, market impact, smart order routing | TWAP, VWAP, POV, and iceberg orders with market impact modeling | ~838 |
| `portfolio-optimization` | portfolio optimization, mean-variance, markowitz, efficient frontier, risk parity, black-litterman, asset allocation | Mean-variance, risk parity, and Black-Litterman portfolio construction | ~740 |
| `quantconnect` | quantconnect, lean engine, qcalgorithm, AddEquity, universe selection, alpha model, lean api | QuantConnect/Lean engine algorithm development and cloud deployment | ~879 |
| `regime-detection` | regime detection, market regime, regime switching, markov, trend detection, volatility regime, hurst exponent, adaptive strategy | Market regime identification and strategy adaptation across conditions | ~576 |
| `risk-metrics` | var, value at risk, cvar, expected shortfall, drawdown, max drawdown, sharpe ratio, sortino ratio, calmar ratio | VaR, CVaR, drawdown analysis, and risk-adjusted performance measurement | ~1052 |
| `statistical-models` | statistical analysis, time series, cointegration, garch, volatility model, correlation, stationarity, adf test | Statistical methods for time-series analysis and volatility modeling | ~572 |
| `stress-testing` | stress test, tail risk, cvar, scenario analysis, correlation breakdown, black swan, extreme events, monte carlo stress | Tail risk analysis, scenario-based stress testing, and extreme event modeling | ~581 |
| `technical-indicators` | rsi, macd, bollinger, atr, adx, moving average, sma, ema, technical indicator, momentum indicator | Vectorized technical analysis indicators for quantitative analysis | ~436 |
| `volatility-modeling` | garch, egarch, volatility surface, vol surface, implied volatility, iv surface, volatility smile, heston | GARCH, volatility surfaces, and term structure analysis | ~534 |

**Total**: ~12,064 lines of finance expertise

### ML/AI Domain

**Location**: `skills/ml/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `agent-development-patterns` | agent development, ai agents, autonomous agents, agent architecture, agent orchestration, multi-agent system, tool calling | AI agent architecture, triggering conditions, tool integration, and orchestration | ~541 |
| `embedding-strategies` | embeddings, embedding model, chunking, vectorization, rag indexing | Embedding strategies for robust semantic retrieval and RAG indexing pipelines | ~55 |
| `hybrid-search-implementation` | hybrid search, lexical plus semantic, bm25 plus vector, reranking | Hybrid retrieval patterns combining lexical and semantic relevance | ~46 |
| `langchain-architecture` | langchain, llm agents, langchain agents, chains, memory, tools integration, react agent, agent executor | LangChain framework with agents, memory, and tool integration patterns | ~390 |
| `llm-evaluation` | llm evaluation, model evaluation, bleu score, rouge score, bertscore, human evaluation, a/b testing, llm as judge | Evaluation strategies for LLM applications with automated metrics and benchmarking | ~446 |
| `mcp-builder` | mcp server, model context protocol, mcp tool, llm integration, mcp development, mcp sdk, build mcp | Creating MCP servers for LLM interaction with external services | ~185 |
| `mcp-integration` | mcp integration, mcp plugin, mcp config, stdio, sse, mcp server setup, mcp.json, claude code mcp | Integrating MCP servers into Claude Code plugins for external tool access | ~156 |
| `ml-pipeline-workflow` | ml pipeline, mlops, model training, model deployment, feature engineering, airflow, kubeflow, continuous training | End-to-end MLOps pipelines from data preparation to production deployment | ~444 |
| `prompt-engineering-patterns` | prompt engineering, few shot learning, chain of thought, zero shot, prompt optimization, system prompts, cot prompting | Advanced prompt engineering for maximizing LLM performance and reliability | ~277 |
| `rag-implementation` | rag, retrieval augmented generation, vector database, semantic search, document qa, embeddings, pinecone, chroma | RAG systems with vector databases and semantic search for grounded AI | ~416 |
| `similarity-search-patterns` | similarity search, nearest neighbor, semantic similarity, relevance ranking | Similarity search patterns for document, code, and entity retrieval | ~50 |
| `vector-index-tuning` | hnsw, ivf, pq, index tuning, recall latency tradeoff, vector db performance | Vector index tuning for recall, latency, and cost in production retrieval | ~51 |

**Total**: ~3,057 lines of ML/AI expertise

### Python Domain

**Location**: `skills/python/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `async-patterns` | async, await, asyncio, coroutine, event loop, async context manager, gather, create_task, semaphore | Python async/await concurrency patterns for high-performance I/O-bound apps | ~618 |
| `fastapi-patterns` | fastapi, fast api, async api, pydantic, sqlalchemy async, async repository, fastapi dependency injection | Production-ready FastAPI with async patterns, DI, and repository pattern | ~727 |
| `packaging-distribution` | packaging, pyproject.toml, poetry, hatch, setup.py, wheel, sdist, pypi, pip install | Modern Python packaging with pyproject.toml, Poetry, and Hatch | ~673 |
| `performance-profiling` | profiling, performance, cprofile, line_profiler, memory profiler, py-spy, optimization, benchmark, bottleneck | Python performance analysis with cProfile, line_profiler, and py-spy | ~653 |
| `testing-patterns` | pytest, testing, fixture, mock, parametrize, coverage, property based testing, hypothesis, unit test | Python testing strategies with pytest, hypothesis, and test automation | ~709 |
| `type-hints` | type hint, typing, mypy, protocol, generic, typeddict, type annotation, overload, literal, type guard | Advanced Python type hinting with mypy, Pyright, and type checkers | ~594 |

**Total**: ~3,974 lines of Python expertise

### Security Domain

**Location**: `skills/security/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `auth-patterns` | oauth2, oidc, openid connect, jwt, json web token, authentication, authorization, session, mfa, bearer token | OAuth2, OIDC, JWT, session management, and MFA patterns | ~646 |
| `better-auth` | better-auth, better auth, typescript auth, passkeys, magic link, oauth typescript, 2fa typescript | Framework-agnostic TypeScript authentication with Better Auth | ~153 |
| `cryptography-basics` | cryptography, encryption, hashing, signing, aes, rsa, hmac, sha256, bcrypt, argon2, digital signature | Hashing, symmetric/asymmetric encryption, signing, and key management | ~684 |
| `input-validation` | input validation, sanitization, sql injection, xss, cross site scripting, injection, allowlist, encoding | Input validation and sanitization to prevent injection attacks | ~608 |
| `k8s-security-policies` | kubernetes security, network policy, pod security, rbac, k8s rbac, security context, gatekeeper, pod security standards | Kubernetes NetworkPolicy, PodSecurityPolicy, RBAC, and Pod Security Standards | ~303 |
| `network-security` | network security, firewall, iptables, tls, ssl, vpn, wireguard, zero trust, network segmentation, vlan, ids | Firewalls, network segmentation, TLS/SSL, VPNs, and zero-trust architecture | ~761 |
| `owasp-top-10` | owasp, security, vulnerability, sql injection, xss, csrf, broken access control, security misconfiguration | OWASP Top 10 vulnerabilities with mitigation strategies | ~678 |
| `secrets-management` | secrets management, vault, aws secrets manager, secret rotation, environment variables, api key, hashicorp vault | HashiCorp Vault, AWS Secrets Manager, and secret rotation strategies | ~646 |
| `secure-coding-practices` | secure coding, input validation, sql injection, xss prevention, csrf, password hashing, encryption, sanitization | Secure coding principles to prevent common vulnerabilities across languages | ~993 |
| `security-testing` | security testing, pentest, penetration testing, dast, fuzzing, zap, burp suite, sqlmap, api security test | Penetration testing, DAST, fuzzing, and automated security test suites | ~736 |
| `threat-modeling` | stride, pasta, mitre attack, threat model, attack tree, dfd, risk assessment, attack vector, ttp | STRIDE, PASTA, and MITRE ATT&CK frameworks for threat identification | ~773 |
| `vulnerability-scanning` | vulnerability, sast, dast, dependency scan, semgrep, snyk, trivy, cve, npm audit, container scan, gitleaks | SAST, dependency scanning, and container security for continuous vulnerability detection | ~920 |

**Total**: ~7,901 lines of security expertise

### Testing Domain

**Location**: `skills/testing/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `code-review-patterns` | code review, pr review, pull request, review checklist, code quality, security review, performance review | Systematic code review for security, correctness, and maintainability | ~536 |
| `condition-based-waiting` | flaky test, race condition, waitFor, polling, timing, async test, condition wait, timeout | Replace arbitrary timeouts with condition polling to eliminate flaky tests | ~114 |
| `contract-testing` | contract testing, pact, consumer driven contract, provider verification, cdc, api contract | Consumer-driven contract testing with Pact for microservice compatibility | ~192 |
| `e2e-testing` | e2e, end-to-end, playwright, cypress, browser testing, user flow, visual regression, accessibility testing | End-to-end testing with Playwright and Cypress for critical user workflows | ~462 |
| `flaky-test-elimination` | flaky test, race condition, timing issue, test timeout, intermittent failure, test reliability, async testing | Eliminate flaky tests with condition-based waiting and proper synchronization | ~537 |
| `integration-testing` | integration test, api testing, database testing, service integration, contract testing, test containers, supertest | Integration testing for APIs, databases, and service boundaries | ~621 |
| `load-testing` | load testing, stress testing, performance testing, k6, artillery, locust, throughput, rps | Load/stress testing with k6, Artillery, and Locust for capacity planning | ~235 |
| `static-analysis` | static analysis, sast, semgrep, sonarqube, eslint, pylint, code scanning, linting, code quality automation | Static analysis tools for automated code quality and security scanning | ~724 |
| `tdd-workflow` | tdd, test-driven development, test first, red green refactor, failing test, test coverage, behavior verification | Strict TDD methodology with tests written before code | ~443 |
| `testing-anti-patterns` | mock, test anti-pattern, tdd, incomplete mock, test-only methods, over-mocking, test smell | Prevent testing anti-patterns like over-mocking and production pollution | ~222 |

**Total**: ~4,086 lines of testing expertise

### TypeScript Domain

**Location**: `skills/typescript/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `javascript-testing-patterns` | javascript testing, jest, vitest, test isolation, async tests | JavaScript and TypeScript testing patterns for deterministic and maintainable test suites | ~61 |
| `modern-javascript-patterns` | modern javascript, es2020, modules, async iteration, composition | Modern JavaScript implementation patterns for maintainable runtime code | ~63 |
| `nextjs-patterns` | nextjs, next.js, app router, server action, route handler, middleware | Next.js App Router, server actions, caching, and streaming patterns | ~250 |
| `react-patterns` | react, hooks, useEffect, useState, server component, suspense, context, rsc | React 18+ hooks, Server Components, Suspense, and state management patterns | ~228 |
| `state-management` | state management, zustand, redux, tanstack query, react query, jotai | Modern React state management with TanStack Query, Zustand, and more | ~201 |
| `type-system-advanced` | branded type, conditional type, template literal type, mapped type, infer, satisfies | Advanced TypeScript type patterns for compile-time safety | ~241 |
| `typescript-advanced-types` | generics, inference, utility types, strict typing, type constraints | Practical advanced TypeScript typing patterns for safer APIs and refactors | ~72 |

**Total**: ~1,116 lines of TypeScript expertise

### UX Domain

**Location**: `skills/ux/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `accessibility-wcag` | accessibility, wcag, a11y, aria, screen reader, keyboard navigation, color contrast, alt text, focus management | WCAG 2.1/2.2 compliance, ARIA patterns, and inclusive design | ~949 |
| `design-systems` | design system, component library, design tokens, storybook, atomic design, ui kit, style guide, pattern library | Building and maintaining scalable design systems for consistent UIs | ~1067 |
| `heuristic-evaluation` | heuristic evaluation, nielsen heuristics, usability heuristics, expert review, ux audit, cognitive walkthrough | Expert inspection methods for identifying usability issues | ~925 |
| `information-architecture` | information architecture, ia, site map, navigation, taxonomy, content strategy, card sorting, tree testing | Organizing and structuring content for optimal findability | ~844 |
| `persona-journey-mapping` | persona, user persona, empathy map, journey map, customer journey, service blueprint, jobs to be done, jtbd | User modeling and journey visualization across touchpoints | ~862 |
| `user-research-methods` | user research, user interview, usability testing, contextual inquiry, diary study, survey design, think aloud | Qualitative and quantitative user research techniques | ~990 |
| `ux-metrics-analytics` | ux metrics, sus, system usability scale, nps, net promoter score, task success rate, a/b testing, core web vitals | Measuring and analyzing user experience through quantitative methods | ~948 |
| `wireframing-prototyping` | wireframe, prototype, mockup, figma, sketch, low fidelity, high fidelity, interaction design, clickable prototype | Wireframes and prototypes at various fidelity levels | ~874 |

**Total**: ~7,459 lines of UX expertise

### Workflow Domain

**Location**: `skills/workflow/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `bug-fix` | bug, fix, debug, error, issue, broken, not working, regression, defect, crash, exception | Systematic bug diagnosis, fixing, and verification | ~326 |
| `changelog-automation` | changelog, release notes, version notes, changelog generation | Structured changelog generation from commits and release metadata | ~47 |
| `context-driven-development` | context-driven development, project context, delivery context, context artifacts | Context-first delivery planning with aligned requirements and execution artifacts | ~50 |
| `create-feature` | feature, implement, add, create, build, new functionality, user story, requirement, epic | End-to-end feature development from requirements to deployment | ~390 |
| `spec-driven-development` | spec driven, specification, PRD, product requirements, design doc, acceptance criteria | Systematic feature implementation from specs with requirements traceability | ~171 |
| `standup-report` | standup, daily report, progress report, status update, yesterday today blockers, scrum report | Automated standup report generation from git history and task management | ~223 |
| `track-management` | execution tracks, dependency mapping, track planning, delivery sequencing | Decompose large initiatives into tracks with ownership and dependency control | ~58 |

**Total**: ~1,265 lines of workflow expertise

## Usage Pattern

### Agent Configuration

Add skills to agent frontmatter:

```yaml
---
name: quantitative-analyst
skills:
  - technical-indicators
  - options-greeks
  - statistical-models
---
```

### Automatic Activation

Skills activate based on trigger keywords in user requests:

```
User: "Calculate RSI and MACD for this price series"
→ technical-indicators skill activates automatically
→ Loads implementations, code examples, validation patterns
```

```
User: "Calculate option Greeks for this call option"
→ options-greeks skill activates automatically
→ Loads Black-Scholes pricing, Greeks formulas, IV solver
```

```
User: "Test if these two stocks are cointegrated"
→ statistical-models skill activates automatically
→ Loads Engle-Granger test, z-score calculations, pairs trading patterns
```

### Skill Invocation (Claude Code Native Support)

Claude Code supports skills natively via the `Skill` tool. When an agent has skills configured, Claude automatically:
1. Checks trigger keywords in the user's request
2. Loads relevant skill metadata
3. Activates full skill content when patterns match
4. Returns to baseline after task completion

## Token Savings Example

### Before Skills (Monolithic Agent)

**quantitative-analyst.md**: 334 lines
- Frontmatter: 24 lines
- Philosophy & delegation: 40 lines
- Technical indicators: ~120 lines (RSI, MACD, BB, ATR code)
- Options Greeks: ~100 lines (Black-Scholes, all Greeks)
- Statistical analysis: ~100 lines (ADF, cointegration, GARCH)
- Quality standards: ~50 lines

**Token usage per invocation**: ~2000 tokens baseline

### After Skills (Modular Design)

**quantitative-analyst.md**: 240 lines (28% reduction)
- Frontmatter with skills reference: 27 lines
- Philosophy & delegation: 40 lines
- Skill activation guides: ~60 lines (pointers to skills)
- Quality standards: ~50 lines
- No detailed implementations (moved to skills)

**Token usage**:
- Baseline (no skills activated): ~600 tokens
- With technical-indicators skill: ~1000 tokens
- With all 3 skills: ~2100 tokens

**Average savings**: ~70% for typical invocations (most don't need all skills)

## Creating New Skills

### Skill Template

```markdown
---
name: skill-name
description: Load when user needs [specific capability description]
trigger_keywords: [keyword1, keyword2, keyword3]
---

# Skill Name

Brief overview of what this skill provides.

## Core Concepts

High-level concepts and when to use this skill.

## Implementation Patterns

Detailed patterns with code examples.

## Best Practices

Guidelines for using this skill effectively.

## Quality Standards

Quality metrics and validation criteria.

---

**Skill Type**: [Domain - Subdomain]
**Complexity**: [Simple/Moderate/Complex]
**Typical Usage**: [When this skill activates]
```

### Guidelines

1. **Single Responsibility**: One skill = one cohesive capability
2. **Self-Contained**: Skills should work independently
3. **Progressive Detail**: Start broad, get specific
4. **Production-Ready**: Include complete, tested code examples
5. **Clear Triggers**: Specific keywords that indicate when skill is needed

### When to Extract a Skill

Extract knowledge into a skill when:
- Agent prompt >300 lines
- Dense technical content (>100 lines of code/formulas)
- Content needed <50% of the time
- Knowledge is reusable across agents
- Updates happen frequently

**Don't extract if**:
- Content <100 lines total
- Needed in >80% of invocations
- Tightly coupled to agent logic
- Constantly referenced

## Skill Maintenance

### Updating Skills

1. **Edit skill file directly** in `skills/` directory
2. **No agent updates needed** - skills load dynamically
3. **Test activation** by invoking with trigger keywords
4. **Version skills** if making breaking changes

### Sharing Skills Across Agents

Multiple agents can reference the same skill:

```yaml
# quantitative-analyst frontmatter
skills:
  - technical-indicators
  - options-greeks

# trading-strategy-architect frontmatter
skills:
  - technical-indicators  # Shared skill
```

### Skill Dependencies

Skills can reference other skills or suggest delegation:

```markdown
## When to Use Other Skills

- For GARCH volatility modeling → Activate `statistical-models` skill
- For ML feature engineering → Activate `feature-engineering` skill
```

## Roadmap

### Phase 1 (Complete)
- ✅ Finance domain skills (technical-indicators, options-greeks, statistical-models)
- ✅ quantitative-analyst integration
- ✅ Token efficiency validation
- ✅ Workflow skills (standup-report, bug-fix, create-feature)

### Phase 2 (Complete)
- ✅ ML/AI skills (12 skills: agent-development-patterns, embedding-strategies, hybrid-search-implementation, langchain-architecture, llm-evaluation, mcp-builder, mcp-integration, ml-pipeline-workflow, prompt-engineering-patterns, rag-implementation, similarity-search-patterns, vector-index-tuning)
- ✅ Python skills (6 skills: async-patterns, fastapi-patterns, packaging-distribution, performance-profiling, testing-patterns, type-hints)
- ✅ Backend skills (5 skills: defense-in-depth-validation, error-handling-patterns, nodejs-backend-patterns, stripe-integration, websocket-patterns)
- ✅ Security skills (12 skills: auth-patterns, better-auth, cryptography-basics, input-validation, k8s-security-policies, network-security, owasp-top-10, secrets-management, secure-coding-practices, security-testing, threat-modeling, vulnerability-scanning)
- ✅ Testing skills (10 skills: code-review-patterns, condition-based-waiting, contract-testing, e2e-testing, flaky-test-elimination, integration-testing, load-testing, static-analysis, tdd-workflow, testing-anti-patterns)
- ✅ API skills (6 skills: api-versioning, caching-strategies, graphql-patterns, openapi-spec-generation, rate-limiting, rest-best-practices)
- ✅ DevOps skills (9 skills: ci-cd-patterns, docker-security-patterns, github-actions-patterns, incident-runbook-templates, kubernetes-advanced-patterns, monorepo-management, on-call-handoff-patterns, opentelemetry-observability, postmortem-writing)
- ✅ Data skills (6 skills: data-quality-frameworks, database-migration, etl-pipeline-patterns, pandas-polars-patterns, streaming-data-patterns, warehouse-design-patterns)
- ✅ UX skills (8 skills: accessibility-wcag, design-systems, heuristic-evaluation, information-architecture, persona-journey-mapping, user-research-methods, ux-metrics-analytics, wireframing-prototyping)
- ✅ Cloud skills (2 skills: multi-cloud-patterns, terraform-state-management)
- ✅ Architecture skills (4 skills: architecture-decision-records, distributed-tracing, event-sourcing-cqrs, microservices-patterns)
- ✅ Database skills (1 skill: query-optimization)
- ✅ Debugging skills (1 skill: debugging-strategies)
- ✅ Finance domain expanded (15 additional skills beyond Phase 1)
- ✅ Agentic skills (6 skills: hook-development, mcp-server-development, memory-taxonomy, multi-agent-patterns, plugin-architecture, subagent-catalog)
- ✅ TypeScript skills (7 skills: javascript-testing-patterns, modern-javascript-patterns, nextjs-patterns, react-patterns, state-management, type-system-advanced, typescript-advanced-types)
- ✅ Workflow skills expanded (7 skills: bug-fix, changelog-automation, context-driven-development, create-feature, spec-driven-development, standup-report, track-management)

### Phase 3 (Planned)
- Cross-domain skill sharing recommendations
- Skill versioning and compatibility policy
- Skill marketplace/registry
- Automated docs generation for catalog drift prevention

## Best Practices for Skill Usage

### For Users

1. **Request what you need naturally** - skills activate automatically
2. **Don't mention skills explicitly** - they're an implementation detail
3. **Expect complete answers** - skills provide production-ready code

### For Agent Developers

1. **Reference skills in agent frontmatter** clearly
2. **Remove detailed implementations** from agent prompts
3. **Add skill activation guides** in agent body
4. **Test skill activation** with various trigger patterns
5. **Monitor token usage** to validate savings

## Quality Standards

All skills must meet:
- **Completeness**: Self-contained with all necessary code/patterns
- **Accuracy**: Production-ready, tested implementations
- **Clarity**: Clear concepts, well-documented code
- **Performance**: Optimized patterns (vectorization, type hints)
- **Safety**: Error handling, input validation, no hardcoded secrets

---

**Skills System**: Progressive Disclosure Architecture
**Status**: Production (Phase 2 Complete, Phase 3 in progress)
**Total Skills**: 120 across 17 active domains (~57,849 lines of expertise)
**Token Savings**: ~70% average reduction vs monolithic agents
**Domains**: Agentic (6), API (6), Architecture (4), Backend (5), Cloud (2), Data (6), Database (1), Debugging (1), DevOps (9), Finance (18), ML/AI (12), Python (6), Security (12), Testing (10), TypeScript (7), UX (8), Workflow (7)
