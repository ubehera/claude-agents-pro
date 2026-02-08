# Agent Examples Analysis Report

**Generated**: 2025-12-11
**Purpose**: Document agent trigger examples and activation patterns for future enhancement
**Total Agents Analyzed**: 45 agents across 8 tiers

---

## Executive Summary

This analysis examined 45 production-ready agents in the claude-agents-pro repository to understand current frontmatter structure and prepare for adding concrete trigger examples. The agents span from meta-orchestration (Tier 0) to specialized finance trading systems (Tier 8).

### Current Frontmatter Structure

All agents follow this YAML frontmatter pattern:

```yaml
name: agent-name                    # Must match filename
description: detailed description   # Keywords and use cases
category: tier-category            # orchestrator, foundation, specialist, integration
complexity: simple | moderate | complex
model: claude-opus-4-6              # Model assignment
model_rationale: explanation
capabilities:
  - list of capabilities
auto_activate:
  keywords: [trigger keywords]
  conditions: [activation conditions]
```

### Proposed Enhancement

Add `examples` section to frontmatter with trigger-commentary pairs:

```yaml
examples:
  - trigger: "User request or task description"
    commentary: "When/why agent activates, expected output"
```

---

## Agent Inventory by Tier

### Tier 0: Meta (3 agents)
- `agent-organizer.md`
- `orchestration-coordinator.md`
- `workflow-validator.md`

### Tier 1: Foundation (9 agents)
- `api-platform-engineer.md`
- `code-reviewer.md`
- `dependency-manager.md`
- `domain-modeling-expert.md`
- `error-diagnostician.md`
- `performance-optimization-specialist.md`
- `refactoring-specialist.md`
- `system-design-specialist.md`
- `test-engineer.md`

### Tier 2: Development (7 agents)
- `bash-expert.md`
- `frontend-expert.md`
- `go-expert.md`
- `mobile-specialist.md`
- `python-expert.md`
- `rust-expert.md`
- `typescript-architect.md`

### Tier 3: Specialists (11 agents)
- `aws-cloud-architect.md`
- `backend-architect.md`
- `data-pipeline-engineer.md`
- `database-architect.md`
- `devops-automation-expert.md`
- `event-driven-architect.md`
- `full-stack-architect.md`
- `kubernetes-architect.md`
- `observability-engineer.md`
- `sre-incident-responder.md`
- `terraform-expert.md`

### Tier 4: Experts (1 agent)
- `machine-learning-engineer.md`

### Tier 6: Integration (4 agents)
- `product-owner.md`
- `research-librarian.md`
- `tech-writer.md`
- `technical-documentation-specialist.md`

### Tier 7: Quality (1 agent)
- `security-architect.md`

### Tier 8: Finance (9 agents)
- `algorithmic-trading-engineer.md`
- `equity-research-analyst.md`
- `market-data-engineer.md`
- `portfolio-manager.md`
- `quantitative-analyst.md`
- `trading-compliance-officer.md`
- `trading-ml-specialist.md`
- `trading-risk-manager.md`
- `trading-strategy-architect.md`

---

## Sample Examples by Tier

### Tier 0: Meta-Orchestration

#### orchestration-coordinator

```yaml
examples:
  - trigger: "Build a multi-service microservices platform with API gateway, authentication, and monitoring"
    commentary: "Invoked for complex multi-domain projects requiring coordination between api-platform-engineer, backend-architect, security-architect, and observability-engineer. Decomposes requirements, sequences work, manages dependencies."

  - trigger: "Coordinate implementation of new payment system across frontend, backend, and compliance"
    commentary: "Routes payment domain modeling to domain-modeling-expert, security review to security-architect, API design to api-platform-engineer, compliance checks to trading-compliance-officer. Ensures consistent communication protocols."

  - trigger: "Optimize development workflow across 5 teams working on shared codebase"
    commentary: "Analyzes workflow bottlenecks, delegates CI/CD improvements to devops-automation-expert, code quality to code-reviewer, identifies parallel work opportunities. Orchestrates quality gates."
```

---

### Tier 1: Foundation

#### api-platform-engineer

```yaml
examples:
  - trigger: "Design REST API for user management with OAuth2 authentication"
    commentary: "Invoked for greenfield API design. Delivers OpenAPI 3.0 spec with endpoints, schemas, error codes, rate limiting, and OAuth2 flows. Includes Kong gateway configuration."

  - trigger: "Add GraphQL federation for microservices architecture"
    commentary: "Creates federated GraphQL gateway schema, defines entity resolution patterns, implements Apollo Gateway configuration. Coordinates with backend-architect on service boundaries."

  - trigger: "Migrate legacy SOAP API to REST with versioning strategy"
    commentary: "Analyzes existing SOAP contracts, designs backward-compatible REST endpoints, implements URL path versioning, creates deprecation timeline. Provides client SDK migration guide."
```

#### system-design-specialist

```yaml
examples:
  - trigger: "Design scalable architecture for real-time chat application with 1M concurrent users"
    commentary: "Creates C4 diagrams (context, container, component), selects WebSocket + Redis Pub/Sub, defines horizontal scaling strategy, documents CAP theorem trade-offs. Includes load estimation calculations."

  - trigger: "Review architecture for e-commerce checkout flow to identify bottlenecks"
    commentary: "Analyzes sequence diagrams, identifies single points of failure, recommends circuit breakers, suggests cache-aside pattern for product catalog, proposes database read replica strategy."

  - trigger: "Design event-driven architecture for order processing system"
    commentary: "Defines event schemas, selects Kafka as message broker, designs topic partitioning strategy, documents saga pattern for distributed transactions, creates failure recovery playbooks."
```

#### test-engineer

```yaml
examples:
  - trigger: "Create comprehensive test suite for payment processing API"
    commentary: "Generates test pyramid strategy (70% unit, 20% integration, 10% E2E), writes pytest fixtures for API contract tests, configures coverage thresholds >85%, creates CI test pipeline."

  - trigger: "Add mutation testing to existing Jest test suite"
    commentary: "Configures Stryker for mutation testing, identifies gaps in assertion quality, improves test cases to catch mutants, establishes mutation score baseline >80%."

  - trigger: "Design load testing strategy for Black Friday traffic (10x normal load)"
    commentary: "Creates k6 load test scenarios, ramps from baseline to 10x over 30 minutes, monitors P95/P99 latency and error rates, identifies breaking points, provides capacity recommendations."
```

---

### Tier 2: Development

#### typescript-architect

```yaml
examples:
  - trigger: "Setup TypeScript monorepo with Next.js, tRPC, and Prisma"
    commentary: "Configures pnpm workspaces with project references, sets up shared tsconfig bases, implements type-safe tRPC endpoints with Zod schemas, configures Turborepo caching for <10s builds."

  - trigger: "Add end-to-end type safety from database to frontend using Prisma and tRPC"
    commentary: "Generates Prisma types from schema, creates tRPC routers with inferred types, shares types across client/server via pnpm workspace, validates runtime data with Zod."

  - trigger: "Optimize TypeScript build performance for large monorepo (100+ packages)"
    commentary: "Implements composite builds with project references, configures incremental compilation, adds esbuild for bundling, reduces tsc --noEmit from 5min to 30s. Documents build architecture."
```

#### python-expert

```yaml
examples:
  - trigger: "Refactor Django application to use async views with PostgreSQL connection pooling"
    commentary: "Converts views to async def, implements asyncpg for connection pooling, adds middleware for request lifecycle, configures PgBouncer, benchmarks 3x throughput improvement."

  - trigger: "Add type hints and mypy strict checking to existing Python codebase"
    commentary: "Adds type stubs for third-party libraries, converts function signatures to typed versions, configures mypy.ini with strict mode, resolves 200+ type errors incrementally."

  - trigger: "Optimize data processing pipeline using Polars instead of Pandas"
    commentary: "Refactors ETL operations from Pandas to Polars, leverages lazy evaluation, implements parallel processing, reduces memory usage by 60% and runtime by 5x for 1GB+ datasets."
```

---

### Tier 3: Specialists

#### backend-architect

```yaml
examples:
  - trigger: "Design microservices decomposition for monolithic e-commerce application"
    commentary: "Applies domain-driven design to identify bounded contexts (Catalog, Orders, Payments, Inventory), defines anti-corruption layers, chooses Kafka for event streaming, creates migration roadmap with strangler fig pattern."

  - trigger: "Implement CQRS and event sourcing for order management system"
    commentary: "Separates command (writes) and query (reads) models, designs event store schema, implements projections for read models, documents eventual consistency guarantees, provides replay mechanism for events."

  - trigger: "Add circuit breaker and retry logic to payment gateway integration"
    commentary: "Implements Polly policies for .NET or resilience4j for Java, configures exponential backoff with jitter, adds bulkhead pattern to isolate failures, monitors circuit breaker state transitions."
```

#### aws-cloud-architect

```yaml
examples:
  - trigger: "Design multi-region disaster recovery architecture for SaaS application on AWS"
    commentary: "Creates active-passive setup with RDS cross-region replication, configures Route53 health checks for automatic failover, implements S3 cross-region replication, documents RPO <15min and RTO <1hr targets."

  - trigger: "Optimize AWS costs for data analytics workload (reduce spend by 40%)"
    commentary: "Analyzes S3 storage classes, migrates infrequent data to Glacier, right-sizes EC2 instances using Compute Optimizer, purchases Reserved Instances for steady-state workloads, implements Lambda for sporadic jobs."

  - trigger: "Implement least-privilege IAM policies for CI/CD pipeline"
    commentary: "Creates service-specific IAM roles, removes wildcard permissions, implements IAM conditions (IP restrictions, MFA), uses IAM Access Analyzer to validate policies, documents security baseline."
```

#### devops-automation-expert

```yaml
examples:
  - trigger: "Build GitHub Actions CI/CD pipeline with automated testing and deployment to EKS"
    commentary: "Creates multi-stage workflow (test → build → deploy), caches dependencies, runs tests in parallel, builds Docker images, pushes to ECR, deploys to EKS using Helm, implements canary deployments."

  - trigger: "Add automated database migrations to deployment pipeline with rollback capability"
    commentary: "Integrates Flyway or Liquibase into CI pipeline, implements pre-deployment validation, creates rollback scripts, adds smoke tests post-migration, documents migration failure recovery procedures."

  - trigger: "Setup monitoring and alerting for Kubernetes cluster with Prometheus and Grafana"
    commentary: "Deploys Prometheus Operator, configures ServiceMonitors for auto-discovery, creates Grafana dashboards for pod metrics, sets up PagerDuty alerts for critical thresholds (>80% CPU, pod restarts)."
```

---

### Tier 4: Experts

#### machine-learning-engineer

```yaml
examples:
  - trigger: "Build MLOps pipeline for fraud detection model with A/B testing and drift monitoring"
    commentary: "Creates training pipeline with MLflow tracking, deploys model to Kubernetes with Seldon Core, implements A/B testing framework (10% treatment), monitors data drift with Evidently, sets up retraining triggers."

  - trigger: "Deploy recommendation system using PyTorch with feature store and real-time inference"
    commentary: "Builds feature engineering pipeline with Feast, trains collaborative filtering model with PyTorch, deploys to TorchServe with GPU acceleration, implements Redis caching for low-latency inference <50ms."

  - trigger: "Add model explainability using SHAP for credit scoring model"
    commentary: "Integrates SHAP explainer into prediction pipeline, generates feature importance plots, creates model card with fairness metrics, implements individual prediction explanations for compliance reporting."
```

---

### Tier 6: Integration

#### technical-documentation-specialist

```yaml
examples:
  - trigger: "Review ADR for caching strategy decision and improve clarity"
    commentary: "Analyzes ADR structure, identifies missing alternatives section, clarifies trade-offs with concrete metrics, adds implementation notes with phased rollout plan, fixes heading hierarchy and formatting."

  - trigger: "Improve README for open-source project to reduce onboarding friction"
    commentary: "Restructures README with quick start section upfront, adds prerequisites with version numbers, includes copy-pasteable installation commands, creates troubleshooting section for common errors."

  - trigger: "Standardize API documentation across microservices using OpenAPI"
    commentary: "Creates OpenAPI spec template with consistent error codes, adds request/response examples for all endpoints, documents authentication flows, establishes versioning conventions, generates developer portal."
```

#### research-librarian

```yaml
examples:
  - trigger: "Find authoritative sources for OAuth 2.1 specification and best practices"
    commentary: "Searches for RFC 6749, Draft OAuth 2.1 spec, OWASP guidance, returns 5 canonical URLs with short annotations, prioritizes official specs over blog posts."

  - trigger: "Locate latest Kubernetes security benchmarks and compliance guides"
    commentary: "Finds CIS Kubernetes Benchmark, NSA/CISA hardening guide, NIST container security guidance, provides version-specific recommendations (1.28+), includes checklist tools."

  - trigger: "Research GDPR data retention requirements for financial services"
    commentary: "Locates GDPR Article 5 text, EDPB guidelines, financial sector-specific regulations, summarizes retention periods (7 years for financial records), provides compliance framework."
```

---

### Tier 7: Quality

#### security-architect

```yaml
examples:
  - trigger: "Perform threat modeling for payment processing system using STRIDE"
    commentary: "Analyzes system boundaries, identifies threats (spoofing payment gateway, tampering amounts, repudiation of transactions), maps to STRIDE categories, recommends mitigations (mTLS, HMAC signatures, audit logs), creates threat model diagram."

  - trigger: "Implement OAuth 2.0 with PKCE for single-page application"
    commentary: "Designs authorization code flow with PKCE, configures token lifetimes (15min access, 7day refresh), implements token refresh logic, adds CSRF protection, documents security considerations."

  - trigger: "Conduct security code review for authentication module"
    commentary: "Identifies OWASP Top 10 vulnerabilities (SQL injection, XSS, broken authentication), verifies password hashing (Argon2), checks session management, validates input sanitization, provides remediation recommendations."
```

---

### Tier 8: Finance

#### algorithmic-trading-engineer

```yaml
examples:
  - trigger: "Integrate Alpaca broker API for automated order execution with retry logic"
    commentary: "Implements BrokerInterface abstraction, adds exponential backoff retry, creates order placement with client-side IDs for idempotency, sets up position reconciliation, logs all trades to database."

  - trigger: "Implement TWAP execution algorithm to split 10,000 share order over 30 minutes"
    commentary: "Divides order into 10 slices, submits limit orders every 3 minutes, dynamically adjusts limit price based on current market, monitors fills, logs execution quality metrics (slippage, completion rate)."

  - trigger: "Build order management system with position tracking and real-time monitoring"
    commentary: "Creates order lifecycle state machine, implements position reconciliation comparing system vs broker positions, sets up WebSocket for real-time order updates, adds Grafana dashboard for order metrics."
```

#### trading-strategy-architect

```yaml
examples:
  - trigger: "Design mean reversion strategy for S&P 500 stocks with z-score signals"
    commentary: "Defines entry signal (z-score < -2), exit conditions (z-score > 0 or stop loss -5%), implements rolling window calculations, adds position sizing rules, specifies risk parameters (max 10% per position)."

  - trigger: "Backtest momentum strategy using Zipline with transaction costs"
    commentary: "Implements momentum ranking, rebalances monthly, includes commission (0.01/share) and slippage (1bps), calculates Sharpe ratio, max drawdown, turnover, compares against SPY benchmark."

  - trigger: "Add risk management rules to existing pairs trading strategy"
    commentary: "Implements portfolio heat limits (max 25% capital at risk), adds correlation matrix for pairs selection, sets stop-loss at 2x historical spread volatility, documents risk metrics tracking."
```

#### quantitative-analyst

```yaml
examples:
  - trigger: "Calculate VaR (Value at Risk) for equity portfolio using historical simulation"
    commentary: "Retrieves 2 years of daily returns, sorts returns to find 5th percentile, scales to 1-day 95% VaR, reports VaR in dollars and percentage terms, validates with backtesting."

  - trigger: "Build Black-Scholes options pricer with Greeks calculation"
    commentary: "Implements Black-Scholes formula for call/put pricing, calculates Delta, Gamma, Theta, Vega, Rho using numerical differentiation, validates against market prices, creates pricing surface visualization."

  - trigger: "Perform cointegration analysis for pairs trading candidates"
    commentary: "Runs Augmented Dickey-Fuller test on price spread, calculates half-life of mean reversion, identifies top 10 cointegrated pairs, estimates hedge ratios using OLS regression, provides trading thresholds."
```

---

## Recommended Example Patterns by Agent Type

### Pattern A: Technical Implementation
**Structure**: `[Action verb] + [technology/pattern] + [context/constraints]`
**Commentary**: `Invoked when [condition], delivers [artifacts], includes [specifics]`

**Example**:
```yaml
trigger: "Implement JWT authentication with refresh tokens for Express API"
commentary: "Invoked for stateless authentication needs. Generates middleware for token validation, implements refresh token rotation, adds rate limiting, provides Postman collection for testing."
```

**Best for**: Development agents (typescript-architect, python-expert, backend-architect)

---

### Pattern B: Architecture & Design
**Structure**: `[Design/Architect/Plan] + [system/component] + [quality attributes]`
**Commentary**: `Creates [diagrams/specs], selects [technologies], documents [trade-offs]`

**Example**:
```yaml
trigger: "Design event-driven architecture for order processing with guaranteed delivery"
commentary: "Creates event schema registry, selects Kafka with idempotent producers, designs consumer retry logic with DLQ, documents consistency guarantees, provides deployment checklist."
```

**Best for**: Architecture agents (system-design-specialist, backend-architect, aws-cloud-architect)

---

### Pattern C: Analysis & Review
**Structure**: `[Analyze/Review/Audit] + [artifact] + [criteria/focus area]`
**Commentary**: `Evaluates [aspects], identifies [issues], recommends [improvements]`

**Example**:
```yaml
trigger: "Review OpenAPI specification for consistency and completeness"
commentary: "Checks all endpoints have descriptions and examples, validates error codes, verifies authentication flows, identifies missing pagination params, recommends versioning improvements."
```

**Best for**: Quality agents (code-reviewer, technical-documentation-specialist, security-architect)

---

### Pattern D: Orchestration & Coordination
**Structure**: `[Coordinate/Orchestrate/Manage] + [workflow] + [teams/domains]`
**Commentary**: `Decomposes [requirements], delegates to [agents], manages [dependencies]`

**Example**:
```yaml
trigger: "Coordinate implementation of payment system across frontend, backend, and compliance teams"
commentary: "Breaks down into API design (api-platform-engineer), security review (security-architect), fraud detection (trading-risk-manager). Sequences work: API contract → backend → frontend → compliance audit."
```

**Best for**: Meta agents (orchestration-coordinator, agent-organizer)

---

### Pattern E: Troubleshooting & Resolution
**Structure**: `[Debug/Fix/Resolve] + [problem] + [symptoms/context]`
**Commentary**: `Diagnoses [root cause], implements [solution], validates [fix]`

**Example**:
```yaml
trigger: "Debug memory leak in Node.js application causing OOM crashes"
commentary: "Uses heap snapshots to identify retained objects, traces object references, finds EventEmitter listener leak, implements cleanup logic, verifies with stress testing."
```

**Best for**: Diagnostic agents (error-diagnostician, performance-optimization-specialist, sre-incident-responder)

---

## Implementation Roadmap

### Phase 1: Foundation Agents (Week 1)
Priority agents for example addition:
1. `api-platform-engineer` - High usage, clear trigger patterns
2. `system-design-specialist` - Architecture decisions
3. `backend-architect` - Service design
4. `typescript-architect` - Type safety patterns
5. `test-engineer` - Testing strategies

### Phase 2: Specialist Agents (Week 2)
6. `aws-cloud-architect` - Cloud infrastructure
7. `devops-automation-expert` - CI/CD pipelines
8. `security-architect` - Threat modeling
9. `database-architect` - Schema design
10. `performance-optimization-specialist` - Performance tuning

### Phase 3: Finance & Integration (Week 3)
11. `algorithmic-trading-engineer` - Order execution
12. `trading-strategy-architect` - Strategy design
13. `technical-documentation-specialist` - Docs review
14. `machine-learning-engineer` - MLOps

### Phase 4: Meta & Remaining (Week 4)
15. `orchestration-coordinator` - Multi-agent workflows
16. All remaining agents (language-specific, specialized domains)

---

## Quality Guidelines for Examples

### Characteristics of Good Examples

✅ **Specific and Concrete**
- Bad: "Design API"
- Good: "Design REST API for user management with OAuth2 authentication"

✅ **Includes Context/Constraints**
- Bad: "Build CI/CD pipeline"
- Good: "Build GitHub Actions CI/CD pipeline with automated testing and deployment to EKS"

✅ **Actionable and Clear**
- Bad: "Help with database"
- Good: "Optimize PostgreSQL query with 30s execution time by adding indexes"

✅ **Realistic Use Cases**
- Based on actual project patterns, not contrived scenarios

✅ **Commentary Includes Deliverables**
- What the agent produces (diagrams, code, specs, configurations)

### Anti-Patterns to Avoid

❌ **Too Generic**
```yaml
trigger: "Build something"
commentary: "Builds it"
```

❌ **Missing Technology Context**
```yaml
trigger: "Setup authentication"  # Which protocol? Which stack?
```

❌ **Vague Commentary**
```yaml
commentary: "Helps with the task"  # What specifically?
```

❌ **No Clear Activation Condition**
```yaml
trigger: "Work on code"  # Too broad, many agents overlap
```

---

## Next Steps

1. **Review & Validate**: Review this analysis with domain experts
2. **Prioritize Agents**: Confirm Phase 1 priority agents
3. **Draft Examples**: Create 2-3 examples per agent following patterns
4. **Validate Format**: Ensure YAML frontmatter compatibility
5. **Bulk Update**: Update all agent files with examples
6. **Documentation**: Update AGENT_CHECKLIST.md with examples requirement
7. **Testing**: Validate agent activation with new examples

---

## Appendix: Frontmatter Template with Examples

```yaml
---
name: agent-name
description: Expert in [domain] for [use cases]. Use for [scenarios].
category: foundation
complexity: complex
model: claude-opus-4-6
model_rationale: Maximum capability for optimal results
capabilities:
  - Capability 1
  - Capability 2
  - Capability 3
auto_activate:
  keywords: [keyword1, keyword2, keyword3]
  conditions: [condition1, condition2]
examples:
  - trigger: "Specific user request with context and constraints"
    commentary: "When agent activates, what it delivers, key technologies used"

  - trigger: "Another realistic scenario with technical details"
    commentary: "Expected artifacts, delegation patterns, quality criteria"

  - trigger: "Third use case showing different aspect of agent"
    commentary: "Outcome description, integration points, success metrics"
---
```

---

**End of Report**
