# Tier 01: Foundation Agents

Foundation agents are your core engineering toolkit for building robust, production-ready systems. These agents cover essential development practices from API design to testing, code review to debugging.

## When to Use Foundation Agents

Use these agents when you need to:
- **Design APIs** with industry-standard specifications
- **Model complex domains** using DDD principles
- **Review code** for quality, security, and maintainability
- **Debug issues** systematically with expert guidance
- **Write tests** that provide meaningful coverage
- **Optimize performance** across the stack
- **Design systems** that scale and remain maintainable

## Available Agents

### [api-platform-engineer](api-platform-engineer.md)
Expert in REST API design, GraphQL schemas, OpenAPI/Swagger specs, API gateways, rate limiting, OAuth 2.0/JWT auth, and API governance.

**Use when:** Designing new APIs, setting up API gateways, creating API documentation, implementing authentication flows.

### [code-reviewer](code-reviewer.md)
Quality-focused reviewer analyzing code for correctness, maintainability, performance, and security. Provides actionable feedback with specific recommendations.

**Use when:** Reviewing PRs, conducting code audits, establishing coding standards, improving existing code quality.

### [domain-modeling-expert](domain-modeling-expert.md)
Strategic DDD practitioner for bounded context mapping, aggregate design, event storming, and ubiquitous language development.

**Use when:** Starting new projects, decomposing monoliths, defining service boundaries, clarifying complex business domains.

### [error-diagnostician](error-diagnostician.md)
Expert debugging and error analysis specialist for diagnosing runtime errors, compilation issues, test failures, and system problems.

**Use when:** Troubleshooting errors, analyzing stack traces, debugging complex issues, investigating production incidents.

### [performance-optimization-specialist](performance-optimization-specialist.md)
Performance expert for optimization, Core Web Vitals, database performance, caching, API latency, memory optimization, and bottleneck identification.

**Use when:** Slow applications, performance issues, optimization strategies, scalability challenges.

### [refactoring-specialist](refactoring-specialist.md)
Safe code refactoring expert for improving code quality without changing behavior using test-driven refactoring techniques.

**Use when:** Technical debt reduction, code modernization, architecture improvement, maintaining backwards compatibility.

### [dependency-manager](dependency-manager.md)
Dependency analysis, update, and security management expert for maintaining healthy dependency graphs.

**Use when:** Dependency updates, CVE remediation, compatibility analysis, dependency graph optimization.

### [system-design-specialist](system-design-specialist.md)
System design expert for distributed systems, microservices, scalability, load balancing, caching, database design, and large-scale architecture.

**Use when:** System architecture, distributed system design, scalability planning, handling millions of users.

### [test-engineer](test-engineer.md)
Expert test automation specialist for creating comprehensive test suites, implementing testing strategies, and ensuring code quality.

**Use when:** Writing tests, setting up test frameworks, improving test coverage.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Design REST/GraphQL APIs | **api-platform-engineer** |
| Review code changes | **code-reviewer** |
| Model domain boundaries | **domain-modeling-expert** |
| Debug production issues | **error-diagnostician** |
| Tune application performance | **performance-optimization-specialist** |
| Refactor legacy code safely | **refactoring-specialist** |
| Manage dependencies | **dependency-manager** |
| Design distributed systems | **system-design-specialist** |
| Create test strategies | **test-engineer** |

## Common Combinations

**New Feature Development:**
1. `domain-modeling-expert` → Clarify domain concepts
2. `api-platform-engineer` → Design API contracts
3. `test-engineer` → Write tests first (TDD)
4. `code-reviewer` → Final quality check

**Performance Investigation:**
1. `error-diagnostician` → Identify symptoms
2. `performance-optimization-specialist` → Profile and fix
3. `code-reviewer` → Validate optimizations

**Technical Debt Reduction:**
1. `dependency-manager` → Update outdated deps
2. `refactoring-specialist` → Improve code quality
3. `test-engineer` → Ensure coverage
4. `code-reviewer` → Validate changes

## Best Practices

- **API-first**: Use `api-platform-engineer` before backend implementation
- **Domain clarity**: Start with `domain-modeling-expert` for complex features
- **Quality gates**: End workflows with `code-reviewer` and `test-engineer`
- **Debug systematically**: Let `error-diagnostician` guide investigation
- **Iterate frequently**: Work with agents in short cycles for better results
