# Testing & Code Quality Skills

Comprehensive testing methodologies, code review patterns, and quality assurance skills for building reliable software.

## Skills Overview

### Testing Methodologies

#### [tdd-workflow.md](tdd-workflow.md)
**Test-Driven Development Workflow**

Strict TDD methodology ensuring tests are written before implementation.

- **When to Use**: All feature implementation, bug fixes, refactoring
- **Core Principle**: No production code without a failing test first
- **Key Patterns**: Red-Green-Refactor cycle, failing test verification
- **Trigger Keywords**: `tdd`, `test-driven development`, `red green refactor`, `failing test`

#### [e2e-testing.md](e2e-testing.md)
**End-to-End Testing Patterns**

Production-grade E2E testing with Playwright and Cypress.

- **When to Use**: Critical user workflows, cross-browser testing, UI automation
- **Core Patterns**: Page Object Model, fixtures, network mocking, visual regression
- **Frameworks**: Playwright, Cypress
- **Trigger Keywords**: `e2e`, `playwright`, `cypress`, `browser testing`, `user flow`

#### [integration-testing.md](integration-testing.md)
**Integration Testing Patterns**

API, database, and service integration testing with real dependencies.

- **When to Use**: API endpoints, database operations, service-to-service communication
- **Core Patterns**: Test containers, transaction testing, contract testing
- **Frameworks**: Supertest, pytest, TestClient, Pact
- **Trigger Keywords**: `integration test`, `api testing`, `database testing`, `contract testing`

#### [flaky-test-elimination.md](flaky-test-elimination.md)
**Flaky Test Elimination**

Systematic patterns for eliminating flaky tests through condition-based waiting.

- **When to Use**: Debugging intermittent failures, race conditions, timing issues
- **Core Principle**: Wait for actual conditions, not arbitrary delays
- **Key Patterns**: Condition polling, event-based waiting, proper synchronization
- **Trigger Keywords**: `flaky test`, `race condition`, `timing issue`, `intermittent failure`

### Code Quality

#### [code-review-patterns.md](code-review-patterns.md)
**Code Review Patterns**

Systematic code review methodology for security, correctness, and maintainability.

- **When to Use**: PR reviews, pre-merge validation, architecture compliance
- **Quality Dimensions**: Security (25%), Correctness (25%), Spec Alignment (20%), Performance (15%), Maintainability (15%)
- **Review Strategies**: Sequential (small PRs), Parallel (large PRs)
- **Trigger Keywords**: `code review`, `pr review`, `security review`, `spec compliance`

#### [static-analysis.md](static-analysis.md)
**Static Analysis Patterns**

SAST tool configuration for automated security and quality scanning.

- **When to Use**: CI/CD setup, DevSecOps implementation, compliance scanning
- **Tools Covered**: Semgrep, ESLint, Pylint, SonarQube, CodeQL, Bandit
- **Key Patterns**: Custom rule creation, CI/CD integration, pre-commit hooks
- **Trigger Keywords**: `static analysis`, `sast`, `semgrep`, `eslint`, `security scanning`

## Quick Reference

### When to Use Each Skill

| Scenario | Skill to Use |
|----------|--------------|
| Implementing new feature | `tdd-workflow` |
| Fixing production bug | `tdd-workflow` + `flaky-test-elimination` |
| Testing user workflow | `e2e-testing` |
| Testing API endpoint | `integration-testing` |
| Test fails intermittently | `flaky-test-elimination` |
| Reviewing pull request | `code-review-patterns` |
| Setting up CI/CD | `static-analysis` |
| Debugging race condition | `flaky-test-elimination` |
| Cross-browser testing | `e2e-testing` |
| Security vulnerability scan | `static-analysis` |

### Testing Pyramid Application

```
        /\
       /E2E\         ← 10%: e2e-testing.md
      /─────\
     /Integr\        ← 20%: integration-testing.md
    /────────\
   /Unit Tests\      ← 70%: tdd-workflow.md
  /────────────\
```

**All levels**: Use `tdd-workflow` methodology (test-first)
**Flaky tests**: Apply `flaky-test-elimination` patterns at any level
**Quality gates**: Enforce with `static-analysis` in CI/CD

## Integration Patterns

### Development Workflow

```
1. tdd-workflow: Write failing test first
   ↓
2. tdd-workflow: Implement minimal code
   ↓
3. integration-testing: Test API/DB integration
   ↓
4. e2e-testing: Test user workflow
   ↓
5. flaky-test-elimination: Fix any timing issues
   ↓
6. code-review-patterns: Review for quality
   ↓
7. static-analysis: Automated security/quality scan
```

### CI/CD Pipeline

```yaml
# Example: Complete testing pipeline
stages:
  - lint          # static-analysis
  - unit-test     # tdd-workflow
  - integration   # integration-testing
  - e2e           # e2e-testing
  - review        # code-review-patterns (automated)
  - security      # static-analysis (SAST)
```

## Best Practices

### Test Quality Standards

From `tdd-workflow`:
- ✅ Write test before implementation
- ✅ Watch test fail before implementing
- ✅ Write minimal code to pass
- ✅ Refactor only when green

From `flaky-test-elimination`:
- ✅ Wait for conditions, not arbitrary delays
- ✅ Use proper synchronization
- ✅ Poll with reasonable intervals (10-50ms)
- ✅ Include clear timeout error messages

### Code Review Checklist

From `code-review-patterns`:
- 🔴 **Critical**: Security vulnerabilities, breaking changes
- 🟡 **Important**: Logic bugs, missing error handling
- 🟢 **Nice-to-have**: Code style, documentation
- ✅ **Good practices**: What was done well

### Static Analysis Setup

From `static-analysis`:
1. Start with security-focused rules
2. Gradually add code quality rules
3. Tune false positive rate to <10%
4. Integrate into CI/CD pipeline
5. Create baseline for existing issues
6. Only block on critical/high severity

## Common Pitfalls

### Testing Anti-Patterns

❌ **Writing tests after code** → Use `tdd-workflow` to enforce test-first
❌ **Flaky tests with timeouts** → Apply `flaky-test-elimination` patterns
❌ **Over-testing with E2E** → Follow testing pyramid in `e2e-testing`
❌ **Missing integration tests** → Use `integration-testing` for boundaries
❌ **Testing mock behavior** → See `tdd-workflow` for proper mocking

### Code Quality Issues

❌ **Skipping code reviews** → Use `code-review-patterns` systematically
❌ **No automated security scanning** → Set up `static-analysis` in CI
❌ **High false positive rate** → Tune rules per `static-analysis` guidelines
❌ **Inconsistent review standards** → Use scoring framework from `code-review-patterns`

## Skill Dependencies

### Recommended Learning Path

1. **Foundation**: Start with `tdd-workflow`
   - Master test-first development
   - Understand Red-Green-Refactor

2. **Reliability**: Add `flaky-test-elimination`
   - Eliminate timing issues
   - Proper async handling

3. **Integration**: Learn `integration-testing`
   - API testing patterns
   - Database testing

4. **E2E**: Master `e2e-testing`
   - Critical path coverage
   - Cross-browser testing

5. **Quality**: Apply `code-review-patterns`
   - Systematic reviews
   - Quality scoring

6. **Automation**: Implement `static-analysis`
   - CI/CD integration
   - Security scanning

## Complementary Skills

These skills work well with other skill sets:

- **Python Skills**: `python/testing-patterns.md` for pytest-specific patterns
- **Finance Skills**: Technical indicators require robust testing
- **Backend Development**: Integration testing crucial for API reliability
- **Frontend Development**: E2E testing validates user experience

## Quality Metrics

### Coverage Targets

- **Unit Tests**: 80-90% line coverage (from `tdd-workflow`)
- **Integration Tests**: 70-80% API coverage (from `integration-testing`)
- **E2E Tests**: 100% critical path coverage (from `e2e-testing`)

### Reliability Metrics

- **Flakiness Rate**: <0.1% (from `flaky-test-elimination`)
- **Code Review Quality Score**: >75/100 (from `code-review-patterns`)
- **Static Analysis**: 0 critical, <5 high severity (from `static-analysis`)

### Performance Targets

- **Unit Test Suite**: <1 minute
- **Integration Test Suite**: <5 minutes
- **E2E Test Suite**: <10 minutes
- **Static Analysis**: <5 minutes

## Contributing

When adding new testing/quality skills:

1. Include proper YAML frontmatter with trigger keywords
2. Provide code examples for primary use cases
3. Document anti-patterns and common mistakes
4. Include quality standards and metrics
5. Link to related skills and complementary resources

## Related Resources

- **Existing Skill**: `/skills/python/testing-patterns.md` - Python-specific testing (pytest, fixtures, mocking)
- **Documentation**: Each skill includes references section for deep dives
- **Examples**: Code examples throughout skills are production-ready

---

**Skill Collection**: Testing & Code Quality
**Skill Count**: 6 comprehensive skills
**Coverage**: Full testing lifecycle from TDD to production quality gates
**Last Updated**: 2025-12-11
