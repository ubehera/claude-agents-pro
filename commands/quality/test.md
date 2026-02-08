---
description: Execute comprehensive testing workflow with coverage and quality gates
args: [scope] [--coverage-target 80] [--type unit|integration|e2e|all]
tools: Task, Bash(npm test:*), Bash(pytest:*), Bash(go test:*), Read, Grep
model: sonnet
---

# Comprehensive Testing Workflow

Execute testing workflow for **$ARGUMENTS** with quality validation.

## Test Scope Analysis

Determine testing scope:
- **No arguments**: Test entire codebase
- **File/directory path**: Test specific module
- **Feature name**: Test related components
- **"all"**: Full test suite with all types

## Test Types

### Unit Testing (Default)
- Target: Individual functions and methods
- Coverage goal: >80% (configurable)
- Tools: Framework-specific (pytest, jest, go test, etc.)
- Delegate to test-engineer for test generation

### Integration Testing
- Target: Component interactions
- Coverage goal: >70%
- Focus: API contracts, database operations, service communication
- Validate with actual dependencies

### End-to-End Testing
- Target: Critical user journeys
- Coverage goal: 100% critical paths
- Tools: Playwright, Cypress, Selenium
- Focus on business-critical workflows

### Performance Testing
- Load testing with realistic scenarios
- Stress testing to find breaking points
- P95 latency validation (<200ms target)
- Resource utilization monitoring

## Test Execution Strategy

1. **Environment Setup**
   - Check test framework availability
   - Verify test configuration
   - Set up test database/mocks if needed

2. **Pre-flight Checks**
   - Lint code for basic issues
   - Type checking (if applicable)
   - Dependency validation

3. **Test Execution**
   ```bash
   # Detect and run appropriate test command
   npm test -- --coverage
   pytest --cov --cov-report=html
   go test -cover ./...
   ```

4. **Coverage Analysis**
   - Generate coverage reports
   - Identify uncovered code paths
   - Suggest additional test cases via test-engineer

5. **Quality Gates**
   - **MVP**: 70% coverage, happy paths tested
   - **Standard**: 80% coverage, error cases covered
   - **Enterprise**: 90% coverage, edge cases validated

## Test Generation

If coverage is below target:
1. Use Task to invoke test-engineer
2. Generate missing test cases
3. Focus on high-risk areas first
4. Add regression tests for bugs

## Continuous Testing

Set up watch mode for development:
```bash
npm test -- --watch
pytest-watch
go test -watch
```

## Test Report

Generate comprehensive report including:
- Overall coverage percentage
- Coverage by file/module
- Test execution time
- Failed test details
- Flaky test identification
- Performance metrics

## Integration Points

- **Pre-commit**: Run fast unit tests
- **CI/CD**: Full test suite with coverage gates
- **Pre-deployment**: E2E and performance validation
- **Post-deployment**: Smoke tests and monitoring

## Examples

```
/test                                    # Test entire project
/test src/auth --coverage-target 90     # Test auth module with 90% target
/test --type e2e                        # Run E2E tests only
/test "user authentication" --type all   # All test types for feature
```

## Follow-up Actions

After testing:
1. Fix failing tests or implementation
2. Add missing test cases for uncovered code
3. Update test documentation
4. Configure CI/CD test gates
5. Set up test monitoring dashboards