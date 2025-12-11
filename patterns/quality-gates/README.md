# Quality Gates System

## Overview

Quality gates enforce measurable checkpoints throughout the development workflow, ensuring work meets defined standards before proceeding. This system implements a three-phase quality-gated workflow based on proven patterns from zhsama's production-grade systems.

## Philosophy

**Quality gates prevent downstream failures by catching issues early**. Each gate represents a commitment to specific quality thresholds that must be met before advancing. This approach:

- Reduces rework by identifying issues at the earliest possible phase
- Enforces consistent quality standards across all feature development
- Provides measurable metrics for project health and progress
- Enables informed go/no-go decisions at critical junctions

## Three-Phase Model

### Phase 1: Analysis (95% threshold)
**Focus**: Understanding the problem completely before writing code.

**Critical Checks**:
- Requirements completeness and stakeholder alignment
- Domain model validation with bounded contexts
- Architecture documentation with decision rationale
- Risk identification and mitigation strategies

**Why 95%**: Analysis errors compound exponentially through implementation. A 5% miss in requirements can result in 50% rework.

### Phase 2: Implementation (80% threshold)
**Focus**: Translating design into production-ready code.

**Critical Checks**:
- Test coverage ≥80% for core logic
- Security scanning with zero critical vulnerabilities
- Lint errors eliminated (zero tolerance)
- Code review approval from qualified reviewer

**Why 80%**: Balances quality with velocity. Higher thresholds risk analysis paralysis; lower thresholds accumulate technical debt.

### Phase 3: Validation (85% threshold)
**Focus**: Proving the solution works in realistic conditions.

**Critical Checks**:
- Integration tests passing across all critical paths
- Performance benchmarks meeting SLA targets
- Documentation complete (API specs, runbooks, architecture)
- Acceptance criteria validated by stakeholders

**Why 85%**: Pre-production validation must be thorough but pragmatic. Real-world edge cases often emerge post-launch.

## Usage Patterns

### CI/CD Integration

```bash
# Pre-commit hook - enforce Phase 2 gates
scripts/quality-gate-checker.py --phase implementation --config .quality-gates.json

# Pre-merge validation - enforce all gates
scripts/quality-gate-checker.py --phase all --strict

# Manual gate check for specific phase
scripts/quality-gate-checker.py --phase analysis --report-format json > analysis-report.json
```

### Workflow Integration

Quality gates integrate with `/workflow-feature-development`:

```bash
# Execute workflow with automatic gate enforcement
/workflow-feature-development ./features/user-auth --gates-enabled

# Check specific gate status
/workflow-feature-development ./features/payments --check-gate=implementation

# Skip gates for rapid prototyping (development mode)
/workflow-feature-development ./features/search --skip-gates --stage=implementation
```

### Manual Gate Validation

When working outside automated workflows, validate gates manually:

```bash
# Check analysis phase gate
python3 scripts/quality-gate-checker.py --phase analysis \
  --config ./features/user-management/.quality-gates.json

# Validate implementation gate with coverage report
python3 scripts/quality-gate-checker.py --phase implementation \
  --coverage-report ./coverage/lcov.info \
  --lint-report ./lint-results.json

# Full validation before production deployment
python3 scripts/quality-gate-checker.py --phase validation \
  --perf-benchmarks ./benchmarks/results.json \
  --exit-code
```

## Configuration

### Quality Gate Configuration File

Each feature maintains a `.quality-gates.json` file:

```json
{
  "project": "feature-name",
  "thresholds": {
    "analysis": 95,
    "implementation": 80,
    "validation": 85
  },
  "phase1": {
    "requirements_complete": true,
    "domain_model_validated": true,
    "architecture_documented": true,
    "risks_identified": true
  },
  "phase2": {
    "test_coverage": 85,
    "security_scan": "pass",
    "lint_errors": 0,
    "code_review": "pass"
  },
  "phase3": {
    "integration_tests": "pass",
    "performance_benchmarks": "pass",
    "documentation_complete": true,
    "acceptance_criteria_met": true
  }
}
```

### Threshold Customization

Adjust thresholds based on project risk profile:

```json
{
  "thresholds": {
    "analysis": 98,      // Critical: payment processing, authentication
    "implementation": 90, // High test coverage for regulated systems
    "validation": 95     // Extensive validation for public APIs
  }
}
```

## Artifacts

Each phase produces specific artifacts stored in feature directories:

```
features/user-management/
├── .quality-gates.json              # Gate configuration
├── analysis/
│   ├── requirements.md              # Business requirements
│   ├── domain-model.md              # DDD model with contexts
│   ├── architecture.md              # Design decisions (ADRs)
│   └── risk-register.md             # Identified risks
├── implementation/
│   ├── src/                         # Source code
│   ├── tests/                       # Test suite
│   ├── coverage/                    # Coverage reports
│   └── code-review.md               # Review outcomes
└── validation/
    ├── integration-tests/           # E2E test suite
    ├── performance/                 # Benchmark results
    ├── docs/                        # API specs, runbooks
    └── acceptance-sign-off.md       # Stakeholder approval
```

## Quality Gate Enforcement

### Automated Enforcement

Quality gates integrate with CI/CD pipelines:

```yaml
# .github/workflows/quality-gates.yml
name: Quality Gate Validation

on:
  pull_request:
    branches: [main, develop]

jobs:
  analysis-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Analysis Phase
        run: |
          python3 scripts/quality-gate-checker.py \
            --phase analysis \
            --config .quality-gates.json \
            --exit-code

  implementation-gate:
    runs-on: ubuntu-latest
    needs: analysis-gate
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: npm test -- --coverage
      - name: Security Scan
        run: npm audit --audit-level=high
      - name: Validate Implementation Phase
        run: |
          python3 scripts/quality-gate-checker.py \
            --phase implementation \
            --config .quality-gates.json \
            --coverage-report coverage/lcov.info \
            --exit-code

  validation-gate:
    runs-on: ubuntu-latest
    needs: implementation-gate
    steps:
      - uses: actions/checkout@v3
      - name: Integration Tests
        run: npm run test:integration
      - name: Performance Benchmarks
        run: npm run benchmark
      - name: Validate Validation Phase
        run: |
          python3 scripts/quality-gate-checker.py \
            --phase validation \
            --config .quality-gates.json \
            --exit-code
```

### Manual Enforcement

For manual workflows, enforce gates through code review checklists:

**Phase 1 Checklist** (see `phase-1-analysis.md`):
- [ ] Requirements documented with acceptance criteria
- [ ] Domain model validated by domain expert
- [ ] Architecture decisions recorded (ADRs)
- [ ] Risk register complete with mitigations

**Phase 2 Checklist** (see `phase-2-implementation.md`):
- [ ] Test coverage ≥80%
- [ ] Security scan passed
- [ ] Zero lint errors
- [ ] Code review approved

**Phase 3 Checklist** (see `phase-3-validation.md`):
- [ ] Integration tests passing
- [ ] Performance meets SLA targets
- [ ] Documentation complete
- [ ] Acceptance criteria validated

## Metrics & Reporting

### Gate Metrics Dashboard

Track quality gate performance across projects:

```bash
# Generate gate metrics report
python3 scripts/quality-gate-checker.py --metrics --all-projects

# Output example:
# Project: user-management
#   Phase 1 (Analysis): 96% ✓ (threshold: 95%)
#   Phase 2 (Implementation): 87% ✓ (threshold: 80%)
#   Phase 3 (Validation): 82% ✗ (threshold: 85%)
#   Status: BLOCKED at Phase 3
```

### Continuous Improvement

Use gate metrics to identify process improvements:

- **Frequent failures at Phase 1**: Improve requirements gathering process
- **Low implementation coverage**: Invest in test automation training
- **Validation bottlenecks**: Enhance integration test infrastructure

## References

- **Phase 1**: [Analysis Phase Gate](phase-1-analysis.md)
- **Phase 2**: [Implementation Phase Gate](phase-2-implementation.md)
- **Phase 3**: [Validation Phase Gate](phase-3-validation.md)
- **Checker Script**: `scripts/quality-gate-checker.py`
- **Templates**: `templates/quality-gates/`

## Best Practices

1. **Define gates early**: Establish quality criteria before starting work
2. **Automate checks**: Integrate gates into CI/CD pipelines
3. **Make visible**: Display gate status on dashboards and PRs
4. **Enforce consistently**: No exceptions without documented waivers
5. **Measure and improve**: Use gate metrics to refine thresholds
6. **Keep pragmatic**: Gates should prevent disasters, not block progress
