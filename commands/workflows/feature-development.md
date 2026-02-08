---
description: Execute complete feature development workflow with DDD and quality gates
args: [feature-context] [--stage=all|requirements|design|api|implementation|testing] [--ddd] [--parallel]
tools: Task, TodoWrite, Read, Write
model: sonnet
---

## Purpose
Complete feature development workflow using Domain-Driven Design principles with automated agent coordination, quality gates, and progress tracking.

This workflow implements a three-phase quality-gated system based on production-proven patterns (see `/patterns/quality-gates/`):
- **Phase 1: Analysis** (95% threshold) - Requirements, domain modeling, architecture
- **Phase 2: Implementation** (80% threshold) - Code, tests, security, review
- **Phase 3: Validation** (85% threshold) - Integration tests, performance, documentation, acceptance

## Quality Gates System

Quality gates enforce measurable checkpoints at each phase, preventing downstream failures by catching issues early. Each gate must meet its threshold before proceeding to the next phase.

**Quality Gate Checker**: `/scripts/quality-gate-checker.py`
**Documentation**: `/patterns/quality-gates/README.md`
**Templates**: `/templates/quality-gates/`

### Phase Mapping

The workflow stages map to quality gate phases:
- **Stages 1-3** (Requirements, Domain, API) → **Phase 1: Analysis** (95% gate)
- **Stage 4** (Implementation) → **Phase 2: Implementation** (80% gate)
- **Stage 5** (Testing) → **Phase 3: Validation** (85% gate)

## Workflow Stages

### Stage 1: Requirements Analysis
**Phase**: Analysis (Phase 1)
**Quality Gate**: 95%
**Agents**: `orchestration-coordinator` → `system-design-specialist`
**Duration**: 1-2 hours
**Deliverables**:
- Business requirements specification
- User journey mapping
- Acceptance criteria definition
- Risk assessment and mitigation
**Artifact**: Use `templates/quality-gates/ANALYSIS_REPORT.template.md`

### Stage 2: Domain Modeling
**Phase**: Analysis (Phase 1)
**Quality Gate**: 95% (cumulative with Stage 1)
**Agents**: `domain-modeling-expert` (strategic DDD), `system-design-specialist` (architecture implications)
**Command**: `/domain-model` for event storming and context mapping
**Duration**: 2-4 hours
**Deliverables**:
- Bounded contexts identification
- Domain model with aggregates
- Event storming results
- Ubiquitous language definition
**Artifact**: Updates `templates/quality-gates/ANALYSIS_REPORT.template.md`

### Stage 3: API Design
**Phase**: Analysis (Phase 1)
**Quality Gate**: 95% (cumulative)
**Agents**: `api-platform-engineer`
**Duration**: 1-3 hours
**Deliverables**:
- OpenAPI/GraphQL specifications
- API contract definitions
- Authentication/authorization patterns
- Error handling strategies
**Gate Check**: Run `python3 scripts/quality-gate-checker.py --phase analysis`

### Stage 4: Implementation
**Phase**: Implementation (Phase 2)
**Quality Gate**: 80%
**Agents**: Technology-specific agents based on stack
**Duration**: 4-16 hours
**Deliverables**:
- Core domain logic implementation
- API layer with validation
- Unit tests (≥80% coverage)
- Integration tests
- Security scan passed (0 critical/high vulnerabilities)
- Lint errors = 0
- Code review approved
**Artifact**: Use `templates/quality-gates/IMPLEMENTATION_SUMMARY.template.md`
**Gate Check**: Run `python3 scripts/quality-gate-checker.py --phase implementation --coverage-report coverage/lcov.info`

### Stage 5: Testing & Validation
**Phase**: Validation (Phase 3)
**Quality Gate**: 85%
**Agents**: `test-engineer`, `security-architect`, `performance-optimization-specialist`
**Duration**: 2-6 hours
**Deliverables**:
- End-to-end test suite (all passing)
- Performance benchmarks (meeting SLA targets)
- Complete documentation (API specs, runbooks, architecture)
- Acceptance criteria validated by stakeholders
**Artifact**: Use `templates/quality-gates/VALIDATION_REPORT.template.md`
**Gate Check**: Run `python3 scripts/quality-gate-checker.py --phase validation --integration-report test-results.json`

## Usage Examples

```bash
# Full feature development workflow with quality gates
/workflow-feature-development ./features/user-management

# Execute specific stage
/workflow-feature-development ./features/payments --stage=api

# DDD-focused approach with domain modeling
/workflow-feature-development ./features/inventory --ddd --stage=design

# Parallel execution where possible
/workflow-feature-development ./features/notifications --parallel --stage=implementation

# Skip to implementation (development mode - gates still validated but not blocking)
/workflow-feature-development ./features/search --stage=implementation --skip-gates

# Check quality gate status manually
python3 scripts/quality-gate-checker.py --phase analysis --config ./features/user-auth/.quality-gates.json
python3 scripts/quality-gate-checker.py --phase implementation --coverage-report coverage/lcov.info --exit-code
python3 scripts/quality-gate-checker.py --phase all --strict --output quality-report.json
```

## Input Validation

```bash
# Validate feature context
FEATURE_CONTEXT=$1
STAGE=${2#--stage=}
FLAGS="$3 $4 $5"

if [ -z "$FEATURE_CONTEXT" ]; then
  echo "❌ Error: Feature context required"
  echo "💡 Usage: /workflow-feature-development [feature-path] [--stage] [--options]"
  echo "📚 Examples:"
  echo "  /workflow-feature-development ./features/user-auth"
  echo "  /workflow-feature-development ./src/payments --stage=api"
  echo "  /workflow-feature-development ./modules/inventory --ddd"
  exit 1
fi

# Create feature directory if it doesn't exist
if [ ! -d "$FEATURE_CONTEXT" ]; then
  echo "📏 Creating feature directory: $FEATURE_CONTEXT"
  mkdir -p "$FEATURE_CONTEXT"
fi

# Validate stage parameter
case "$STAGE" in
  "all"|"")
    STAGE="all"
    echo "🚀 Executing full feature development workflow"
    ;;
  "requirements"|"design"|"api"|"implementation"|"testing")
    echo "🎨 Executing stage: $STAGE"
    ;;
  *)
    echo "❌ Error: Invalid stage: $STAGE"
    echo "💡 Available stages: all, requirements, design, api, implementation, testing"
    exit 1
    ;;
esac
```

## Workflow Implementation

### Progress Tracking Setup

Create comprehensive TodoWrite tracking for the feature:

**Feature Development: $FEATURE_CONTEXT**
- [ ] Phase 1: Analysis (Stages 1-3) (Target: 95% quality gate)
  - [ ] Stage 1: Requirements Analysis
  - [ ] Stage 2: Domain Modeling
  - [ ] Stage 3: API Design
- [ ] Phase 2: Implementation (Stage 4) (Target: 80% quality gate)
  - [ ] Core domain logic with tests
  - [ ] Security scan passed
  - [ ] Code review approved
- [ ] Phase 3: Validation (Stage 5) (Target: 85% quality gate)
  - [ ] Integration tests passing
  - [ ] Performance benchmarks met
  - [ ] Documentation complete
  - [ ] Acceptance criteria validated

**Quality Gate Configuration**:
Initialize `.quality-gates.json` in feature directory:
```json
{
  "project": "feature-name",
  "thresholds": {
    "analysis": 95,
    "implementation": 80,
    "validation": 85
  },
  "phase1": {
    "requirements_complete": false,
    "domain_model_validated": false,
    "architecture_documented": false,
    "risks_identified": false
  },
  "phase2": {
    "test_coverage": 0,
    "security_scan": "fail",
    "lint_errors": 0,
    "code_review": "fail"
  },
  "phase3": {
    "integration_tests": "fail",
    "performance_benchmarks": "fail",
    "documentation_complete": false,
    "acceptance_criteria_met": false
  }
}
```

### Stage Execution Logic

```bash
# Execute workflow based on stage selection
case "$STAGE" in
  "all")
    execute_full_workflow "$FEATURE_CONTEXT" "$FLAGS"
    ;;
  "requirements")
    execute_requirements_stage "$FEATURE_CONTEXT" "$FLAGS"
    ;;
  "design")
    execute_design_stage "$FEATURE_CONTEXT" "$FLAGS"
    ;;
  "api")
    execute_api_stage "$FEATURE_CONTEXT" "$FLAGS"
    ;;
  "implementation")
    execute_implementation_stage "$FEATURE_CONTEXT" "$FLAGS"
    ;;
  "testing")
    execute_testing_stage "$FEATURE_CONTEXT" "$FLAGS"
    ;;
esac

# Quality gate validation
validate_quality_gate() {
  local phase=$1
  local context=$2

  echo "🔍 Quality Gate Check: Phase $phase"

  # Run quality-gate-checker.py
  local config_file="$context/.quality-gates.json"

  if [ ! -f "$config_file" ]; then
    echo "⚠️  Warning: .quality-gates.json not found in $context"
    echo "⚠️  Creating default configuration..."
    create_default_gate_config "$context"
  fi

  case "$phase" in
    "analysis")
      python3 scripts/quality-gate-checker.py \
        --phase analysis \
        --config "$config_file"
      ;;
    "implementation")
      python3 scripts/quality-gate-checker.py \
        --phase implementation \
        --config "$config_file" \
        --coverage-report "$context/coverage/lcov.info" \
        --lint-report "$context/lint-results.json"
      ;;
    "validation")
      python3 scripts/quality-gate-checker.py \
        --phase validation \
        --config "$config_file" \
        --integration-report "$context/test-results/integration.json" \
        --perf-report "$context/benchmarks/k6-results.json"
      ;;
  esac

  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "✅ Quality gate PASSED for phase: $phase"
    return 0
  else
    echo "❌ Quality gate FAILED for phase: $phase"
    echo "💡 Review gate criteria: patterns/quality-gates/phase-$phase.md"
    return 1
  fi
}

# Create default quality gate configuration
create_default_gate_config() {
  local context=$1
  cat > "$context/.quality-gates.json" << 'EOF'
{
  "project": "feature-name",
  "thresholds": {
    "analysis": 95,
    "implementation": 80,
    "validation": 85
  },
  "phase1": {
    "requirements_complete": false,
    "domain_model_validated": false,
    "architecture_documented": false,
    "risks_identified": false
  },
  "phase2": {
    "test_coverage": 0,
    "security_scan": "fail",
    "lint_errors": 0,
    "code_review": "fail"
  },
  "phase3": {
    "integration_tests": "fail",
    "performance_benchmarks": "fail",
    "documentation_complete": false,
    "acceptance_criteria_met": false
  }
}
EOF
  echo "✅ Created default .quality-gates.json in $context"
}
```

### Agent Coordination Sequence

#### Stage 1: Requirements Analysis
Delegate to `orchestration-coordinator` for requirements extraction:

**Task**: Requirements analysis and business context extraction for feature: $FEATURE_CONTEXT

**Instructions**:
1. Analyze business requirements and user needs
2. Define acceptance criteria and success metrics
3. Identify stakeholders and constraints
4. Create user journey mapping
5. Assess risks and define mitigation strategies
6. Use template: `templates/quality-gates/ANALYSIS_REPORT.template.md`
7. Update `.quality-gates.json` with phase1 status
8. Prepare context for domain modeling stage

**Quality Gate**: Phase 1 Analysis (95% threshold) - Part 1 of 3
**Artifact**: Analysis report covering requirements section

#### Stage 2: Domain Modeling (DDD)
Delegate to `domain-modeling-expert` for strategic DDD analysis:

**Task**: Domain-driven design analysis for feature: $FEATURE_CONTEXT

**Instructions**:
1. Apply Domain-Driven Design principles with event storming
2. Identify bounded contexts and their relationships using context mapping
3. Define domain model with aggregates, entities, and value objects
4. Establish ubiquitous language and domain glossary
5. Model domain events and command flows
6. Define integration patterns between contexts (ACL, OHS, etc.)
7. Coordinate with `system-design-specialist` for architectural implications
8. Update analysis report with domain model section
9. Update `.quality-gates.json` phase1.domain_model_validated

**Command**: Use `/domain-model $FEATURE_CONTEXT --technique event-storming` for facilitation

**Quality Gate**: Phase 1 Analysis (95% threshold) - Part 2 of 3
**Artifact**: Analysis report with domain modeling complete

#### Stage 3: API Design
Delegate to `api-platform-engineer` for contract specification:

**Task**: API design and contract specification for feature: $FEATURE_CONTEXT

**Instructions**:
1. Design RESTful APIs following OpenAPI 3.0 standards
2. Define authentication and authorization patterns
3. Specify error handling and validation rules
4. Design event-driven APIs for domain events
5. Create API documentation and examples
6. Ensure API governance compliance
7. Complete architecture and risk sections of analysis report
8. Update `.quality-gates.json` - mark all phase1 criteria complete
9. Run gate checker: `python3 scripts/quality-gate-checker.py --phase analysis`

**Quality Gate**: Phase 1 Analysis (95% threshold) - Final check
**Gate Check**: Must pass before proceeding to Phase 2 (Implementation)
**Artifact**: Complete analysis report with all sections filled

#### Stage 4: Implementation
Delegate to technology-specific agents based on project stack:

**Task**: Implementation of feature: $FEATURE_CONTEXT

**Technology Selection**:
- TypeScript/Node.js: `typescript-architect`
- Python: `python-expert`
- Frontend: `frontend-expert`
- Backend Services: `backend-architect`

**Instructions**:
1. Implement core domain logic with proper separation
2. Build API layer with validation and error handling
3. Create unit tests with ≥80% coverage
4. Implement integration tests for API contracts
5. Follow project coding standards and patterns
6. Run security scan: `npm audit --audit-level=high` (0 critical/high vulnerabilities)
7. Run linter: `npm run lint` (0 errors)
8. Submit pull request and obtain code review approval
9. Use template: `templates/quality-gates/IMPLEMENTATION_SUMMARY.template.md`
10. Update `.quality-gates.json` phase2 with actual metrics
11. Run gate checker: `python3 scripts/quality-gate-checker.py --phase implementation --coverage-report coverage/lcov.info`

**Quality Gate**: Phase 2 Implementation (80% threshold)
**Gate Check**: Must pass before proceeding to Phase 3 (Validation)
**Artifact**: Implementation summary with coverage, security, and review results

#### Stage 5: Testing & Validation
Delegate to quality specialists for comprehensive validation:

**Testing**: `test-engineer`
**Security**: `security-architect`
**Performance**: `performance-optimization-specialist`

**Instructions**:
1. Create end-to-end test suite covering critical user journeys
2. Execute integration tests across service boundaries (all must pass)
3. Run performance benchmarks and validate against SLA targets
4. Complete API documentation (OpenAPI/GraphQL specs)
5. Create operational runbook with troubleshooting guide
6. Validate acceptance criteria with stakeholders
7. Obtain stakeholder sign-off
8. Use template: `templates/quality-gates/VALIDATION_REPORT.template.md`
9. Update `.quality-gates.json` phase3 with results
10. Run gate checker: `python3 scripts/quality-gate-checker.py --phase validation --integration-report test-results.json --perf-report benchmarks/k6-results.json`

**Quality Gate**: Phase 3 Validation (85% threshold)
**Gate Check**: Must pass for production readiness
**Artifact**: Validation report with production deployment checklist

## Feature Flags & Options

### DDD Flag (`--ddd`)
When `--ddd` flag is provided:
- Enhanced domain modeling with event storming
- Deeper bounded context analysis
- More thorough ubiquitous language development
- Extended design stage duration

### Parallel Flag (`--parallel`)
When `--parallel` flag is provided:
- Execute independent tasks simultaneously
- Coordinate parallel agent execution
- Merge results with conflict resolution
- Faster overall workflow completion

### Skip Gates Flag (`--skip-gates`)
Development mode option:
- Skip quality gate validation
- Faster iteration cycles
- Still track progress and metrics
- Warning about skipped validation

## Success Metrics & Reporting

### Quality Gate Metrics
Quality gates provide objective measurements across all phases:

**Phase 1: Analysis (95% threshold)**
- requirements_complete: boolean
- domain_model_validated: boolean
- architecture_documented: boolean
- risks_identified: boolean

**Phase 2: Implementation (80% threshold)**
- test_coverage: numeric ≥80%
- security_scan: pass/fail
- lint_errors: numeric = 0
- code_review: pass/fail

**Phase 3: Validation (85% threshold)**
- integration_tests: pass/fail
- performance_benchmarks: pass/fail
- documentation_complete: boolean
- acceptance_criteria_met: boolean

### Workflow Completion Report

Generate comprehensive report with quality gate results:

```bash
# Export all gate results to JSON
python3 scripts/quality-gate-checker.py \
  --phase all \
  --config .quality-gates.json \
  --coverage-report coverage/lcov.info \
  --integration-report test-results.json \
  --perf-report benchmarks/k6-results.json \
  --output quality-report.json
```

```
🏁 Feature Development Complete: [feature-name]
⏱️ Duration: [total-time]
📋 Phases: [completed-phases]

📈 Quality Gates Results:
✅ Phase 1 (Analysis): 100% (threshold: 95%)
✅ Phase 2 (Implementation): 100% (threshold: 80%)
✅ Phase 3 (Validation): 100% (threshold: 85%)

📊 Overall Score: PASSED - Production Ready

📝 Deliverables:
- Analysis Report: [feature-context]/analysis-report.md
- Domain Model: [feature-context]/domain-model.md
- API Specifications: [feature-context]/docs/api/
- Implementation: [feature-context]/src/
- Implementation Summary: [feature-context]/implementation-summary.md
- Test Suite: [feature-context]/tests/
- Validation Report: [feature-context]/validation-report.md
- Quality Gate Results: [feature-context]/quality-report.json

🚀 Next Steps:
- Schedule production deployment
- Configure monitoring and alerts
- Brief on-call team on runbook
- Execute gradual rollout with feature flags
- Monitor success metrics for 7-14 days
```

## Error Recovery

If any stage fails:
1. **Identify Failure Point**: Log detailed error information
2. **Quality Gate Analysis**: Determine if threshold can be adjusted
3. **Agent Retry**: Attempt stage re-execution with refined context
4. **Manual Intervention**: Provide specific remediation steps
5. **Workflow Resumption**: Continue from last successful stage

### Common Failure Scenarios
- **Requirements Unclear**: Request stakeholder clarification
- **Domain Complexity**: Break into smaller bounded contexts
- **API Conflicts**: Review existing API standards and patterns
- **Implementation Blocks**: Suggest alternative approaches or patterns
- **Quality Issues**: Provide specific improvement recommendations