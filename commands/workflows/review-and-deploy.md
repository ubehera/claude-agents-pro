---
description: Complete quality review and deployment pipeline
args: [target] [--env dev|staging|prod] [--checks all|security|performance]
tools: Task, TodoWrite, Read, Bash(git:*), Bash(gh:*)
model: opus
---

# /review-and-deploy - Quality Review & Deployment Pipeline

Orchestrates a comprehensive review and deployment workflow through multiple specialized agents with enforced quality gates and automated deployment processes.

## Usage

```bash
/review-and-deploy [options]
```

## Options

- `--env <environment>` - Target deployment environment (dev/staging/prod)
- `--skip-gates` - Skip non-critical quality gates (dev only)
- `--parallel-reviews` - Run security and performance reviews in parallel
- `--rollback-ready` - Prepare automated rollback mechanisms
- `--canary` - Deploy with canary release strategy

## Workflow Stages

### Stage 1: Pre-Review Analysis
**Agent**: `code-reviewer`
- Code diff analysis and complexity assessment
- Test coverage verification
- Breaking changes identification
- Review scope and risk assessment

**Quality Gate**: Code readiness check (>80% coverage, no critical issues)

### Stage 2: Parallel Review Process

#### Code Quality Review
**Agent**: `code-reviewer`
- Code style and pattern compliance
- Architecture consistency validation
- Dependency analysis and security scanning
- Documentation completeness check

#### Security Review
**Agent**: `security-architect`
- Threat model validation
- Vulnerability scanning (SAST/DAST)
- Authentication/authorization review
- Data protection compliance check

#### Performance Review
**Agent**: `performance-optimization-specialist`
- Performance regression analysis
- Load testing validation
- Resource utilization assessment
- Scalability impact evaluation

**Quality Gate**: All reviews pass with acceptable scores

### Stage 3: Infrastructure Preparation
**Agent**: `devops-automation-expert`
- Environment provisioning validation
- CI/CD pipeline execution
- Configuration management
- Infrastructure security compliance

**Agent**: `aws-cloud-architect` (if AWS deployment)
- Resource allocation optimization
- Network security validation
- Service mesh configuration
- Auto-scaling configuration

### Stage 4: Pre-Deployment Validation
**Agent**: `test-engineer`
- Integration test execution
- End-to-end test validation
- Smoke test preparation
- Rollback test verification

**Quality Gate**: All automated tests pass

### Stage 5: Deployment Orchestration
**Agent**: `devops-automation-expert`
- Blue-green or canary deployment execution
- Database migration coordination
- Service dependency management
- Traffic routing configuration

### Stage 6: Post-Deployment Monitoring
**Agent**: `observability-engineer`
- Real-time metrics monitoring
- Alert threshold validation
- Log analysis and error tracking
- Performance baseline establishment

**Agent**: `sre-incident-responder`
- Incident response readiness
- Rollback procedure validation
- Health check monitoring
- Service level objective tracking

**Quality Gate**: Service health validation within acceptable parameters

## Example Workflows

### Production Deployment
```bash
/review-and-deploy --env prod --parallel-reviews --rollback-ready --canary
```
**Timeline**: ~45-60 minutes
1. **Pre-Review** (5 min): `code-reviewer` analyzes changes
2. **Parallel Reviews** (20 min): `security-architect` + `performance-optimization-specialist`
3. **Infrastructure Prep** (10 min): `devops-automation-expert` + `aws-cloud-architect`
4. **Pre-Deploy Tests** (10 min): `test-engineer` validates readiness
5. **Canary Deployment** (5 min): `devops-automation-expert` deploys 10% traffic
6. **Monitoring** (15 min): `observability-engineer` + `sre-incident-responder` validate

### Development Deployment
```bash
/review-and-deploy --env dev --skip-gates
```
**Timeline**: ~15-20 minutes
1. **Quick Review** (5 min): `code-reviewer` basic validation
2. **Fast Deploy** (5 min): `devops-automation-expert` direct deployment
3. **Smoke Tests** (5 min): `test-engineer` basic functionality validation
4. **Dev Monitoring** (5 min): `observability-engineer` basic metrics

### Staging Deployment with Full Validation
```bash
/review-and-deploy --env staging --parallel-reviews
```
**Timeline**: ~30-40 minutes
1. **Comprehensive Review** (15 min): All review agents in parallel
2. **Infrastructure Validation** (10 min): Platform team preparation
3. **Full Test Suite** (10 min): `test-engineer` complete validation
4. **Staging Deployment** (5 min): Standard deployment process

## Quality Gates Configuration

### Critical Gates (Cannot Skip)
- Security vulnerability scan: PASS
- Authentication/authorization validation: APPROVED
- Database migration safety: VALIDATED
- Rollback procedure: TESTED

### Standard Gates (Configurable)
- Code coverage: >80%
- Performance regression: <10% degradation
- Documentation completeness: >90%
- Integration tests: ALL PASS

### Environment-Specific Gates
```yaml
production:
  - Load test validation: REQUIRED
  - Canary deployment: MANDATORY
  - Rollback test: VALIDATED
  - SLO compliance: VERIFIED

staging:
  - Integration tests: FULL SUITE
  - Performance tests: BASELINE
  - Security scan: COMPREHENSIVE

development:
  - Smoke tests: BASIC
  - Unit tests: >70% coverage
  - Security scan: MINIMAL
```

## Error Handling & Rollback

### Automatic Rollback Triggers
- Critical service errors (>5% error rate)
- Performance degradation (>50% latency increase)
- Security incident detection
- Infrastructure failure

### Rollback Orchestration
**Agent**: `sre-incident-responder`
1. Immediate traffic routing to previous version
2. Service health verification
3. Incident notification and logging
4. Post-rollback analysis coordination

### Post-Incident Analysis
**Agent**: `orchestration-coordinator`
1. Multi-agent retrospective coordination
2. Root cause analysis compilation
3. Process improvement recommendations
4. Quality gate refinement suggestions

## Monitoring & Observability

### Real-Time Dashboards
- Deployment progress tracking
- Quality gate status indicators
- Service health metrics
- Performance baseline comparisons

### Alert Integration
- Quality gate failures → Immediate notification
- Deployment errors → Escalation protocols
- Performance regressions → Automatic analysis
- Security incidents → Rapid response activation

## Configuration Examples

### Minimal Fast Deployment
```yaml
environment: development
quality_gates:
  critical_only: true
  coverage_threshold: 50%
agents:
  - code-reviewer (quick scan)
  - test-engineer (smoke tests)
  - devops-automation-expert (direct deploy)
```

### Full Production Pipeline
```yaml
environment: production
quality_gates:
  all_enabled: true
  coverage_threshold: 90%
  performance_threshold: strict
deployment_strategy: canary
agents:
  - code-reviewer
  - security-architect
  - performance-optimization-specialist
  - devops-automation-expert
  - aws-cloud-architect
  - test-engineer
  - observability-engineer
  - sre-incident-responder
```

The review and deployment pipeline ensures consistent quality standards while providing flexibility for different environments and deployment strategies.