---
description: Comprehensive quality gate orchestration with multi-agent validation
args: [--level mvp|standard|enterprise] [--enforce strict|warning]
tools: Task, TodoWrite, Read, Grep
model: opus
---

# /quality-gates - Multi-Agent Quality Assurance Orchestration

Orchestrates comprehensive quality validation through multiple specialized agents with progressive quality gates, automated enforcement, and continuous quality monitoring throughout development workflows.

## Usage

```bash
/quality-gates <gate_type> [options]
```

## Quality Gate Types

### Development Quality Gates

#### Code Quality Gate
```bash
/quality-gates code-quality --scope <file|component|service|system> --strict
```

**Quality Agents**:
- `code-reviewer`: Code style, patterns, architecture compliance
- `test-engineer`: Test coverage and quality validation
- `performance-optimization-specialist`: Code performance analysis
- `security-architect`: Security code review

**Quality Metrics**:
```yaml
critical_criteria:
  test_coverage: >85%
  security_scan: CLEAN
  code_complexity: <15_cyclomatic
  dependency_scan: NO_CRITICAL_VULNERABILITIES

standard_criteria:
  code_style: COMPLIANT
  documentation: >80%_coverage
  type_safety: STRICT_MODE
  performance_regression: <10%
```

#### Integration Quality Gate
```bash
/quality-gates integration --services <service_list> --comprehensive
```

**Quality Agents**:
- `test-engineer`: Integration and contract testing
- `api-platform-engineer`: API contract validation
- `database-architect`: Data consistency verification
- `observability-engineer`: Integration monitoring validation

**Quality Metrics**:
```yaml
critical_criteria:
  integration_tests: ALL_PASS
  contract_compliance: VERIFIED
  data_consistency: VALIDATED
  service_health: HEALTHY

standard_criteria:
  response_times: <SLA_TARGETS
  error_rates: <1%
  monitoring_coverage: >90%
```

### Security Quality Gates

#### Security Validation Gate
```bash
/quality-gates security --depth <basic|comprehensive|audit> --compliance <standard>
```

**Quality Agents**:
- `security-architect`: Comprehensive security analysis
- `code-reviewer`: Security-focused code review
- `test-engineer`: Security testing validation
- `aws-cloud-architect`: Infrastructure security review

**Security Validation Layers**:
```yaml
layer_1_static_analysis:
  code_scanning: SAST_tools
  dependency_scanning: SCA_tools
  secrets_detection: SECRET_scanners
  compliance_check: POLICY_validation

layer_2_dynamic_analysis:
  vulnerability_scanning: DAST_tools
  penetration_testing: MANUAL_testing
  authentication_testing: AUTH_validation
  authorization_testing: AUTHZ_validation

layer_3_architecture_review:
  threat_modeling: STRIDE_analysis
  security_patterns: IMPLEMENTATION_review
  compliance_mapping: STANDARD_alignment
  incident_response: READINESS_validation
```

#### Compliance Quality Gate
```bash
/quality-gates compliance --standards "SOX,HIPAA,GDPR" --audit-ready
```

**Compliance Validation**:
- Data protection and privacy controls
- Audit trail and logging requirements
- Access control and authorization
- Encryption and data security standards

### Performance Quality Gates

#### Performance Validation Gate
```bash
/quality-gates performance --targets <latency|throughput|resource> --baseline-comparison
```

**Quality Agents**:
- `performance-optimization-specialist`: Performance analysis and optimization
- `test-engineer`: Load and stress testing
- `observability-engineer`: Performance monitoring validation
- `database-architect`: Database performance review

**Performance Criteria**:
```yaml
response_time_targets:
  p50_latency: <100ms
  p95_latency: <200ms
  p99_latency: <500ms

throughput_targets:
  requests_per_second: >1000
  concurrent_users: >500
  data_processing: >100MB/s

resource_utilization:
  cpu_usage: <80%
  memory_usage: <85%
  disk_io: <75%_saturation
  network_bandwidth: <70%_utilization
```

#### Scalability Quality Gate
```bash
/quality-gates scalability --load-factors <2x|5x|10x> --auto-scaling
```

**Scalability Validation**:
- Horizontal scaling behavior
- Load balancing effectiveness
- Resource bottleneck identification
- Auto-scaling trigger validation

### Deployment Quality Gates

#### Pre-Deployment Gate
```bash
/quality-gates pre-deployment --environment <staging|production> --comprehensive
```

**Quality Agents**:
- `devops-automation-expert`: Deployment readiness validation
- `observability-engineer`: Monitoring and alerting verification
- `sre-incident-responder`: Incident response readiness
- `security-architect`: Production security validation

**Deployment Readiness**:
```yaml
infrastructure_readiness:
  environment_provisioning: COMPLETE
  configuration_management: VALIDATED
  secrets_management: CONFIGURED
  network_security: VERIFIED

application_readiness:
  build_artifacts: VALIDATED
  database_migrations: TESTED
  feature_flags: CONFIGURED
  rollback_plan: PREPARED

monitoring_readiness:
  metrics_collection: ACTIVE
  log_aggregation: CONFIGURED
  alerting_rules: VALIDATED
  dashboard_availability: CONFIRMED
```

#### Post-Deployment Gate
```bash
/quality-gates post-deployment --validation-window <duration> --auto-rollback
```

**Post-Deployment Validation**:
- Service health verification
- Performance baseline confirmation
- Error rate monitoring
- User experience validation

## Progressive Quality Assurance

### Quality Gate Stages

#### Stage 1: Development Gates (Fast Feedback)
```yaml
duration: <5_minutes
gates:
  - Unit tests: PASS
  - Code compilation: SUCCESS
  - Basic security scan: CLEAN
  - Code style check: COMPLIANT

agents:
  - code-reviewer (lightweight scan)
  - test-engineer (unit tests only)
```

#### Stage 2: Integration Gates (Comprehensive)
```yaml
duration: <15_minutes
gates:
  - Integration tests: PASS
  - Security vulnerability scan: CLEAN
  - Performance regression check: <5%
  - API contract validation: COMPLIANT

agents:
  - test-engineer (full integration suite)
  - security-architect (comprehensive scan)
  - performance-optimization-specialist (regression analysis)
  - api-platform-engineer (contract validation)
```

#### Stage 3: Production Gates (Exhaustive)
```yaml
duration: <30_minutes
gates:
  - Load testing: TARGETS_MET
  - Security audit: APPROVED
  - Compliance validation: CERTIFIED
  - Disaster recovery: TESTED

agents:
  - All quality agents (comprehensive validation)
  - Specialized compliance and audit agents
```

### Quality Gate Orchestration

#### Parallel Quality Validation
```bash
/quality-gates parallel-validation --max-parallel 4
```

**Parallel Execution**:
- Code quality + Security scan + Performance test + Integration validation
- Independent quality dimensions validated simultaneously
- Consolidated quality report generation
- Fail-fast on critical quality issues

#### Sequential Quality Validation
```bash
/quality-gates sequential-validation --fail-fast
```

**Sequential Execution**:
1. Code quality (fastest feedback)
2. Security validation (critical for progression)
3. Performance validation (resource intensive)
4. Integration validation (comprehensive)

## Quality Enforcement & Automation

### Automated Quality Enforcement
```yaml
enforcement_policies:
  critical_failures:
    action: BLOCK_PROGRESSION
    notification: IMMEDIATE
    escalation: AUTOMATIC

  warning_failures:
    action: ALLOW_WITH_APPROVAL
    notification: DELAYED
    escalation: MANUAL

  informational:
    action: LOG_ONLY
    notification: SUMMARY
    escalation: NONE
```

### Quality Metrics Dashboard
```yaml
quality_dimensions:
  code_quality:
    metrics: [coverage, complexity, duplication, maintainability]
    trend: IMPROVING/STABLE/DEGRADING
    threshold: ABOVE/BELOW_TARGET

  security_posture:
    metrics: [vulnerabilities, compliance_score, threat_coverage]
    trend: IMPROVING/STABLE/DEGRADING
    risk_level: LOW/MEDIUM/HIGH/CRITICAL

  performance_health:
    metrics: [latency, throughput, error_rate, resource_usage]
    trend: IMPROVING/STABLE/DEGRADING
    sla_compliance: WITHIN/OUTSIDE_TARGETS
```

### Continuous Quality Monitoring
```yaml
monitoring_agents:
  - observability-engineer: Real-time quality metrics
  - performance-optimization-specialist: Performance drift detection
  - security-architect: Security posture monitoring
  - code-reviewer: Code quality trend analysis

monitoring_frequency:
  critical_metrics: REAL_TIME
  standard_metrics: HOURLY
  trend_analysis: DAILY
  comprehensive_reports: WEEKLY
```

## Quality Gate Examples

### Critical Production Release
```bash
/quality-gates production-release --zero-downtime --comprehensive-validation
```

**Gate Sequence**:
```yaml
gate_1_pre_validation: # 10 minutes
  agents: [code-reviewer, security-architect, test-engineer]
  criteria: [security_clean, tests_pass, coverage_85%]

gate_2_integration_validation: # 20 minutes
  agents: [test-engineer, api-platform-engineer, database-architect]
  criteria: [integration_tests_pass, contracts_valid, data_consistent]

gate_3_performance_validation: # 15 minutes
  agents: [performance-optimization-specialist, observability-engineer]
  criteria: [load_tests_pass, monitoring_active, baselines_established]

gate_4_deployment_readiness: # 10 minutes
  agents: [devops-automation-expert, sre-incident-responder]
  criteria: [infrastructure_ready, rollback_tested, incident_response_ready]

gate_5_post_deployment_validation: # 15 minutes
  agents: [observability-engineer, sre-incident-responder, test-engineer]
  criteria: [health_checks_pass, performance_stable, error_rates_low]
```

### Feature Branch Quality Gate
```bash
/quality-gates feature-branch --fast-feedback --essential-only
```

**Lightweight Gate Sequence**:
```yaml
gate_1_code_quality: # 3 minutes
  agents: [code-reviewer]
  criteria: [style_compliant, complexity_acceptable, coverage_70%]

gate_2_security_basics: # 2 minutes
  agents: [security-architect]
  criteria: [secrets_scan_clean, dependency_scan_clean]

gate_3_unit_validation: # 5 minutes
  agents: [test-engineer]
  criteria: [unit_tests_pass, no_regressions]
```

### Emergency Hotfix Quality Gate
```bash
/quality-gates emergency-hotfix --minimal-safe --expedited
```

**Expedited Gate Sequence**:
```yaml
gate_1_critical_validation: # 2 minutes
  agents: [code-reviewer, security-architect]
  criteria: [no_critical_vulnerabilities, syntax_valid]

gate_2_targeted_testing: # 3 minutes
  agents: [test-engineer]
  criteria: [affected_components_tested, no_breaking_changes]

gate_3_rapid_deployment: # 2 minutes
  agents: [devops-automation-expert]
  criteria: [rollback_ready, monitoring_active]
```

## Quality Gate Configuration

### Organization-Specific Standards
```yaml
enterprise_standards:
  code_coverage: 90%
  security_scan: comprehensive
  performance_targets: strict
  compliance: full_audit

startup_standards:
  code_coverage: 70%
  security_scan: essential
  performance_targets: baseline
  compliance: basic
```

### Environment-Specific Gates
```yaml
production:
  quality_level: MAXIMUM
  validation_time: COMPREHENSIVE
  approval_required: TRUE

staging:
  quality_level: HIGH
  validation_time: STANDARD
  approval_required: CONDITIONAL

development:
  quality_level: ESSENTIAL
  validation_time: MINIMAL
  approval_required: FALSE
```

The quality gates system ensures consistent, comprehensive quality validation through intelligent agent coordination while providing flexibility for different project contexts and organizational requirements.