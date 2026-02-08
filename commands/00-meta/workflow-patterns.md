---
description: Predefined multi-agent workflow patterns for common scenarios
args: <pattern-name> [--customize] [--quality-level mvp|standard|enterprise]
tools: Task, TodoWrite, Read, Write
model: opus
---

# /workflow-patterns - Reusable Multi-Agent Workflow Templates

Provides predefined, battle-tested workflow patterns for common development scenarios. Each pattern includes agent coordination, quality gates, and customizable parameters for different project contexts.

## Usage

```bash
/workflow-patterns <pattern_name> [options] [customizations]
```

## Core Workflow Patterns

### Development Patterns

#### Feature Development Pipeline
```bash
/workflow-patterns feature-pipeline --scope <full|backend|frontend> --quality <mvp|standard|enterprise>
```

**Pattern Flow**:
1. **Research & Requirements**: `research-librarian` → requirements analysis
2. **Architecture Design**: `system-design-specialist` → technical design
3. **API Specification**: `api-platform-engineer` → contract definition
4. **Parallel Implementation**:
   - Backend: `backend-architect` + language specialist
   - Frontend: `frontend-expert` (if full-stack)
   - Database: `database-architect` (if data changes)
5. **Integration Testing**: `test-engineer` → comprehensive testing
6. **Security Review**: `security-architect` → vulnerability assessment
7. **Performance Validation**: `performance-optimization-specialist` → optimization
8. **Deployment**: `devops-automation-expert` → production release

**Quality Gates**: Requirements >90%, tests pass, security clear, performance targets met

#### Bug Investigation & Resolution
```bash
/workflow-patterns bug-resolution --severity <critical|high|medium|low> --domain <frontend|backend|infrastructure>
```

**Pattern Flow**:
1. **Initial Triage**: `error-diagnostician` → problem analysis
2. **Root Cause Analysis**: Domain specialist → detailed investigation
3. **Impact Assessment**: `system-design-specialist` → system impact
4. **Solution Design**: Domain specialist → fix strategy
5. **Implementation**: Domain specialist → fix development
6. **Testing**: `test-engineer` → regression testing
7. **Code Review**: `code-reviewer` → quality validation
8. **Deployment**: `devops-automation-expert` → production fix

**Quality Gates**: Root cause identified, fix verified, no regressions introduced

#### Refactoring & Technical Debt
```bash
/workflow-patterns refactoring --scope <component|service|system> --impact <low|medium|high>
```

**Pattern Flow**:
1. **Debt Assessment**: `code-reviewer` → technical debt analysis
2. **Architecture Review**: `system-design-specialist` → refactoring strategy
3. **Impact Analysis**: Domain specialists → change impact assessment
4. **Incremental Planning**: `orchestration-coordinator` → phased approach
5. **Implementation**: Domain specialists → incremental changes
6. **Continuous Testing**: `test-engineer` → regression prevention
7. **Performance Monitoring**: `performance-optimization-specialist` → impact tracking

**Quality Gates**: Backward compatibility, performance maintained, test coverage preserved

### Operations Patterns

#### Incident Response Workflow
```bash
/workflow-patterns incident-response --severity <p0|p1|p2|p3> --domain <service|infrastructure|security>
```

**Pattern Flow**:
1. **Immediate Response**: `sre-incident-responder` → incident containment
2. **Impact Assessment**: `observability-engineer` → scope analysis
3. **Escalation**: Domain specialists → expert involvement
4. **Resolution**: Multi-agent coordination → fix implementation
5. **Verification**: `test-engineer` → resolution validation
6. **Communication**: `orchestration-coordinator` → stakeholder updates
7. **Post-Incident Review**: Multi-agent → lessons learned

**Quality Gates**: Incident contained, root cause identified, preventive measures planned

#### Performance Optimization Campaign
```bash
/workflow-patterns performance-optimization --target <latency|throughput|resource> --scope <application|database|infrastructure>
```

**Pattern Flow**:
1. **Baseline Measurement**: `observability-engineer` → current metrics
2. **Profiling Analysis**: `performance-optimization-specialist` → bottleneck identification
3. **Optimization Strategy**: Domain specialists → improvement plan
4. **Implementation**: Specialists → optimization implementation
5. **Load Testing**: `test-engineer` → performance validation
6. **Monitoring**: `observability-engineer` → continuous tracking
7. **Results Analysis**: `performance-optimization-specialist` → impact assessment

**Quality Gates**: Performance targets achieved, no functionality regressions, monitoring active

#### Security Audit & Remediation
```bash
/workflow-patterns security-audit --scope <application|infrastructure|compliance> --standard <owasp|sox|hipaa>
```

**Pattern Flow**:
1. **Threat Modeling**: `security-architect` → security assessment
2. **Vulnerability Scanning**: `security-architect` → automated scanning
3. **Code Security Review**: `code-reviewer` → manual analysis
4. **Infrastructure Review**: `aws-cloud-architect` → platform security
5. **Remediation Planning**: `security-architect` → fix prioritization
6. **Implementation**: Domain specialists → security fixes
7. **Validation**: `test-engineer` → security testing
8. **Compliance Check**: `security-architect` → standard compliance

**Quality Gates**: No critical vulnerabilities, compliance requirements met, security tests pass

### Integration Patterns

#### Third-Party Integration
```bash
/workflow-patterns third-party-integration --type <api|service|library> --complexity <simple|complex>
```

**Pattern Flow**:
1. **Integration Research**: `research-librarian` → vendor analysis
2. **Architecture Planning**: `system-design-specialist` → integration design
3. **Contract Definition**: `api-platform-engineer` → interface specification
4. **Security Assessment**: `security-architect` → vendor security review
5. **Implementation**: Domain specialists → integration development
6. **Testing Strategy**: `test-engineer` → integration testing
7. **Monitoring Setup**: `observability-engineer` → integration monitoring
8. **Documentation**: `research-librarian` → integration documentation

**Quality Gates**: Security approved, integration tested, monitoring configured, documentation complete

#### Data Migration Workflow
```bash
/workflow-patterns data-migration --source <type> --target <type> --volume <small|large|enterprise>
```

**Pattern Flow**:
1. **Data Analysis**: `database-architect` → source data assessment
2. **Migration Strategy**: `data-pipeline-engineer` → migration approach
3. **Schema Design**: `database-architect` → target schema
4. **Pipeline Development**: `data-pipeline-engineer` → migration pipeline
5. **Testing Strategy**: `test-engineer` → data validation testing
6. **Performance Optimization**: `performance-optimization-specialist` → migration optimization
7. **Deployment Planning**: `devops-automation-expert` → production migration
8. **Monitoring**: `observability-engineer` → migration monitoring

**Quality Gates**: Data integrity verified, performance acceptable, rollback plan tested

### Research & Innovation Patterns

#### Technology Evaluation
```bash
/workflow-patterns tech-evaluation --domain <frontend|backend|infrastructure|data> --decision-timeline <weeks|months>
```

**Pattern Flow**:
1. **Market Research**: `research-librarian` → technology landscape
2. **Technical Analysis**: Domain specialists → deep-dive evaluation
3. **Proof of Concept**: Domain specialists → prototype development
4. **Performance Testing**: `performance-optimization-specialist` → benchmark testing
5. **Security Assessment**: `security-architect` → security evaluation
6. **Integration Analysis**: `system-design-specialist` → ecosystem fit
7. **Recommendation**: `orchestration-coordinator` → decision synthesis

**Quality Gates**: Comprehensive analysis, POC validated, risks identified, recommendation clear

#### Architecture Evolution
```bash
/workflow-patterns architecture-evolution --target <microservices|serverless|event-driven> --timeline <gradual|aggressive>
```

**Pattern Flow**:
1. **Current State Analysis**: `system-design-specialist` → architecture assessment
2. **Future State Design**: `system-design-specialist` → target architecture
3. **Migration Strategy**: `system-design-specialist` → transformation plan
4. **Risk Assessment**: Domain specialists → risk analysis
5. **Incremental Implementation**: Domain specialists → phased migration
6. **Quality Assurance**: `test-engineer` + `code-reviewer` → continuous validation
7. **Performance Monitoring**: `performance-optimization-specialist` → impact tracking
8. **Deployment Coordination**: `devops-automation-expert` → production changes

**Quality Gates**: Architecture validated, migration plan approved, risk mitigation planned

## Pattern Customization Options

### Quality Level Customization
```yaml
mvp:
  test_coverage: 60%
  security_review: basic
  documentation: minimal
  performance_testing: smoke_tests

standard:
  test_coverage: 80%
  security_review: comprehensive
  documentation: complete
  performance_testing: load_tests

enterprise:
  test_coverage: 90%
  security_review: audit_level
  documentation: comprehensive_with_runbooks
  performance_testing: stress_and_chaos
```

### Timeline Customization
```yaml
urgent:
  quality_gates: reduced
  parallel_execution: maximum
  documentation: deferred

standard:
  quality_gates: full
  parallel_execution: optimized
  documentation: concurrent

thorough:
  quality_gates: comprehensive
  parallel_execution: conservative
  documentation: comprehensive_upfront
```

### Team Size Customization
```yaml
small_team:
  agents: 3-5
  roles: combined_responsibilities
  coordination: lightweight

standard_team:
  agents: 6-10
  roles: specialized
  coordination: structured

large_team:
  agents: 10+
  roles: highly_specialized
  coordination: full_orchestration
```

## Pattern Composition & Chaining

### Sequential Pattern Chaining
```bash
/workflow-patterns chain feature-pipeline bug-resolution performance-optimization
# Chains multiple patterns in sequence with shared context
```

### Parallel Pattern Execution
```bash
/workflow-patterns parallel security-audit performance-optimization
# Executes patterns in parallel where dependencies allow
```

### Conditional Pattern Branching
```bash
/workflow-patterns conditional feature-pipeline --on-failure bug-resolution --on-performance-issue performance-optimization
# Executes different patterns based on outcomes
```

## Pattern Analytics & Optimization

### Pattern Performance Metrics
- Completion time by pattern type
- Quality gate success rates
- Agent utilization efficiency
- Stakeholder satisfaction scores

### Pattern Evolution
- Success rate tracking and improvement
- Agent performance analysis within patterns
- Quality gate optimization
- Workflow bottleneck identification

### Custom Pattern Creation
```bash
/workflow-patterns create-custom <pattern_name> --template <base_pattern> --customizations <config_file>
# Creates organization-specific patterns based on proven templates
```

## Integration with Project Management

### Agile Integration
- Sprint planning with pattern estimation
- Story point mapping to pattern complexity
- Retrospective pattern effectiveness analysis

### DevOps Integration
- CI/CD pipeline pattern integration
- Automated pattern triggering based on events
- Pattern success metrics in deployment pipelines

### Compliance Integration
- Regulatory requirement mapping to patterns
- Audit trail generation for pattern execution
- Compliance gate integration within patterns

The workflow patterns system provides reusable, proven multi-agent coordination templates that can be adapted to specific organizational needs while maintaining quality standards and efficient agent collaboration.