---
description: Dynamic agent team composition for different project types
args: <project-type> [--scope small|medium|large] [--priority speed|quality|security]
tools: Task, TodoWrite, Read
model: opus
---

# /team-formation - Dynamic Agent Team Composition

Intelligently forms and coordinates agent teams based on project requirements, complexity assessment, and domain expertise mapping. Optimizes team composition for different project types and development methodologies.

## Usage

```bash
/team-formation <project_type> [options]
```

## Project Types & Team Formations

### Web Application Teams

#### Full-Stack Web Application
```bash
/team-formation web-app --stack modern --complexity high
```
**Team Composition**:
- **Lead**: `system-design-specialist` (architecture)
- **API**: `api-platform-engineer` (contracts & governance)
- **Frontend**: `frontend-expert` (React/Vue/Angular)
- **Backend**: `backend-architect` + `typescript-architect`/`python-expert`
- **Data**: `database-architect` (schema & optimization)
- **Platform**: `aws-cloud-architect` + `devops-automation-expert`
- **Quality**: `test-engineer` + `code-reviewer`
- **Security**: `security-architect` (auth & compliance)
- **Observability**: `observability-engineer` + `sre-incident-responder`

#### API-First Development
```bash
/team-formation api-first --microservices
```
**Team Composition**:
- **Lead**: `api-platform-engineer` (API governance)
- **Design**: `system-design-specialist` (service boundaries)
- **Implementation**: `backend-architect` + language specialists
- **Data**: `database-architect` (service data stores)
- **Gateway**: `aws-cloud-architect` (API gateway & routing)
- **Testing**: `test-engineer` (contract testing)
- **Documentation**: `research-librarian` (API documentation)

#### Frontend-Focused Application
```bash
/team-formation frontend-app --spa --pwa
```
**Team Composition**:
- **Lead**: `frontend-expert` (UI architecture)
- **Backend**: `api-platform-engineer` (BFF pattern)
- **Performance**: `performance-optimization-specialist` (client optimization)
- **Testing**: `test-engineer` (E2E & visual testing)
- **Deployment**: `devops-automation-expert` (CDN & static hosting)

### Mobile Application Teams

#### Cross-Platform Mobile
```bash
/team-formation mobile-app --cross-platform --native-features
```
**Team Composition**:
- **Lead**: `mobile-specialist` (cross-platform strategy)
- **Backend**: `backend-architect` (mobile-optimized APIs)
- **API**: `api-platform-engineer` (mobile-first contracts)
- **Performance**: `performance-optimization-specialist` (mobile optimization)
- **Testing**: `test-engineer` (device testing strategies)
- **Platform**: `devops-automation-expert` (mobile CI/CD)

#### Native Mobile Development
```bash
/team-formation mobile-native --ios --android --performance-critical
```
**Team Composition**:
- **Lead**: `mobile-specialist` (native development)
- **iOS**: `mobile-specialist` (Swift/SwiftUI focus)
- **Android**: `mobile-specialist` (Kotlin/Compose focus)
- **Backend**: `backend-architect` (mobile backend services)
- **Performance**: `performance-optimization-specialist` (native optimization)
- **Testing**: `test-engineer` (native testing frameworks)

### Data & Analytics Teams

#### Data Platform Development
```bash
/team-formation data-platform --streaming --analytics --ml
```
**Team Composition**:
- **Lead**: `data-pipeline-engineer` (data architecture)
- **ML**: `machine-learning-engineer` (ML pipelines)
- **Backend**: `backend-architect` (data services)
- **Infrastructure**: `aws-cloud-architect` (data infrastructure)
- **Performance**: `performance-optimization-specialist` (query optimization)
- **Quality**: `test-engineer` (data quality testing)

#### Machine Learning System
```bash
/team-formation ml-system --production --monitoring --automation
```
**Team Composition**:
- **Lead**: `machine-learning-engineer` (ML architecture)
- **Data**: `data-pipeline-engineer` (feature engineering)
- **Backend**: `python-expert` (ML service implementation)
- **Infrastructure**: `aws-cloud-architect` (ML infrastructure)
- **Monitoring**: `observability-engineer` (model monitoring)
- **Quality**: `test-engineer` (ML testing strategies)

### Infrastructure & Platform Teams

#### Cloud Infrastructure
```bash
/team-formation cloud-infrastructure --aws --kubernetes --observability
```
**Team Composition**:
- **Lead**: `aws-cloud-architect` (cloud strategy)
- **Automation**: `devops-automation-expert` (IaC & CI/CD)
- **SRE**: `sre-incident-responder` (reliability engineering)
- **Monitoring**: `observability-engineer` (platform observability)
- **Security**: `security-architect` (infrastructure security)
- **Performance**: `performance-optimization-specialist` (infrastructure optimization)

#### DevOps & Platform Engineering
```bash
/team-formation devops-platform --automation --monitoring --security
```
**Team Composition**:
- **Lead**: `devops-automation-expert` (platform automation)
- **Cloud**: `aws-cloud-architect` (cloud platform)
- **SRE**: `sre-incident-responder` (platform reliability)
- **Security**: `security-architect` (platform security)
- **Observability**: `observability-engineer` (platform monitoring)

### Quality & Security Teams

#### Quality Assurance Team
```bash
/team-formation qa-team --automation --performance --security
```
**Team Composition**:
- **Lead**: `test-engineer` (test strategy)
- **Code Quality**: `code-reviewer` (quality gates)
- **Performance**: `performance-optimization-specialist` (performance testing)
- **Security**: `security-architect` (security testing)
- **Automation**: `devops-automation-expert` (test automation)

#### Security & Compliance Team
```bash
/team-formation security-team --compliance --audit --monitoring
```
**Team Composition**:
- **Lead**: `security-architect` (security strategy)
- **Code Security**: `code-reviewer` (security code review)
- **Infrastructure**: `aws-cloud-architect` (infrastructure security)
- **Monitoring**: `observability-engineer` (security monitoring)
- **Incident Response**: `sre-incident-responder` (security incidents)
- **Research**: `research-librarian` (security standards)

## Team Formation Options

### Complexity-Based Scaling
```yaml
simple:
  team_size: 3-4 agents
  quality_gates: basic
  coordination: minimal

moderate:
  team_size: 5-7 agents
  quality_gates: standard
  coordination: structured

complex:
  team_size: 8-12 agents
  quality_gates: comprehensive
  coordination: orchestrated

enterprise:
  team_size: 10-15 agents
  quality_gates: enterprise
  coordination: full_governance
```

### Development Methodology Integration
```yaml
agile:
  iteration_length: 2_weeks
  review_cycles: weekly
  agent_rotation: flexible

waterfall:
  phase_gates: strict
  documentation: comprehensive
  agent_handoffs: formal

devops:
  automation_focus: high
  deployment_frequency: continuous
  monitoring_emphasis: real_time
```

## Team Coordination Patterns

### Leadership & Coordination Roles
- **Team Lead**: Primary architect for domain (system-design-specialist, api-platform-engineer, etc.)
- **Quality Lead**: `code-reviewer` or `test-engineer` for quality coordination
- **Operations Lead**: `devops-automation-expert` or `sre-incident-responder`
- **Coordinator**: `orchestration-coordinator` for complex multi-team projects

### Communication Protocols
- **Daily Standups**: Progress synchronization across agents
- **Sprint Planning**: Work distribution and dependency mapping
- **Code Reviews**: Multi-agent review processes
- **Retrospectives**: Team performance and process improvement

### Handoff Procedures
```yaml
design_to_implementation:
  artifacts: [architecture_docs, api_specs, data_models]
  quality_gates: [design_review, stakeholder_approval]
  next_agents: [implementation_team]

implementation_to_testing:
  artifacts: [code, unit_tests, documentation]
  quality_gates: [code_review, coverage_check]
  next_agents: [test_engineer, security_architect]

testing_to_deployment:
  artifacts: [test_results, performance_metrics]
  quality_gates: [all_tests_pass, security_clear]
  next_agents: [devops_automation_expert, observability_engineer]
```

## Team Performance Optimization

### Agent Load Balancing
- Monitor individual agent workload and performance
- Redistribute tasks based on expertise and availability
- Prevent agent bottlenecks through intelligent routing

### Skill Gap Analysis
- Identify missing expertise for specific project requirements
- Recommend additional agent inclusion or external consultation
- Suggest agent specialization development paths

### Team Efficiency Metrics
- Task completion velocity across agents
- Quality gate pass rates by team composition
- Inter-agent collaboration effectiveness
- Stakeholder satisfaction by team type

## Dynamic Team Adaptation

### Mid-Project Team Adjustments
```bash
/team-formation adjust --add-specialist <domain>
/team-formation adjust --scale-up performance
/team-formation adjust --security-focus
```

### Crisis Response Teams
```bash
/team-formation incident-response --severity critical
# Forms: sre-incident-responder, security-architect, system-design-specialist,
#        observability-engineer, devops-automation-expert
```

### Research & Innovation Teams
```bash
/team-formation research-team --domain <technology>
# Forms: research-librarian, relevant domain specialists,
#        system-design-specialist for feasibility
```

## Team Formation Examples

### Startup MVP Team (Lean)
```bash
/team-formation startup-mvp --lean --fast-iteration
```
**3-Agent Team**: `full-stack-architect`, `test-engineer`, `devops-automation-expert`

### Enterprise Digital Transformation
```bash
/team-formation enterprise-transformation --migration --compliance --scalability
```
**12-Agent Team**: Full platform team with security, compliance, and migration specialists

### Open Source Project
```bash
/team-formation open-source --community --documentation --maintenance
```
**6-Agent Team**: Focus on code quality, documentation, community engagement, and long-term maintenance

The team formation system ensures optimal agent collaboration through intelligent team composition based on project requirements, complexity, and organizational constraints.