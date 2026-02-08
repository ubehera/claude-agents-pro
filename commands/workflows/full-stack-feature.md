---
description: End-to-end feature development orchestrating system design, API development, frontend implementation, database modeling, and deployment with integrated testing and quality gates
args: <feature-name> [--architecture microservices|monolith] [--quality mvp|standard|enterprise]
tools: Task, TodoWrite, Read, Write, MultiEdit
model: opus
---

# /full-stack-feature - Complete Feature Development Pipeline

Orchestrates comprehensive full-stack feature development from requirements analysis through production deployment, coordinating specialized agents across all tiers with integrated quality gates and progress tracking.

## Usage

```bash
/full-stack-feature <feature_description> [options]
```

## Options

- `--api-first` - Start with API design and contracts (default)
- `--frontend-first` - Begin with UI/UX mockups and user flows
- `--backend-first` - Start with data modeling and business logic
- `--mvp` - Minimum viable product scope with reduced quality gates
- `--enterprise` - Full enterprise-grade with comprehensive quality gates
- `--stack <tech>` - Technology stack preference (react/vue/angular, node/python/java)
- `--parallel` - Enable parallel development where possible
- `--iterative` - Use iterative development with frequent reviews

## Feature Development Stages

### Stage 1: Requirements & System Design
**Duration**: 15-20 minutes

#### Requirements Analysis
**Agent**: `research-librarian`
- Feature requirements research and analysis
- User story and acceptance criteria definition
- Market standards and best practices research
- Technical constraint identification

#### System Architecture
**Agent**: `system-design-specialist`
- High-level system architecture design
- Service boundary definition
- Data flow and integration planning
- Non-functional requirements specification

**Quality Gate**: Requirements clarity >90%, architecture review approved

### Stage 2: API & Data Design
**Duration**: 20-25 minutes

#### API Specification
**Agent**: `api-platform-engineer`
- RESTful API design and OpenAPI specification
- Authentication/authorization strategy
- Rate limiting and caching considerations
- API versioning and migration strategy

#### Database Design
**Agent**: `database-architect`
- Logical data model design
- Database schema optimization
- Index strategy and query optimization
- Data migration planning

**Quality Gate**: API contracts validated, data model review approved

### Stage 3: Parallel Implementation
**Duration**: 45-60 minutes (parallel execution)

#### Backend Development
**Agent**: `backend-architect` + language-specific agent
- Business logic implementation
- API endpoint development
- Database integration and ORM setup
- Authentication and authorization implementation

#### Frontend Development
**Agent**: `frontend-expert`
- UI component development
- State management implementation
- API integration and error handling
- Responsive design and accessibility

#### Infrastructure Preparation
**Agent**: `aws-cloud-architect` + `devops-automation-expert`
- Environment setup and configuration
- CI/CD pipeline preparation
- Monitoring and logging infrastructure
- Security configurations

**Quality Gate**: Core functionality complete, integration ready

### Stage 4: Integration & Testing
**Duration**: 25-30 minutes

#### Comprehensive Testing
**Agent**: `test-engineer`
- Unit test coverage validation
- Integration test development
- End-to-end test automation
- Performance test implementation

#### Security Validation
**Agent**: `security-architect`
- Security threat assessment
- Vulnerability scanning and remediation
- Authentication/authorization testing
- Data protection compliance validation

**Quality Gate**: Test coverage >85%, security scan passed

### Stage 5: Performance & Quality Optimization
**Duration**: 20-25 minutes

#### Performance Optimization
**Agent**: `performance-optimization-specialist`
- Performance profiling and bottleneck identification
- Database query optimization
- Frontend loading optimization
- Caching strategy implementation

#### Code Quality Review
**Agent**: `code-reviewer`
- Code quality and style compliance
- Architecture pattern adherence
- Documentation completeness
- Technical debt assessment

**Quality Gate**: Performance targets met, code quality >8.5/10

### Stage 6: Deployment & Monitoring
**Duration**: 15-20 minutes

#### Production Deployment
**Agent**: `devops-automation-expert`
- Production environment deployment
- Database migration execution
- Configuration management
- Blue-green or canary deployment

#### Observability Setup
**Agent**: `observability-engineer`
- Application metrics and monitoring
- Log aggregation and analysis
- Alert configuration and thresholds
- Performance baseline establishment

**Quality Gate**: Successful deployment, monitoring active, health checks passing

## Example Workflows

### E-Commerce Product Catalog Feature
```bash
/full-stack-feature "product catalog with search and filtering" --api-first --stack react-node --parallel
```

**Coordinated Agent Workflow**:
1. `research-librarian`: E-commerce catalog patterns, search standards
2. `system-design-specialist`: Microservices architecture, search service design
3. `api-platform-engineer`: Product API, search API, filtering endpoints
4. `database-architect`: Product schema, search indexes, category hierarchy
5. **Parallel Implementation**:
   - `backend-architect` + `typescript-architect`: Node.js service implementation
   - `frontend-expert`: React components, search UI, filtering interface
   - `aws-cloud-architect`: Elasticsearch setup, caching layer
6. `test-engineer`: E2E search scenarios, performance tests
7. `security-architect`: Input validation, SQL injection prevention
8. `performance-optimization-specialist`: Search performance, pagination
9. `devops-automation-expert`: Production deployment with search service

### User Authentication System
```bash
/full-stack-feature "multi-factor authentication system" --enterprise --backend-first
```

**Security-Focused Workflow**:
1. `research-librarian`: MFA standards, security best practices
2. `security-architect`: Threat model, security requirements
3. `system-design-specialist`: Authentication service architecture
4. `database-architect`: User credentials, session management
5. `api-platform-engineer`: Auth API, token management
6. `backend-architect`: Authentication service, MFA implementation
7. `frontend-expert`: Login UI, MFA verification components
8. `test-engineer`: Security testing, authentication scenarios
9. `observability-engineer`: Security monitoring, audit logging

### Real-Time Chat Feature
```bash
/full-stack-feature "real-time messaging with presence indicators" --iterative --stack react-python
```

**Real-Time Workflow**:
1. `system-design-specialist`: WebSocket architecture, message queuing
2. `api-platform-engineer`: Real-time API design, WebSocket protocols
3. `database-architect`: Message storage, user presence tracking
4. `backend-architect` + `python-expert`: Python WebSocket server
5. `frontend-expert`: React real-time components, WebSocket integration
6. `performance-optimization-specialist`: Message delivery optimization
7. `observability-engineer`: Real-time metrics, connection monitoring

## Quality Gate Configuration

### MVP Quality Gates (Reduced)
```yaml
requirements_clarity: >80%
test_coverage: >70%
security_scan: basic
performance_baseline: established
documentation: minimal
```

### Standard Quality Gates
```yaml
requirements_clarity: >90%
test_coverage: >85%
security_scan: comprehensive
performance_targets: met
code_quality: >8.0
documentation: complete
```

### Enterprise Quality Gates (Comprehensive)
```yaml
requirements_clarity: >95%
test_coverage: >90%
security_scan: full_audit
performance_targets: strict
code_quality: >8.5
documentation: comprehensive
compliance: validated
accessibility: wcag_compliant
```

## Technology Stack Integration

### React + Node.js Stack
```yaml
agents:
  frontend: frontend-expert
  backend: typescript-architect
  database: database-architect
  deployment: devops-automation-expert
tools:
  frontend: React, TypeScript, Jest, Cypress
  backend: Node.js, Express, TypeScript, Jest
  database: PostgreSQL, Redis
  deployment: Docker, AWS, GitHub Actions
```

### Vue + Python Stack
```yaml
agents:
  frontend: frontend-expert
  backend: python-expert
  database: database-architect
  deployment: aws-cloud-architect
tools:
  frontend: Vue.js, TypeScript, Vitest
  backend: FastAPI, Python, pytest
  database: PostgreSQL, Redis
  deployment: AWS Lambda, CloudFormation
```

## Progress Tracking & Communication

### TodoWrite Integration
- Automatic task creation for each development stage
- Progress tracking across parallel workstreams
- Dependency management between agents
- Quality gate checkpoint tracking

### Inter-Agent Communication
- Shared artifact repository for specifications
- Context passing between development stages
- Design decision documentation and rationale
- Issue escalation and resolution tracking

### Stakeholder Updates
- Progress dashboard with stage completion status
- Quality gate status and metrics tracking
- Risk identification and mitigation status
- Timeline adjustment and impact analysis

## Error Recovery & Risk Mitigation

### Common Issue Patterns
- **API Contract Mismatch**: Automatic validation and reconciliation
- **Database Migration Conflicts**: Rollback procedures and resolution
- **Integration Test Failures**: Automated debugging and root cause analysis
- **Performance Degradation**: Automatic optimization recommendations

### Escalation Procedures
- Critical issues → `orchestration-coordinator` for multi-agent resolution
- Security concerns → `security-architect` immediate assessment
- Performance issues → `performance-optimization-specialist` priority review
- Infrastructure problems → `sre-incident-responder` rapid response

The full-stack feature development command provides a comprehensive, orchestrated approach to building complete features with quality assurance and automated coordination across all development tiers.