# Phase 1: Analysis Quality Gate

**Threshold**: 95%
**Purpose**: Ensure complete understanding of the problem before implementation
**Duration**: 1-4 hours for typical features

## Why 95%?

Analysis is the foundation. A 5% miss in requirements translates to 30-50% rework during implementation. The cost of fixing a requirements defect increases exponentially:

- **Analysis phase**: 1x effort to fix
- **Implementation phase**: 10x effort to fix
- **Production**: 100x effort to fix

We enforce a 95% threshold to catch critical gaps early while remaining pragmatic about unknowable edge cases.

## Gate Criteria

### 1. Requirements Completeness (Boolean)
**Status**: `requirements_complete: true|false`

**Validation Checklist**:
- [ ] Business objectives clearly stated with measurable KPIs
- [ ] User stories written in Given-When-Then format
- [ ] Acceptance criteria defined for each story
- [ ] Success metrics specified (e.g., "reduce checkout time by 30%")
- [ ] Constraints documented (budget, timeline, technical limitations)
- [ ] Stakeholder sign-off obtained

**Artifacts Required**:
- `requirements.md` with business context
- `user-stories.md` with acceptance criteria
- Stakeholder approval (email, meeting notes, or sign-off document)

**Common Failures**:
- Vague requirements: "improve user experience" (no measurable target)
- Missing constraints: undocumented technical debt or budget limits
- Stakeholder misalignment: different expectations between teams

**How to Pass**:
```markdown
# requirements.md
## Business Objective
Reduce user authentication time from 15s to <3s to improve conversion rates.

## User Stories
**Story 1**: As a returning user, I want to authenticate with biometrics so that I can access my account quickly.
- **Given** I am a returning user with biometrics enabled
- **When** I open the app
- **Then** I should be authenticated within 2 seconds

## Acceptance Criteria
1. Authentication completes in <3s for 95th percentile
2. Fallback to password available within 1 tap
3. Biometric data never leaves the device

## Success Metrics
- P95 auth time: <3s (current: 15s)
- Auth success rate: >99.5%
- User adoption: >60% within 30 days
```

### 2. Domain Model Validated (Boolean)
**Status**: `domain_model_validated: true|false`

**Validation Checklist**:
- [ ] Bounded contexts identified with clear boundaries
- [ ] Aggregates defined with root entities
- [ ] Domain invariants documented
- [ ] Ubiquitous language glossary created
- [ ] Context map showing relationships (ACL, OHS, Shared Kernel, etc.)
- [ ] Domain expert reviewed and approved model

**Artifacts Required**:
- `domain-model.md` with bounded contexts and aggregates
- `ubiquitous-language.md` glossary
- Context map diagram (C4, UML, or Miro board screenshot)
- Domain expert sign-off

**Common Failures**:
- Anemic domain model: all logic in services, entities as data bags
- Missing boundaries: one giant "Application" context
- Unclear invariants: no documented business rules

**How to Pass**:
```markdown
# domain-model.md
## Bounded Contexts

### Authentication Context
**Responsibility**: Manage user identity and access control
**Aggregates**:
- **User** (root): identity, credentials, permissions
- **Session**: active authentication state, expiration

**Invariants**:
- A user can have max 5 active sessions
- Session expires after 24 hours of inactivity
- Password must meet complexity requirements

### Payment Context
**Responsibility**: Process financial transactions
**Relationship to Auth**: Downstream consumer via ACL (Anti-Corruption Layer)

## Ubiquitous Language
- **User**: Authenticated entity with identity
- **Session**: Time-bound authentication state
- **Credential**: Authentication proof (password, biometric hash)
```

### 3. Architecture Documented (Boolean)
**Status**: `architecture_documented: true|false`

**Validation Checklist**:
- [ ] Architecture Decision Records (ADRs) for major choices
- [ ] Component diagram showing system structure
- [ ] Data flow diagrams for critical paths
- [ ] Technology stack justified with rationale
- [ ] NFRs specified (performance, availability, security)
- [ ] Integration patterns documented

**Artifacts Required**:
- ADRs in `architecture/decisions/` directory
- C4 context and container diagrams
- `nfr-requirements.md` with SLOs

**Common Failures**:
- No rationale: "We chose PostgreSQL" (why not MySQL, DynamoDB?)
- Missing NFRs: no performance targets, availability requirements
- Undocumented assumptions: "Assumes single region deployment"

**How to Pass**:
```markdown
# architecture/decisions/001-database-selection.md
# ADR 001: Use PostgreSQL for User Data

## Context
Authentication system requires ACID transactions for user registration and session management.

## Decision
Use PostgreSQL 15 with read replicas for authentication context.

## Rationale
- **ACID guarantees**: Prevent duplicate accounts, ensure session consistency
- **JSON support**: Flexible user metadata without schema migrations
- **Proven scale**: Handles 10K req/s with proper indexing
- **Team expertise**: Current team experienced with PostgreSQL

## Alternatives Considered
- **MySQL**: Lacks advanced JSON querying
- **DynamoDB**: Eventually consistent, complex transaction handling
- **MongoDB**: No native ACID for multi-document operations

## Consequences
- **Positive**: Strong consistency, team familiarity
- **Negative**: Vertical scaling limits (mitigated by read replicas)

## NFRs
- **Availability**: 99.9% uptime (43 min downtime/month)
- **Performance**: P95 query latency <50ms
- **Scalability**: Support 100K concurrent sessions
```

### 4. Risks Identified (Boolean)
**Status**: `risks_identified: true|false`

**Validation Checklist**:
- [ ] Technical risks documented with likelihood and impact
- [ ] Mitigation strategies defined for high-priority risks
- [ ] Assumptions explicitly stated
- [ ] Dependencies on external systems identified
- [ ] Fallback plans for critical failure modes

**Artifacts Required**:
- `risk-register.md` with risk matrix
- Mitigation plans for high-impact risks

**Common Failures**:
- Generic risks: "System might fail" (not actionable)
- No mitigation: risks listed without response strategy
- Blind spots: external dependencies not considered

**How to Pass**:
```markdown
# risk-register.md
## High-Priority Risks

### Risk 1: Biometric API Availability
- **Likelihood**: Medium (15% chance of API downtime)
- **Impact**: High (blocks all biometric auth)
- **Mitigation**:
  - Implement fallback to password within 500ms
  - Cache biometric verification for 5 minutes
  - Monitor API health with 30s intervals
- **Contingency**: Gracefully degrade to password-only mode

### Risk 2: Migration Data Loss
- **Likelihood**: Low (5% chance during migration)
- **Impact**: Critical (user data corruption)
- **Mitigation**:
  - Blue-green deployment with rollback plan
  - Dry-run migration on staging with 100K users
  - Real-time data validation during migration
- **Contingency**: Automated rollback within 60 seconds

## Assumptions
1. Biometric API uptime >99.5%
2. User device supports WebAuthn
3. Network latency <200ms for 95% of users

## Dependencies
- External: Biometric vendor API (SLA: 99.9%)
- Internal: User service API (SLA: 99.95%)
```

## Scoring Calculation

**Formula**: `(requirements_complete + domain_model_validated + architecture_documented + risks_identified) / 4 * 100`

**Example**:
```json
{
  "requirements_complete": true,
  "domain_model_validated": true,
  "architecture_documented": true,
  "risks_identified": false
}
```
**Score**: `(1 + 1 + 1 + 0) / 4 * 100 = 75%` ❌ FAIL (threshold: 95%)

**Passing Example**:
```json
{
  "requirements_complete": true,
  "domain_model_validated": true,
  "architecture_documented": true,
  "risks_identified": true
}
```
**Score**: `(1 + 1 + 1 + 1) / 4 * 100 = 100%` ✅ PASS

## Gate Validation

### Automated Validation

```bash
# Run quality gate checker
python3 scripts/quality-gate-checker.py \
  --phase analysis \
  --config ./features/user-auth/.quality-gates.json \
  --exit-code

# Output:
# ✅ Requirements complete: PASS
# ✅ Domain model validated: PASS
# ✅ Architecture documented: PASS
# ✅ Risks identified: PASS
#
# Phase 1 (Analysis) Score: 100% ✅
# Threshold: 95%
# Status: PASSED
```

### Manual Validation

Use the analysis report template:

```bash
# Generate analysis report
cp templates/quality-gates/ANALYSIS_REPORT.template.md features/user-auth/analysis-report.md

# Fill out checklist and submit for review
# Ensure all checkboxes are marked before proceeding to Phase 2
```

## Common Failure Scenarios

### Scenario 1: Incomplete Requirements
**Symptom**: Vague acceptance criteria, missing success metrics
**Fix**: Schedule requirements workshop with stakeholders
**Timeline**: 2-4 hours

### Scenario 2: Weak Domain Model
**Symptom**: No clear aggregates, missing business rules
**Fix**: Conduct event storming session with domain expert
**Timeline**: 3-6 hours

### Scenario 3: Missing ADRs
**Symptom**: Technology choices without rationale
**Fix**: Document decisions retroactively, capture assumptions
**Timeline**: 1-2 hours

### Scenario 4: Unidentified Risks
**Symptom**: No risk register, external dependencies ignored
**Fix**: Risk assessment workshop, dependency mapping
**Timeline**: 1-3 hours

## Integration with Workflow

When using `/workflow-feature-development`, Phase 1 executes automatically:

```bash
# Analysis phase with automatic gate validation
/workflow-feature-development ./features/user-auth --stage=requirements

# Output includes gate status:
# 📋 Phase 1: Analysis
# ✅ Requirements: COMPLETE
# ✅ Domain Model: VALIDATED
# ✅ Architecture: DOCUMENTED
# ✅ Risks: IDENTIFIED
#
# 🎯 Gate Score: 100% (threshold: 95%)
# ✅ PASSED - Proceeding to Phase 2
```

## Waiver Process

In rare cases, gates may be waived (e.g., prototype, spike work):

```json
{
  "waiver": {
    "approved_by": "tech-lead@company.com",
    "reason": "Time-boxed prototype for feasibility study",
    "expiration": "2024-12-31",
    "conditions": "Must complete analysis before production deployment"
  }
}
```

**Waiver criteria**:
- Documented justification
- Explicit approval from technical lead or architect
- Time-bound with mandatory completion date
- No waivers for production deployments

## Next Steps

After passing Phase 1:
1. **Update TodoWrite**: Mark analysis phase complete
2. **Proceed to Phase 2**: Implementation with validated design
3. **Archive artifacts**: Store analysis deliverables in feature directory
4. **Communicate progress**: Update stakeholders with analysis outcomes

**Related Documentation**:
- [Phase 2: Implementation Gate](phase-2-implementation.md)
- [Quality Gates Overview](README.md)
- [Analysis Report Template](../../templates/quality-gates/ANALYSIS_REPORT.template.md)
