# Analysis Phase Report

**Feature**: [Feature Name]
**Date**: [YYYY-MM-DD]
**Author**: [Name]
**Phase**: 1 - Analysis
**Threshold**: 95%

---

## Phase 1 Quality Gate Checklist

### 1. Requirements Completeness
**Status**: [ ] Complete / [ ] Incomplete

#### Business Objectives
> Clearly state the business problem being solved

- **Problem Statement**: [Describe the business problem]
- **Target Outcome**: [Measurable business outcome]
- **Success Metrics**: [KPIs and measurement criteria]
  - Metric 1: [e.g., "Reduce checkout time from 15s to <3s"]
  - Metric 2: [e.g., "Increase conversion rate by 10%"]

#### User Stories
> Document user stories in Given-When-Then format

**Story 1**: [User Story Title]
- **As a** [user type]
- **I want to** [action]
- **So that** [benefit]

**Acceptance Criteria**:
1. **Given** [initial context]
   **When** [action]
   **Then** [expected outcome]
2. [Additional criteria...]

**Story 2**: [Next user story]
- [Repeat format...]

#### Constraints
> Document technical, budget, timeline, and regulatory constraints

- **Technical**: [e.g., "Must support IE11", "API rate limit 100 req/s"]
- **Budget**: [e.g., "Infrastructure cost <$500/month"]
- **Timeline**: [e.g., "Launch by Q2 2024"]
- **Regulatory**: [e.g., "GDPR compliance required", "PCI-DSS Level 1"]

#### Stakeholder Sign-Off
- [ ] Product Manager: [Name] (Date: [YYYY-MM-DD])
- [ ] Engineering Lead: [Name] (Date: [YYYY-MM-DD])
- [ ] Design Lead: [Name] (Date: [YYYY-MM-DD])
- [ ] Security Team: [Name] (Date: [YYYY-MM-DD])

---

### 2. Domain Model Validation
**Status**: [ ] Validated / [ ] Not Validated

#### Bounded Contexts
> Identify and define bounded contexts with clear boundaries

**Context 1**: [Context Name]
- **Responsibility**: [What this context owns]
- **Core Domain**: [ ] Yes / [ ] No
- **Aggregates**: [List key aggregates]
  - [Aggregate Name] (root: [Entity])
- **Domain Events**: [Key events this context publishes]
  - [EventName]: [Description]

**Context 2**: [Next context]
- [Repeat format...]

#### Context Map
> Document relationships between bounded contexts

```
[Context A] ---(ACL)---> [Context B]
[Context C] ----(Shared Kernel)---- [Context D]
[Context E] ---(Customer/Supplier)---> [Context F]
```

**Relationship Types**:
- **ACL (Anti-Corruption Layer)**: [Context consuming through translation]
- **Shared Kernel**: [Shared models between contexts]
- **Customer/Supplier**: [Upstream/downstream relationship]
- **Open Host Service**: [Published API for consumers]

#### Aggregates & Entities
> Define core aggregates with their entities and value objects

**Aggregate**: [Aggregate Name]
- **Root Entity**: [Entity name]
- **Entities**: [Child entities]
- **Value Objects**: [Value objects within aggregate]
- **Invariants**: [Business rules that must always hold]
  1. [Invariant 1: e.g., "User can have max 5 active sessions"]
  2. [Invariant 2]

#### Ubiquitous Language
> Define domain-specific terminology

| Term | Definition | Context |
|------|------------|---------|
| [Term 1] | [Clear definition] | [Applicable context] |
| [Term 2] | [Definition] | [Context] |
| [Term 3] | [Definition] | [Context] |

#### Domain Expert Review
- [ ] Domain expert consulted: [Name]
- [ ] Model validated against real-world scenarios
- [ ] Ubiquitous language agreed upon
- [ ] Approval obtained: [Date: YYYY-MM-DD]

---

### 3. Architecture Documentation
**Status**: [ ] Documented / [ ] Not Documented

#### Architecture Decision Records (ADRs)

**ADR 001**: [Decision Title]
- **Status**: [Proposed / Accepted / Superseded]
- **Context**: [Why this decision is needed]
- **Decision**: [What was decided]
- **Rationale**: [Why this decision was made]
- **Alternatives Considered**:
  - [Alternative 1]: [Why rejected]
  - [Alternative 2]: [Why rejected]
- **Consequences**:
  - **Positive**: [Benefits]
  - **Negative**: [Drawbacks and mitigations]

**ADR 002**: [Next decision]
- [Repeat format...]

#### System Architecture

**Architecture Style**: [ ] Monolith / [ ] Microservices / [ ] Serverless / [ ] Hybrid

**Component Diagram**:
```
[Include C4 context or container diagram]
```

**Technology Stack**:
- **Frontend**: [Framework/library and version]
- **Backend**: [Language, framework, runtime]
- **Database**: [Type, version, and rationale]
- **Caching**: [Solution and strategy]
- **Message Queue**: [If applicable]
- **Infrastructure**: [Cloud provider, IaC tool]

#### Non-Functional Requirements (NFRs)

**Performance**:
- **Response Time**: P50: [<Xms], P95: [<Yms], P99: [<Zms]
- **Throughput**: [X req/s minimum], [Y req/s target]
- **Concurrent Users**: [X concurrent users]

**Availability**:
- **Uptime SLA**: [99.9% = 43min downtime/month]
- **Recovery Time Objective (RTO)**: [<X hours]
- **Recovery Point Objective (RPO)**: [<Y minutes]

**Security**:
- **Authentication**: [Mechanism: JWT, OAuth2, etc.]
- **Authorization**: [RBAC, ABAC, etc.]
- **Data Encryption**: [At rest: AES-256, In transit: TLS 1.3]
- **Compliance**: [Standards: PCI-DSS, HIPAA, GDPR, SOC2]

**Scalability**:
- **Horizontal Scaling**: [Strategy]
- **Database Scaling**: [Read replicas, sharding, etc.]
- **Caching Strategy**: [Layer, TTL, invalidation]

**Observability**:
- **Logging**: [Aggregation tool, retention policy]
- **Metrics**: [Collection tool, key metrics]
- **Tracing**: [Distributed tracing tool]
- **Alerting**: [On-call integration, escalation policy]

#### Integration Patterns
> Document how this feature integrates with existing systems

- **System A**: [Integration pattern: REST, Event-driven, etc.]
- **System B**: [Integration pattern and data flow]
- **External APIs**: [Third-party dependencies and SLAs]

---

### 4. Risk Identification
**Status**: [ ] Identified / [ ] Not Identified

#### Risk Register

**Risk 1**: [Risk Title]
- **Description**: [Detailed risk description]
- **Likelihood**: [ ] Low (5%) / [ ] Medium (15%) / [ ] High (30%)
- **Impact**: [ ] Low / [ ] Medium / [ ] High / [ ] Critical
- **Priority**: [Low / Medium / High / Critical]
- **Mitigation Strategy**:
  1. [Mitigation action 1]
  2. [Mitigation action 2]
- **Contingency Plan**: [Fallback if risk materializes]
- **Owner**: [Responsible person]

**Risk 2**: [Next risk]
- [Repeat format...]

#### Risk Matrix
```
           │ Low    │ Medium │ High   │ Critical
───────────┼────────┼────────┼────────┼──────────
Low (5%)   │   1    │   2    │   3    │    4
Medium (15%)│   5    │   6    │   7    │    8
High (30%) │   9    │  10    │  11    │   12

[Place risk numbers in appropriate cells]
```

#### Assumptions
> Explicit assumptions that must hold for success

1. [Assumption 1: e.g., "API uptime >99.5%"]
2. [Assumption 2: e.g., "User devices support WebAuthn"]
3. [Assumption 3]

#### Dependencies
> External dependencies and their risk profiles

| Dependency | Type | SLA | Risk if Unavailable |
|------------|------|-----|---------------------|
| [Service A] | [Internal/External] | [99.9%] | [Impact description] |
| [API B] | [External] | [99.5%] | [Impact and mitigation] |

#### Technical Debt
> Existing technical debt that impacts this feature

- **Debt Item 1**: [Description and impact]
  - **Mitigation**: [How to work around or pay down]
- **Debt Item 2**: [Description]

---

## Quality Gate Score

### Scoring
- **Requirements Complete**: [ ] Yes (25 points) / [ ] No (0 points)
- **Domain Model Validated**: [ ] Yes (25 points) / [ ] No (0 points)
- **Architecture Documented**: [ ] Yes (25 points) / [ ] No (0 points)
- **Risks Identified**: [ ] Yes (25 points) / [ ] No (0 points)

**Total Score**: [X / 100]
**Threshold**: 95%
**Status**: [ ] PASS ✅ / [ ] FAIL ❌

### Gate Decision
- [ ] **PASS** - Proceed to Phase 2 (Implementation)
- [ ] **FAIL** - Address gaps before proceeding
- [ ] **WAIVER** - Approved by [Name] on [Date] (Reason: [Justification])

---

## Artifacts Generated

- [ ] `requirements.md` - Business requirements and user stories
- [ ] `domain-model.md` - Bounded contexts and aggregates
- [ ] `architecture/` - ADRs and system diagrams
- [ ] `risk-register.md` - Risk assessment and mitigation
- [ ] `ubiquitous-language.md` - Domain terminology glossary
- [ ] `.quality-gates.json` - Quality gate configuration

---

## Next Steps

1. [ ] Review analysis report with stakeholders
2. [ ] Obtain sign-offs from all required parties
3. [ ] Update `.quality-gates.json` with phase1 results
4. [ ] Schedule Phase 2 (Implementation) kickoff
5. [ ] Communicate analysis outcomes to development team

---

## Notes & Observations

[Add any additional context, concerns, or observations]

---

**Reviewed By**: [Name, Title] (Date: [YYYY-MM-DD])
**Approved By**: [Name, Title] (Date: [YYYY-MM-DD])
