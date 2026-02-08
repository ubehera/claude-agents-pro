---
name: spec-driven-development
description: Load when implementing features from specifications, PRDs, or design documents with traceability
trigger_keywords: [spec driven, specification, PRD, product requirements, design doc, feature spec, acceptance criteria, requirements traceability, given when then]
---

# Spec-Driven Development Skill

Systematic approach to implementing features from specifications with full traceability from requirements to tests.

## Overview

Spec-driven development ensures every line of code traces back to a requirement and every requirement has a corresponding test. This eliminates scope creep, missed requirements, and untested features.

**When to Use**:
- Implementing features from PRDs or design documents
- Projects with compliance/audit requirements
- Complex features spanning multiple components
- Teams requiring traceability for certification

## Workflow

```
1. Parse Spec    → Extract requirements, acceptance criteria
2. Decompose     → Break into implementable tasks
3. Contract      → Define API/data contracts from spec
4. Test First    → Write tests from acceptance criteria (Given-When-Then)
5. Implement     → Code against contracts, passing tests
6. Trace         → Verify every requirement has tests and code
7. Review        → Validate against original spec
```

## Step 1: Parse Specification

```markdown
## Requirement Extraction Template

### Feature: [Feature Name]
Source: [PRD link or document section]

### Functional Requirements
| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| FR-001 | Users can create accounts with email/password | Must | Given valid email and password ≥8 chars, When user submits registration, Then account is created and verification email sent |
| FR-002 | Users can log in with credentials | Must | Given valid credentials, When user submits login, Then JWT token returned with 1h expiry |
| FR-003 | Users can reset forgotten passwords | Should | Given registered email, When user requests reset, Then reset link sent valid for 24h |

### Non-Functional Requirements
| ID | Requirement | Metric | Target |
|----|------------|--------|--------|
| NFR-001 | Login response time | P95 latency | <200ms |
| NFR-002 | Account creation rate limit | Requests/IP/hour | 10 |
| NFR-003 | Password storage | Algorithm | Argon2id |

### Out of Scope
- Social login (OAuth2) — deferred to Phase 2
- Multi-factor authentication — separate spec
```

## Step 2: Decompose into Tasks

```markdown
## Implementation Tasks (from spec)

### API Layer (from FR-001, FR-002, FR-003)
- [ ] POST /api/auth/register → FR-001
- [ ] POST /api/auth/login → FR-002
- [ ] POST /api/auth/forgot-password → FR-003
- [ ] POST /api/auth/reset-password → FR-003

### Business Logic
- [ ] Password hashing with Argon2id → NFR-003
- [ ] JWT generation with 1h expiry → FR-002
- [ ] Rate limiting (10/IP/hour) → NFR-002
- [ ] Email verification flow → FR-001

### Data Layer
- [ ] User table migration
- [ ] Password reset token table
- [ ] Index on email (unique)

### Tests (from acceptance criteria)
- [ ] Unit: password hashing/verification
- [ ] Unit: JWT generation/validation
- [ ] Integration: registration flow end-to-end
- [ ] Integration: login flow end-to-end
- [ ] Integration: password reset flow
- [ ] Performance: login P95 < 200ms
```

## Step 3: Tests from Acceptance Criteria

```typescript
// Translate Given-When-Then directly to tests

// FR-001: Account creation
describe('POST /api/auth/register', () => {
  it('creates account with valid email and password (FR-001)', async () => {
    // Given: valid email and password ≥8 chars
    const input = { email: 'test@example.com', password: 'securepass123' };

    // When: user submits registration
    const response = await request(app).post('/api/auth/register').send(input);

    // Then: account is created
    expect(response.status).toBe(201);
    expect(response.body).toHaveProperty('userId');

    // Then: verification email sent
    expect(emailService.send).toHaveBeenCalledWith(
      expect.objectContaining({ to: 'test@example.com', template: 'verification' })
    );
  });

  it('rejects password shorter than 8 characters (FR-001)', async () => {
    const input = { email: 'test@example.com', password: 'short' };
    const response = await request(app).post('/api/auth/register').send(input);
    expect(response.status).toBe(400);
    expect(response.body.errors).toContainEqual(
      expect.objectContaining({ field: 'password' })
    );
  });

  it('rate limits to 10 requests per IP per hour (NFR-002)', async () => {
    for (let i = 0; i < 10; i++) {
      await request(app).post('/api/auth/register').send({
        email: `user${i}@example.com`, password: 'securepass123',
      });
    }
    const response = await request(app).post('/api/auth/register').send({
      email: 'user11@example.com', password: 'securepass123',
    });
    expect(response.status).toBe(429);
  });
});
```

## Step 4: Traceability Matrix

```markdown
## Requirements Traceability Matrix

| Req ID | Requirement | Test IDs | Code Location | Status |
|--------|------------|----------|---------------|--------|
| FR-001 | Account creation | T-001, T-002, T-003 | src/auth/register.ts | ✅ Implemented |
| FR-002 | User login | T-004, T-005 | src/auth/login.ts | ✅ Implemented |
| FR-003 | Password reset | T-006, T-007, T-008 | src/auth/reset.ts | ✅ Implemented |
| NFR-001 | Login P95 <200ms | T-PERF-001 | tests/perf/login.k6.js | ✅ Verified |
| NFR-002 | Rate limiting | T-009 | src/middleware/rate-limit.ts | ✅ Implemented |
| NFR-003 | Argon2id hashing | T-010 | src/auth/password.ts | ✅ Implemented |

### Coverage
- Requirements with tests: 6/6 (100%)
- Requirements implemented: 6/6 (100%)
- Tests passing: 10/10 (100%)
```

## Best Practices

1. **Parse before coding** — extract all requirements and acceptance criteria first
2. **Test from spec** — translate Given-When-Then directly to test cases
3. **Trace everything** — every requirement → task → code → test
4. **Out-of-scope is sacred** — don't implement what's not in the spec
5. **Review against spec** — final review compares implementation to original requirements
6. **Version the spec** — if spec changes, update traceability matrix

---

**Skill Type**: Workflow — Development Process
**Complexity**: Moderate
**Typical Usage**: Feature implementation from PRDs, compliance-driven development
