# Phase 3: Validation Quality Gate

**Threshold**: 85%
**Purpose**: Prove the solution works in realistic production-like conditions
**Duration**: 2-6 hours for typical features

## Why 85%?

Validation ensures the feature behaves correctly in integrated environments. An 85% threshold balances thoroughness with pragmatism:
- Critical user journeys are validated end-to-end
- Performance meets real-world SLA requirements
- Documentation enables operations and support teams
- Acceptance criteria validated by stakeholders

Higher thresholds delay releases; lower thresholds risk production incidents.

## Gate Criteria

### 1. Integration Tests (Pass/Fail)
**Status**: `integration_tests: "pass"|"fail"`

**Validation Checklist**:
- [ ] Critical user journeys tested end-to-end
- [ ] Cross-service integration validated
- [ ] Database transactions tested with real schemas
- [ ] External API integrations tested (or mocked with contracts)
- [ ] Error scenarios and edge cases covered
- [ ] Tests run against staging environment successfully

**Scope**:
Integration tests validate behavior across system boundaries:
- API contracts between services
- Database persistence and retrieval
- Message queue publishing and consumption
- Third-party API integration
- Authentication and authorization flows

**Common Failures**:
- **Unit-test mindset**: Testing internal implementation, not integration
- **Missing error scenarios**: Only happy path tested
- **Environment dependencies**: Tests pass locally, fail in CI
- **Flaky tests**: Timing issues, shared state, race conditions

**How to Pass**:

```typescript
// ✅ GOOD: Integration test covering full authentication flow
describe('Authentication Flow Integration', () => {
  let testDb: Database;
  let authApi: TestServer;

  beforeAll(async () => {
    // Setup test database with migrations
    testDb = await setupTestDatabase();
    await runMigrations(testDb);

    // Start API server
    authApi = await startTestServer();
  });

  afterAll(async () => {
    await authApi.close();
    await testDb.close();
  });

  it('should complete full registration and login flow', async () => {
    // Step 1: Register new user
    const registerResponse = await authApi.post('/auth/register', {
      email: 'newuser@example.com',
      password: 'SecurePass123!',
      name: 'Test User'
    });
    expect(registerResponse.status).toBe(201);
    expect(registerResponse.body.userId).toBeDefined();

    // Step 2: Verify user created in database
    const user = await testDb.query(
      'SELECT * FROM users WHERE email = $1',
      ['newuser@example.com']
    );
    expect(user.rows).toHaveLength(1);
    expect(user.rows[0].email_verified).toBe(false);

    // Step 3: Verify email sent (check queue)
    const emailQueue = await getMessageQueue('email-verification');
    const messages = await emailQueue.getMessages();
    expect(messages).toContainEqual(
      expect.objectContaining({
        to: 'newuser@example.com',
        template: 'verify-email'
      })
    );

    // Step 4: Simulate email verification
    const verificationToken = messages[0].token;
    const verifyResponse = await authApi.post('/auth/verify-email', {
      token: verificationToken
    });
    expect(verifyResponse.status).toBe(200);

    // Step 5: Login with verified account
    const loginResponse = await authApi.post('/auth/login', {
      email: 'newuser@example.com',
      password: 'SecurePass123!'
    });
    expect(loginResponse.status).toBe(200);
    expect(loginResponse.body.accessToken).toBeDefined();
    expect(loginResponse.body.refreshToken).toBeDefined();

    // Step 6: Validate session created
    const sessionId = loginResponse.body.sessionId;
    const session = await testDb.query(
      'SELECT * FROM sessions WHERE id = $1',
      [sessionId]
    );
    expect(session.rows[0].user_id).toBe(user.rows[0].id);
    expect(session.rows[0].expires_at).toBeGreaterThan(new Date());
  });

  it('should handle authentication failure correctly', async () => {
    // Attempt login with wrong password
    const loginResponse = await authApi.post('/auth/login', {
      email: 'newuser@example.com',
      password: 'WrongPassword'
    });

    expect(loginResponse.status).toBe(401);
    expect(loginResponse.body.error).toBe('INVALID_CREDENTIALS');

    // Verify no session created
    const sessions = await testDb.query(
      'SELECT * FROM sessions WHERE user_id = (SELECT id FROM users WHERE email = $1)',
      ['newuser@example.com']
    );
    expect(sessions.rows).toHaveLength(0);
  });
});
```

**Integration Test Report**:
```
✅ User Registration Flow: PASS (2.3s)
✅ Email Verification Flow: PASS (1.8s)
✅ Login with Valid Credentials: PASS (1.2s)
✅ Login with Invalid Credentials: PASS (0.9s)
✅ Session Expiration Handling: PASS (1.5s)
✅ Password Reset Flow: PASS (2.1s)
✅ MFA Enrollment and Verification: PASS (2.7s)

Integration Tests: 7/7 PASSED ✅
Total Duration: 12.5s
```

### 2. Performance Benchmarks (Pass/Fail)
**Status**: `performance_benchmarks: "pass"|"fail"`

**Validation Checklist**:
- [ ] Response time meets SLA targets (e.g., P95 < 200ms)
- [ ] Throughput meets expected load (e.g., 1000 req/s)
- [ ] Resource utilization acceptable (CPU < 70%, memory < 80%)
- [ ] Database query performance optimized (N+1 queries eliminated)
- [ ] Load testing completed successfully
- [ ] No memory leaks detected

**SLA Targets**:
```yaml
performance_sla:
  response_time:
    p50: < 50ms
    p95: < 200ms
    p99: < 500ms
  throughput:
    minimum: 1000 req/s
    target: 2000 req/s
  resource_utilization:
    cpu: < 70%
    memory: < 80%
    database_connections: < 80% pool
```

**Common Failures**:
- **N+1 query patterns**: Fetching related data in loops
- **Missing indices**: Full table scans on large datasets
- **Memory leaks**: Unclosed connections, event listener accumulation
- **Synchronous blocking**: CPU-intensive operations on main thread

**How to Pass**:

```typescript
// ❌ BAD: N+1 query pattern
async function getUsersWithOrders() {
  const users = await db.query('SELECT * FROM users');

  for (const user of users) {
    // N+1: Executes query for each user
    const orders = await db.query('SELECT * FROM orders WHERE user_id = $1', [user.id]);
    user.orders = orders;
  }

  return users;
}

// ✅ GOOD: Batch loading with single query
async function getUsersWithOrders() {
  const query = `
    SELECT
      u.*,
      json_agg(o.*) as orders
    FROM users u
    LEFT JOIN orders o ON o.user_id = u.id
    GROUP BY u.id
  `;

  const result = await db.query(query);
  return result.rows;
}
```

**Performance Benchmarking**:

```bash
# Load testing with k6
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 1000 },  // Ramp up to 1000 users
    { duration: '5m', target: 1000 },  // Stay at 1000 users
    { duration: '2m', target: 0 },     // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<200', 'p(99)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  let response = http.post('https://staging.api.com/auth/login', {
    email: 'loadtest@example.com',
    password: 'TestPass123!'
  });

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });

  sleep(1);
}
```

**Benchmark Report**:
```
✅ Response Time P50: 45ms (target: <50ms)
✅ Response Time P95: 178ms (target: <200ms)
✅ Response Time P99: 423ms (target: <500ms)
✅ Throughput: 1250 req/s (target: 1000 req/s)
✅ CPU Utilization: 62% (target: <70%)
✅ Memory Utilization: 71% (target: <80%)
✅ Error Rate: 0.08% (target: <1%)

Performance Benchmarks: PASS ✅
```

### 3. Documentation Complete (Boolean)
**Status**: `documentation_complete: true|false`

**Validation Checklist**:
- [ ] API documentation (OpenAPI/Swagger) up to date
- [ ] Architecture Decision Records (ADRs) written
- [ ] Runbook for operations team created
- [ ] Troubleshooting guide with common issues
- [ ] Deployment instructions documented
- [ ] Rollback procedures defined

**Required Documentation**:

**API Documentation** (`docs/api/auth-api.yml`):
```yaml
openapi: 3.0.0
info:
  title: Authentication API
  version: 1.0.0
paths:
  /auth/login:
    post:
      summary: Authenticate user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [email, password]
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  minLength: 12
      responses:
        '200':
          description: Authentication successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  accessToken:
                    type: string
                  refreshToken:
                    type: string
                  expiresIn:
                    type: integer
        '401':
          description: Invalid credentials
        '429':
          description: Rate limit exceeded
```

**Runbook** (`docs/runbooks/auth-service.md`):
```markdown
# Authentication Service Runbook

## Service Overview
- **Purpose**: Handle user authentication and session management
- **SLA**: 99.9% uptime, P95 response time <200ms
- **Dependencies**: PostgreSQL, Redis, Email Service

## Monitoring
- **Dashboard**: https://grafana.company.com/d/auth-service
- **Alerts**: PagerDuty integration for P1 incidents
- **Logs**: CloudWatch Logs group `/aws/lambda/auth-service`

## Common Issues

### High Error Rate (5xx)
**Symptoms**: Error rate >1% for >5 minutes
**Diagnosis**:
```bash
# Check database connectivity
aws rds describe-db-instances --db-instance-identifier auth-db

# Check Redis connectivity
redis-cli -h auth-redis.cache.amazonaws.com ping
```
**Resolution**:
1. Verify database connection pool not exhausted
2. Check Redis memory usage
3. Review recent deployments for regression
4. Rollback if necessary

### Slow Response Times
**Symptoms**: P95 latency >500ms
**Diagnosis**:
```bash
# Check database slow queries
SELECT query, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```
**Resolution**:
1. Identify slow queries
2. Add missing indexes
3. Optimize N+1 patterns
4. Scale read replicas if needed

## Deployment
See [Deployment Guide](../deployment/auth-service.md)

## Rollback Procedure
```bash
# Rollback to previous version
aws lambda update-function-code \
  --function-name auth-service \
  --s3-bucket deployments \
  --s3-key auth-service/v1.2.3.zip
```
```

**Common Failures**:
- **Outdated API docs**: Changes made without updating OpenAPI spec
- **Missing runbooks**: No operational guidance for on-call engineers
- **Incomplete troubleshooting**: Common issues not documented

**How to Pass**:
- Generate API docs from code annotations
- Create runbooks as part of feature development
- Document every production incident in troubleshooting guide
- Review documentation with operations team

### 4. Acceptance Criteria Met (Boolean)
**Status**: `acceptance_criteria_met: true|false`

**Validation Checklist**:
- [ ] All user stories marked complete
- [ ] Acceptance criteria validated by stakeholder
- [ ] Demo completed and approved
- [ ] Success metrics tracked and meeting targets
- [ ] Known limitations documented
- [ ] Sign-off obtained

**Acceptance Criteria Validation**:

```markdown
# User Story: Biometric Authentication

## Acceptance Criteria

### Criterion 1: Fast Authentication
**Requirement**: User authenticates in <3s for P95
**Validation**:
- ✅ Measured P95: 2.1s
- ✅ Performance benchmarks passing
- ✅ Tested with 1000 concurrent users

### Criterion 2: Fallback to Password
**Requirement**: User can switch to password within 1 tap
**Validation**:
- ✅ UI includes "Use Password Instead" button
- ✅ Single tap transitions to password input
- ✅ Tested on iOS and Android

### Criterion 3: Security Compliance
**Requirement**: Biometric data never leaves device
**Validation**:
- ✅ Security audit completed
- ✅ Network traffic analysis shows no biometric data transmission
- ✅ Privacy policy updated

### Criterion 4: User Adoption
**Requirement**: >60% adoption within 30 days
**Validation**:
- ⏳ Tracking metric implemented
- ⏳ Dashboard created for monitoring
- ⏳ A/B test configured (will validate post-launch)

## Stakeholder Sign-Off
**Product Manager**: ✅ Approved (2024-11-15)
**Engineering Lead**: ✅ Approved (2024-11-15)
**Security Team**: ✅ Approved (2024-11-14)

## Known Limitations
- Biometric authentication requires iOS 13+ or Android 10+
- Fallback to password always available for older devices
```

**Common Failures**:
- **Incomplete criteria**: Some acceptance criteria not validated
- **Missing stakeholder approval**: No sign-off from product owner
- **Success metrics undefined**: No way to measure feature success
- **Demo not conducted**: Feature not demonstrated to stakeholders

**How to Pass**:
1. Schedule demo with stakeholders before validation phase
2. Walk through each acceptance criterion with evidence
3. Document any deviations or compromises
4. Obtain explicit approval in writing (email, JIRA comment, etc.)
5. Set up tracking for success metrics

## Scoring Calculation

**Formula**:
```python
integration_score = (integration_tests == "pass") * 25
performance_score = (performance_benchmarks == "pass") * 25
documentation_score = (documentation_complete == True) * 25
acceptance_score = (acceptance_criteria_met == True) * 25

total_score = integration_score + performance_score + documentation_score + acceptance_score
```

**Example 1** (FAIL):
```json
{
  "integration_tests": "pass",
  "performance_benchmarks": "fail",
  "documentation_complete": true,
  "acceptance_criteria_met": false
}
```
**Score**: `25 + 0 + 25 + 0 = 50%` ❌ FAIL (threshold: 85%)

**Example 2** (PASS):
```json
{
  "integration_tests": "pass",
  "performance_benchmarks": "pass",
  "documentation_complete": true,
  "acceptance_criteria_met": true
}
```
**Score**: `25 + 25 + 25 + 25 = 100%` ✅ PASS

## Gate Validation

### Automated Validation

```bash
# Run quality gate checker
python3 scripts/quality-gate-checker.py \
  --phase validation \
  --config ./features/user-auth/.quality-gates.json \
  --integration-report ./test-results/integration.json \
  --perf-report ./benchmarks/k6-results.json \
  --exit-code

# Output:
# ✅ Integration tests: PASS (7/7 suites)
# ✅ Performance benchmarks: PASS (P95: 178ms)
# ✅ Documentation complete: YES
# ✅ Acceptance criteria met: APPROVED
#
# Phase 3 (Validation) Score: 100% ✅
# Threshold: 85%
# Status: PASSED - READY FOR PRODUCTION
```

### Manual Validation

Use the validation report template:

```bash
# Generate validation report
cp templates/quality-gates/VALIDATION_REPORT.template.md features/user-auth/validation-report.md

# Complete checklist and obtain stakeholder approvals
```

## Common Failure Scenarios

### Scenario 1: Integration Tests Failing
**Symptom**: Tests pass locally, fail in CI
**Fix**: Investigate environment differences, fix test setup
**Timeline**: 2-6 hours

### Scenario 2: Performance Below SLA
**Symptom**: P95 response time 350ms (target: <200ms)
**Fix**: Profile code, optimize queries, add caching
**Timeline**: 4-12 hours

### Scenario 3: Incomplete Documentation
**Symptom**: No runbook exists for new service
**Fix**: Create runbook with operations team input
**Timeline**: 2-4 hours

### Scenario 4: Stakeholder Not Available
**Symptom**: Cannot get acceptance sign-off
**Fix**: Schedule demo, use async approval via email/JIRA
**Timeline**: 1-3 days

## Integration with Workflow

```bash
# Validation phase with automatic gate validation
/workflow-feature-development ./features/user-auth --stage=testing

# Output includes gate status:
# 🔍 Phase 3: Validation
# ✅ Integration Tests: 7/7 PASS
# ✅ Performance: P95 178ms < 200ms
# ✅ Documentation: COMPLETE
# ✅ Acceptance: APPROVED
#
# 🎯 Gate Score: 100% (threshold: 85%)
# ✅ PASSED - READY FOR PRODUCTION DEPLOYMENT
#
# 📋 Next Steps:
# 1. Create deployment plan
# 2. Schedule production release
# 3. Configure monitoring alerts
# 4. Prepare rollback procedure
```

## Production Readiness Checklist

After passing Phase 3:
- [ ] Deployment plan reviewed and approved
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set in PagerDuty/OpsGenie
- [ ] Runbook accessible to on-call engineers
- [ ] Rollback procedure tested
- [ ] Feature flags configured for gradual rollout
- [ ] Stakeholders notified of deployment schedule

## Next Steps

After passing Phase 3:
1. **Schedule production deployment**: Coordinate with DevOps team
2. **Configure monitoring**: Set up dashboards and alerts
3. **Enable feature flags**: Plan gradual rollout strategy
4. **Brief on-call team**: Share runbook and escalation procedures
5. **Track success metrics**: Monitor KPIs for 7-14 days post-launch

**Related Documentation**:
- [Quality Gates Overview](README.md)
- [Phase 1: Analysis Gate](phase-1-analysis.md)
- [Phase 2: Implementation Gate](phase-2-implementation.md)
- [Validation Report Template](../../templates/quality-gates/VALIDATION_REPORT.template.md)
