# Phase 2: Implementation Quality Gate

**Threshold**: 80%
**Purpose**: Ensure production-ready code with adequate testing and security
**Duration**: 4-16 hours for typical features

## Why 80%?

Implementation quality balances velocity with maintainability. An 80% threshold ensures:
- Core business logic is thoroughly tested
- Critical security vulnerabilities are caught
- Code quality standards prevent technical debt accumulation
- Team maintains sustainable development pace

Lower thresholds accumulate debt; higher thresholds risk analysis paralysis.

## Gate Criteria

### 1. Test Coverage (Numeric: ≥80)
**Status**: `test_coverage: <percentage>`

**Validation Checklist**:
- [ ] Unit tests for core domain logic (≥90% coverage)
- [ ] Integration tests for API endpoints (≥70% coverage)
- [ ] Edge case coverage for critical paths
- [ ] Test pyramid structure maintained (unit > integration > e2e)
- [ ] Coverage report generated and reviewed
- [ ] Tests execute in <5 minutes locally

**Measurement**:
```bash
# Generate coverage report
npm test -- --coverage
# or
pytest --cov=src --cov-report=term --cov-report=json

# Coverage thresholds in package.json
{
  "jest": {
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
      }
    }
  }
}
```

**Common Failures**:
- **Testing implementation details**: Tests coupled to internal structure
- **Low branch coverage**: Happy path only, no error handling tests
- **Flaky tests**: Tests fail intermittently due to timing or state issues
- **Slow test suite**: Tests take >10 minutes, discouraging frequent execution

**How to Pass**:
```typescript
// ✅ GOOD: Test behavior, not implementation
describe('UserAuthentication', () => {
  it('should authenticate user with valid credentials', async () => {
    const authService = new AuthenticationService(mockUserRepo);
    const result = await authService.authenticate({
      email: 'user@example.com',
      password: 'ValidPass123!'
    });

    expect(result.success).toBe(true);
    expect(result.session).toBeDefined();
    expect(result.session.expiresAt).toBeGreaterThan(Date.now());
  });

  it('should reject authentication with invalid password', async () => {
    const authService = new AuthenticationService(mockUserRepo);
    const result = await authService.authenticate({
      email: 'user@example.com',
      password: 'WrongPassword'
    });

    expect(result.success).toBe(false);
    expect(result.error).toBe('INVALID_CREDENTIALS');
    expect(result.session).toBeUndefined();
  });

  it('should rate-limit after 5 failed attempts', async () => {
    const authService = new AuthenticationService(mockUserRepo);

    // Simulate 5 failed attempts
    for (let i = 0; i < 5; i++) {
      await authService.authenticate({ email: 'user@example.com', password: 'wrong' });
    }

    const result = await authService.authenticate({
      email: 'user@example.com',
      password: 'ValidPass123!'
    });

    expect(result.success).toBe(false);
    expect(result.error).toBe('RATE_LIMITED');
  });
});
```

**Coverage Breakdown**:
```
File                  | % Stmts | % Branch | % Funcs | % Lines |
----------------------|---------|----------|---------|---------|
auth-service.ts       |   95.12 |    88.89 |     100 |   95.12 |
user-repository.ts    |   87.50 |    75.00 |   83.33 |   87.50 |
session-manager.ts    |   92.31 |    85.71 |     100 |   92.31 |
----------------------|---------|----------|---------|---------|
Total                 |   91.67 |    83.33 |   94.44 |   91.67 | ✅
```

### 2. Security Scan (Pass/Fail)
**Status**: `security_scan: "pass"|"fail"`

**Validation Checklist**:
- [ ] Dependency vulnerability scan (npm audit, Snyk, etc.)
- [ ] Zero critical or high-severity vulnerabilities
- [ ] SAST (Static Application Security Testing) passed
- [ ] Secrets detection scan passed (no API keys, passwords in code)
- [ ] Security best practices enforced (input validation, output encoding)
- [ ] Authentication and authorization patterns reviewed

**Tools**:
```bash
# Dependency scanning
npm audit --audit-level=high
# or
snyk test --severity-threshold=high

# SAST scanning
semgrep --config=auto src/
# or
bandit -r src/ -ll  # Python

# Secrets detection
git-secrets --scan
# or
trufflehog filesystem ./
```

**Common Failures**:
- **Outdated dependencies**: Using packages with known vulnerabilities
- **Hardcoded secrets**: API keys, database credentials in source
- **Missing input validation**: SQL injection, XSS vulnerabilities
- **Insecure authentication**: Weak password policies, missing MFA

**How to Pass**:

```typescript
// ✅ GOOD: Input validation and sanitization
import { z } from 'zod';
import { sanitize } from 'dompurify';

const LoginSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(12).max(128)
});

async function authenticate(input: unknown) {
  // Validate input against schema
  const validated = LoginSchema.parse(input);

  // Use parameterized queries (prevents SQL injection)
  const user = await db.query(
    'SELECT * FROM users WHERE email = $1',
    [validated.email]
  );

  // Constant-time comparison (prevents timing attacks)
  const isValid = await crypto.timingSafeEqual(
    Buffer.from(user.passwordHash),
    Buffer.from(await hashPassword(validated.password))
  );

  return isValid;
}

// ✅ GOOD: Secrets management
import { SecretsManager } from '@aws-sdk/client-secrets-manager';

async function getDatabaseCredentials() {
  const secretsManager = new SecretsManager({ region: 'us-east-1' });
  const secret = await secretsManager.getSecretValue({
    SecretId: process.env.DB_SECRET_ARN  // ✅ ARN from environment, not hardcoded
  });
  return JSON.parse(secret.SecretString);
}
```

**Security Scan Report**:
```
✅ npm audit: 0 vulnerabilities found
✅ Snyk: No high-severity issues
✅ Semgrep: All checks passed
✅ git-secrets: No secrets detected

Security Scan: PASS ✅
```

### 3. Lint Errors (Numeric: = 0)
**Status**: `lint_errors: <count>`

**Validation Checklist**:
- [ ] Zero ESLint/Pylint errors (warnings acceptable)
- [ ] Code formatting consistent (Prettier, Black)
- [ ] TypeScript strict mode enabled with zero errors
- [ ] Import organization and unused imports removed
- [ ] Naming conventions followed consistently
- [ ] Complexity metrics within acceptable ranges

**Configuration**:
```json
// .eslintrc.json
{
  "extends": ["eslint:recommended", "plugin:@typescript-eslint/recommended"],
  "rules": {
    "no-console": "error",
    "no-unused-vars": "error",
    "@typescript-eslint/no-explicit-any": "error",
    "complexity": ["error", 10],
    "max-lines-per-function": ["error", 50]
  }
}
```

**Common Failures**:
- **Unused imports**: Dead code accumulation
- **Type safety violations**: Using `any` instead of proper types
- **High complexity**: Functions with cyclomatic complexity >10
- **Inconsistent formatting**: Mix of tabs/spaces, varying brace styles

**How to Pass**:

```typescript
// ❌ BAD: Lint errors
import { useState, useEffect } from 'react';  // unused import

function processData(data: any) {  // no-explicit-any error
  if (data.type == 'user') {  // should use ===
    console.log(data);  // no-console error
    return data.name
  }
}

// ✅ GOOD: No lint errors
import { User } from './types';

function processUser(user: User): string {
  if (user.type === 'admin') {
    logger.info('Processing admin user', { userId: user.id });
    return user.name;
  }
  return user.displayName;
}
```

**Lint Report**:
```
✨ ESLint: 0 errors, 0 warnings
✨ Prettier: All files formatted
✨ TypeScript: 0 errors, strict mode enabled

Lint Check: PASS ✅
```

### 4. Code Review (Pass/Fail)
**Status**: `code_review: "pass"|"fail"`

**Validation Checklist**:
- [ ] Pull request reviewed by qualified team member
- [ ] Architectural patterns followed
- [ ] No commented-out code or debug statements
- [ ] Error handling comprehensive and consistent
- [ ] Performance considerations addressed
- [ ] Reviewer approval obtained

**Review Focus Areas**:

**Architecture & Design**:
- Domain model alignment with DDD principles
- Proper separation of concerns (API/domain/infrastructure)
- Dependency injection and testability
- Appropriate design patterns for use case

**Code Quality**:
- Readability and maintainability
- Naming clarity and consistency
- Function/method size and complexity
- DRY principle (avoid duplication)

**Error Handling**:
- Comprehensive error coverage
- Meaningful error messages
- Proper error propagation
- Graceful degradation

**Performance**:
- No N+1 query patterns
- Appropriate caching strategies
- Efficient algorithms and data structures
- Resource cleanup (connections, file handles)

**Common Failures**:
- **No reviewer assigned**: PR merged without review
- **Rubber-stamp approval**: Review completed in <5 minutes for large PR
- **Unaddressed comments**: Reviewer feedback ignored
- **Missing context**: PR description lacks rationale

**How to Pass**:

```markdown
# Pull Request Template

## Description
Implements JWT-based authentication with biometric fallback for mobile app.

## Changes
- Added `AuthenticationService` with JWT generation
- Implemented `BiometricProvider` with fallback to password
- Added integration tests for auth flow
- Updated API documentation with auth endpoints

## Related Issues
Closes #123, #124

## Testing
- [x] Unit tests added (coverage: 92%)
- [x] Integration tests passing
- [x] Manual testing on iOS and Android
- [x] Performance: P95 auth time <2s

## Security Considerations
- JWT secrets stored in AWS Secrets Manager
- Biometric data never leaves device
- Rate limiting: 5 attempts per 15 minutes
- Session expiration: 24 hours

## Reviewer Checklist
- [ ] Code follows project patterns
- [ ] Error handling comprehensive
- [ ] Tests adequate and passing
- [ ] Documentation updated
- [ ] Performance acceptable

## Reviewer: @tech-lead
**Status**: ✅ APPROVED

**Comments**:
- Excellent separation of concerns between auth and biometric layers
- JWT expiration handling is robust
- Minor: Consider adding metrics for auth success/failure rates
```

## Scoring Calculation

**Formula**:
```python
coverage_score = (test_coverage >= 80) * 25
security_score = (security_scan == "pass") * 25
lint_score = (lint_errors == 0) * 25
review_score = (code_review == "pass") * 25

total_score = coverage_score + security_score + lint_score + review_score
```

**Example 1** (FAIL):
```json
{
  "test_coverage": 65,
  "security_scan": "pass",
  "lint_errors": 3,
  "code_review": "pass"
}
```
**Score**: `0 + 25 + 0 + 25 = 50%` ❌ FAIL (threshold: 80%)

**Example 2** (PASS):
```json
{
  "test_coverage": 87,
  "security_scan": "pass",
  "lint_errors": 0,
  "code_review": "pass"
}
```
**Score**: `25 + 25 + 25 + 25 = 100%` ✅ PASS

## Gate Validation

### Automated Validation

```bash
# Run quality gate checker
python3 scripts/quality-gate-checker.py \
  --phase implementation \
  --config ./features/user-auth/.quality-gates.json \
  --coverage-report ./coverage/lcov.info \
  --lint-report ./lint-results.json \
  --exit-code

# Output:
# ✅ Test coverage: 87% (threshold: 80%)
# ✅ Security scan: PASS
# ✅ Lint errors: 0
# ✅ Code review: APPROVED
#
# Phase 2 (Implementation) Score: 100% ✅
# Threshold: 80%
# Status: PASSED
```

### CI/CD Integration

```yaml
# .github/workflows/implementation-gate.yml
name: Implementation Gate

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  test-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: npm ci
      - name: Run tests with coverage
        run: npm test -- --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          fail_ci_if_error: true
          threshold: 80

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run npm audit
        run: npm audit --audit-level=high
      - name: Run Snyk scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  lint-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ESLint
        run: npm run lint -- --max-warnings 0
      - name: Check TypeScript
        run: npm run type-check

  quality-gate:
    needs: [test-coverage, security-scan, lint-check]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Implementation Gate
        run: |
          python3 scripts/quality-gate-checker.py \
            --phase implementation \
            --config .quality-gates.json \
            --exit-code
```

## Common Failure Scenarios

### Scenario 1: Insufficient Coverage
**Symptom**: Coverage report shows 65%
**Fix**: Add unit tests for uncovered branches
**Timeline**: 2-4 hours

```bash
# Identify uncovered lines
npm test -- --coverage --coverageReporters=html
open coverage/lcov-report/index.html
```

### Scenario 2: Security Vulnerabilities
**Symptom**: npm audit shows critical vulnerabilities
**Fix**: Update dependencies or apply patches
**Timeline**: 1-3 hours

```bash
# Update vulnerable packages
npm audit fix

# For unfixable vulnerabilities, evaluate risk
npm audit fix --force
# or add to audit ignore with justification
```

### Scenario 3: Lint Errors
**Symptom**: ESLint reports 15 errors
**Fix**: Auto-fix where possible, manual correction otherwise
**Timeline**: 30 minutes - 2 hours

```bash
# Auto-fix most issues
npm run lint -- --fix

# Manual fixes for complex cases
# Example: Refactor complex function to reduce cyclomatic complexity
```

### Scenario 4: Code Review Blocked
**Symptom**: Reviewer requests changes
**Fix**: Address feedback and re-request review
**Timeline**: Variable (2 hours - 2 days)

## Integration with Workflow

```bash
# Implementation phase with automatic gate validation
/workflow-feature-development ./features/user-auth --stage=implementation

# Output includes gate status:
# 🔨 Phase 2: Implementation
# ✅ Test Coverage: 87%
# ✅ Security Scan: PASS
# ✅ Lint Errors: 0
# ✅ Code Review: APPROVED
#
# 🎯 Gate Score: 100% (threshold: 80%)
# ✅ PASSED - Proceeding to Phase 3
```

## Next Steps

After passing Phase 2:
1. **Update TodoWrite**: Mark implementation phase complete
2. **Proceed to Phase 3**: Validation with integration tests
3. **Archive artifacts**: Store implementation summary and test reports
4. **Deploy to staging**: Enable integration and E2E testing

**Related Documentation**:
- [Phase 3: Validation Gate](phase-3-validation.md)
- [Quality Gates Overview](README.md)
- [Implementation Summary Template](../../templates/quality-gates/IMPLEMENTATION_SUMMARY.template.md)
