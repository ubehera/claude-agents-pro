---
name: code-review-patterns
description: Load when conducting systematic code reviews for security, correctness, performance, and maintainability across pull requests
trigger_keywords: [code review, pr review, pull request, review checklist, code quality, security review, performance review, spec compliance]
---

# Code Review Patterns

Systematic code review methodology focusing on security, correctness, spec alignment, performance, and maintainability.

## Overview

Code reviews validate changes against acceptance criteria, architectural standards, and quality benchmarks. Use structured evaluation to catch issues before production.

**When to Use**:
- Pull request reviews
- Pre-merge code validation
- Architecture compliance checks
- Security vulnerability assessment
- Performance optimization verification

## Core Concepts

- **Weighted Quality Dimensions**: Security (25%) and Correctness (25%) outweigh Performance (15%) and Maintainability (15%) - critical issues block merge, nice-to-haves don't
- **Context-First Review**: Read specs, user stories, and API contracts before reviewing code - understanding intent prevents false positives and catches missed requirements
- **Severity-Based Triage**: Critical issues (security, data loss) block merge; Important issues (bugs, performance) should fix; Nice-to-haves are optional - clear priority prevents review fatigue
- **Specific, Actionable Feedback**: Point to exact lines, provide fix examples, explain impact - vague feedback like "make this better" wastes time and frustrates developers
- **Highlight Good Practices**: Reinforcing quality work encourages repetition and builds team knowledge - reviews aren't just for finding problems

## Review Strategy Selection

### Sequential Review (< 5 files, single concern)

**Best For**:
- Small bug fixes
- Feature additions to single module
- Documentation updates
- Configuration changes

**Process**:
1. Gather context from specs and requirements
2. Review sequentially through quality dimensions
3. Generate unified report with findings
4. Recommend approval/revision/rework

### Parallel Review (> 5 files, multiple concerns)

**Best For**:
- Multi-component features
- Cross-cutting changes
- Architectural modifications
- Major refactors

**Process**:
1. Identify independent review aspects (security, API, UI, data)
2. Delegate to specialist reviewers for each dimension
3. Consolidate findings across reviewers
4. Generate aggregate assessment

## Context Gathering Phase

### Documentation Review

**Required Reading**:
- Feature specs: `docs/feature-spec/F-##-*.md`
- User stories: `docs/user-stories/US-###-*.md`
- API contracts: `docs/api-contracts.yaml`
- Architecture docs: `docs/system-design.md`
- Design specs: `docs/design-spec.md` (UI changes)
- Implementation plan: `docs/plans/<slug>/plan.md`

### Scope Assessment

**Determine**:
- Files changed and affected features
- Story implementations (US-### IDs)
- API/database/schema modifications
- Breaking changes or compatibility impacts
- Test coverage requirements

## Quality Dimensions Framework

### Security (Weight: 25/100)

**Critical Checks**:
- ✅ Input validation and sanitization
- ✅ Authentication and authorization enforcement
- ✅ Sensitive data handling (PII, credentials)
- ✅ SQL injection prevention
- ✅ XSS prevention in user input
- ✅ CSRF protection on state-changing operations
- ✅ Secrets not hardcoded
- ✅ Secure defaults for configuration

**Examples**:

```typescript
// ❌ CRITICAL: SQL injection vulnerability
const query = `SELECT * FROM users WHERE email = '${email}'`;

// ✅ GOOD: Parameterized query
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [email]);
```

```typescript
// ❌ CRITICAL: No authentication check
app.delete('/api/users/:id', async (req, res) => {
    await deleteUser(req.params.id);
    res.sendStatus(204);
});

// ✅ GOOD: Authentication and authorization
app.delete('/api/users/:id', requireAuth, async (req, res) => {
    const user = req.user;
    if (user.id !== req.params.id && !user.isAdmin) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    await deleteUser(req.params.id);
    res.sendStatus(204);
});
```

### Correctness (Weight: 25/100)

**Critical Checks**:
- ✅ Logic matches acceptance criteria
- ✅ Edge cases handled (empty arrays, null values, boundaries)
- ✅ Error handling complete and specific
- ✅ Null/undefined checks present
- ✅ Type safety maintained
- ✅ Race conditions prevented
- ✅ Transaction boundaries correct

**Examples**:

```typescript
// ❌ IMPORTANT: Missing null check
function getFullName(user) {
    return user.firstName + ' ' + user.lastName;
}

// ✅ GOOD: Defensive programming
function getFullName(user) {
    if (!user || !user.firstName || !user.lastName) {
        return 'Unknown';
    }
    return `${user.firstName} ${user.lastName}`;
}
```

```python
# ❌ IMPORTANT: Missing error handling
def process_payment(amount):
    charge = stripe.Charge.create(amount=amount)
    return charge.id

# ✅ GOOD: Comprehensive error handling
def process_payment(amount):
    try:
        charge = stripe.Charge.create(amount=amount)
        return {'success': True, 'chargeId': charge.id}
    except stripe.error.CardError as e:
        return {'success': False, 'error': 'Card declined'}
    except stripe.error.RateLimitError:
        return {'success': False, 'error': 'Rate limit exceeded'}
    except Exception as e:
        logger.error(f"Payment failed: {e}")
        return {'success': False, 'error': 'Payment processing failed'}
```

### Spec Alignment (Weight: 20/100)

**Critical Checks**:
- ✅ API endpoints match `docs/api-contracts.yaml`
- ✅ Request/response schemas conform to spec
- ✅ Error responses follow documented format
- ✅ Data events match `docs/data-plan.md`
- ✅ UI components match `docs/design-spec.md`
- ✅ Implementation follows feature spec logic

**Examples**:

```typescript
// ❌ CRITICAL: Breaks API contract
// Spec says: POST /api/users returns 201 with user object
app.post('/api/users', async (req, res) => {
    const user = await createUser(req.body);
    res.json(user);  // ❌ Wrong status code (200 instead of 201)
});

// ✅ GOOD: Follows API contract
app.post('/api/users', async (req, res) => {
    const user = await createUser(req.body);
    res.status(201).json(user);
});
```

### Performance (Weight: 15/100)

**Critical Checks**:
- ✅ Algorithm efficiency (avoid O(n²) when O(n) possible)
- ✅ Database query optimization (indexes, joins)
- ✅ N+1 query prevention
- ✅ Caching appropriate for read-heavy operations
- ✅ Resource usage reasonable (memory, CPU)
- ✅ Async operations don't block unnecessarily

**Examples**:

```typescript
// ❌ IMPORTANT: N+1 query problem
async function getUsersWithPosts() {
    const users = await db.query('SELECT * FROM users');
    return Promise.all(
        users.map(async user => ({
            ...user,
            posts: await db.query('SELECT * FROM posts WHERE userId = $1', [user.id])
        }))
    );
}

// ✅ GOOD: Single optimized query
async function getUsersWithPosts() {
    const result = await db.query(`
        SELECT u.*, json_agg(p.*) as posts
        FROM users u
        LEFT JOIN posts p ON p.userId = u.id
        GROUP BY u.id
    `);
    return result.rows;
}
```

```python
# ❌ IMPORTANT: Inefficient algorithm
def find_duplicates(items):
    duplicates = []
    for i, item in enumerate(items):
        for j, other in enumerate(items[i+1:]):
            if item == other and item not in duplicates:
                duplicates.append(item)
    return duplicates  # O(n²)

# ✅ GOOD: Efficient algorithm
def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)  # O(n)
```

### Maintainability (Weight: 15/100)

**Critical Checks**:
- ✅ Code clarity and readability
- ✅ Consistent with codebase patterns
- ✅ Appropriate abstraction levels
- ✅ Functions are focused and small (<100 lines)
- ✅ Comments explain "why", not "what"
- ✅ Magic numbers extracted to constants
- ✅ DRY principle applied

**Examples**:

```typescript
// ❌ NICE-TO-HAVE: Magic numbers
function calculateDiscount(total) {
    if (total > 100) {
        return total * 0.1;
    } else if (total > 50) {
        return total * 0.05;
    }
    return 0;
}

// ✅ GOOD: Named constants
const DISCOUNT_TIERS = {
    PREMIUM: { threshold: 100, rate: 0.1 },
    STANDARD: { threshold: 50, rate: 0.05 },
};

function calculateDiscount(total) {
    if (total >= DISCOUNT_TIERS.PREMIUM.threshold) {
        return total * DISCOUNT_TIERS.PREMIUM.rate;
    } else if (total >= DISCOUNT_TIERS.STANDARD.threshold) {
        return total * DISCOUNT_TIERS.STANDARD.rate;
    }
    return 0;
}
```

## Finding Priority Levels

### 🔴 CRITICAL (Must fix before merge)

**Severity**: Breaking changes, security vulnerabilities, data corruption risks

**Examples**:
- SQL injection vulnerability
- Missing authentication on protected endpoints
- Breaking API contract changes
- Data loss scenarios
- Race conditions causing corruption

**Format**:
```
🔴 CRITICAL: SQL Injection Vulnerability

Location: api/users.ts:45
Problem: User input directly interpolated into SQL query
Impact: Attackers can execute arbitrary SQL commands
Fix: Use parameterized queries with prepared statements
Spec reference: N/A (security requirement)

// Current code:
const query = `SELECT * FROM users WHERE email = '${email}'`;

// Required fix:
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [email]);
```

### 🟡 IMPORTANT (Should fix)

**Severity**: Logic bugs, missing error handling, performance issues

**Examples**:
- Missing null/undefined checks
- Incomplete error handling
- N+1 query problems
- Missing edge case handling
- Accessibility violations

**Format**:
```
🟡 IMPORTANT: Missing Error Handling

Location: services/payment.ts:67
Problem: Payment API call has no error handling
Impact: Unhandled exceptions cause server crashes
Fix: Add try-catch with specific error handling
```

### 🟢 NICE-TO-HAVE (Optional improvements)

**Severity**: Code style, minor refactors, documentation

**Examples**:
- Magic numbers extraction
- Function complexity reduction
- Better variable names
- Additional comments
- Code duplication

### ✅ GOOD PRACTICES

**Highlight**: What was done well for learning and reinforcement

**Examples**:
- Excellent test coverage
- Clear separation of concerns
- Well-documented complex logic
- Appropriate use of TypeScript types
- Performance optimization

## Review Report Structure

```markdown
# Code Review: [Feature/PR Title]

## Summary
**Quality Score:** 78/100
**Issues Found:**
- 🔴 Critical: 1
- 🟡 Important: 3
- 🟢 Nice-to-have: 2

**Assessment:** NEEDS REVISION

## Spec Compliance
- [x] APIs match `docs/api-contracts.yaml`
- [ ] Events match `docs/data-plan.md` (missing USER_DELETED event)
- [x] UI matches `docs/design-spec.md`
- [x] Logic satisfies acceptance criteria

## Findings by Priority

### 🔴 Critical Issues (Must Fix)

#### 1. SQL Injection Vulnerability
**Location:** `api/users.ts:45`
**Problem:** User email directly interpolated into SQL query
**Impact:** Attackers can execute arbitrary SQL commands
**Fix:** Use parameterized queries
```typescript
// Replace this:
const query = `SELECT * FROM users WHERE email = '${email}'`;

// With this:
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [email]);
```

### 🟡 Important Issues (Should Fix)

#### 1. Missing Error Handling
**Location:** `services/payment.ts:67`
**Problem:** No try-catch around Stripe API call
**Impact:** Unhandled exceptions crash server
**Fix:** Add comprehensive error handling

#### 2. N+1 Query Problem
**Location:** `api/posts.ts:23`
**Problem:** Fetching users in loop for each post
**Impact:** Performance degrades with scale
**Fix:** Use JOIN or batch loading

#### 3. Missing Analytics Event
**Location:** `controllers/user.ts:89`
**Problem:** User deletion not tracked per `docs/data-plan.md`
**Impact:** Analytics incomplete for user lifecycle
**Fix:** Emit USER_DELETED event

### 🟢 Nice-to-Have Suggestions

#### 1. Magic Number Extraction
**Location:** `utils/discount.ts:12`
**Suggestion:** Extract 0.1 and 0.05 to named constants

#### 2. Function Complexity
**Location:** `services/order.ts:45`
**Suggestion:** Extract validation logic to separate function

### ✅ Good Practices

- Excellent test coverage (95% for new code)
- Clear TypeScript types throughout
- Proper use of async/await
- Good separation of concerns

## Quality Dimension Scores

- **Security:** 15/25 (SQL injection issue)
- **Correctness:** 20/25 (missing error handling)
- **Spec Alignment:** 16/20 (missing analytics event)
- **Performance:** 10/15 (N+1 query)
- **Maintainability:** 17/15 (excellent code organization)

**Total:** 78/100

## Recommendations

1. **Fix critical SQL injection** before merge (security requirement)
2. **Add error handling** for external API calls (reliability)
3. **Optimize database queries** to prevent N+1 problem (performance)
4. **Emit missing analytics event** per data plan (spec compliance)
5. Consider addressing nice-to-have suggestions in follow-up PR

**Verdict:** NEEDS REVISION - Address critical and important issues before merge
```

## Review Workflow

### Step 1: Context Gathering (5 min)
- Read feature spec, user stories, API contracts
- Understand acceptance criteria
- Identify affected systems

### Step 2: Initial Scan (5 min)
- Skim all changed files
- Identify review focus areas
- Note potential red flags

### Step 3: Deep Review (20-30 min)
- Security: Check authentication, input validation, data handling
- Correctness: Verify logic, error handling, edge cases
- Spec: Compare against documented contracts
- Performance: Review algorithms, queries, resource usage
- Maintainability: Assess clarity, patterns, abstractions

### Step 4: Testing Review (5 min)
- Verify test coverage for new code
- Check test quality (not just mocking)
- Ensure edge cases tested

### Step 5: Report Generation (10 min)
- Categorize findings by priority
- Provide specific fix recommendations
- Calculate quality score
- Make approval recommendation

## Best Practices

1. **Be Specific**: Point to exact lines, provide fix examples
2. **Be Constructive**: Suggest improvements, not just problems
3. **Be Consistent**: Use priority levels consistently
4. **Be Thorough**: Don't skip security checks
5. **Be Fast**: Timely reviews unblock development
6. **Highlight Good Work**: Positive reinforcement drives quality

## Common Anti-Patterns to Catch

### Security Anti-Patterns
- Hardcoded secrets or API keys
- Missing authentication/authorization
- Unsanitized user input
- Insecure defaults

### Logic Anti-Patterns
- Missing null checks
- Unhandled promise rejections
- Race conditions
- Off-by-one errors

### Performance Anti-Patterns
- N+1 queries
- Synchronous blocking operations
- Unnecessary data loading
- Missing indexes on queried fields

### Maintainability Anti-Patterns
- God objects/functions
- Deep nesting (>3 levels)
- Magic numbers
- Unclear variable names

---

**Skill Type**: Code Quality - Review
**Complexity**: Moderate to High
**Typical Usage**: Activated for PR reviews and code quality assessments
**Performance**: Systematic approach ensures comprehensive coverage in 30-45 minutes
