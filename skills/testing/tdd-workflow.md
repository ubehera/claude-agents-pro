---
name: tdd-workflow
description: Load when implementing features with test-driven development, ensuring tests are written before code and actually verify behavior
trigger_keywords: [tdd, test-driven development, test first, red green refactor, failing test, test coverage, behavior verification]
---

# Test-Driven Development Workflow

Strict TDD methodology ensuring all code is driven by failing tests first, proving that tests actually verify behavior.

## Overview

Test-Driven Development (TDD) is writing tests before implementation code. The core principle: **if you didn't watch the test fail, you don't know if it tests the right thing.**

**When to Use**:
- New features and functionality
- Bug fixes (reproduce bug with test first)
- Refactoring (safety net for changes)
- Behavior modifications

**When to Ask First**:
- Throwaway prototypes
- Configuration files
- Generated code

## Core Concepts

- **Red-Green-Refactor Cycle**: Write failing test (RED), implement minimal code to pass (GREEN), improve code quality (REFACTOR) - this order is non-negotiable
- **Watching Tests Fail Proves They Work**: If you didn't see the test fail first, you don't know it tests the right thing - tests written after implementation pass immediately and prove nothing
- **Minimal Implementation**: Write only enough code to make the failing test pass - no extra features, no premature optimization, no untested functionality
- **Tests Drive Design**: Difficulty writing tests signals design problems - if testing is hard, the interface is too coupled or complex
- **Sunk Cost Fallacy**: Code written without TDD should be deleted and restarted test-first - unverified code is technical debt regardless of time invested

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

**Consequences of Violation**:
- Delete the code
- Start over with TDD
- No exceptions

Tests written after code prove nothing because they pass immediately. You never verified they catch the bug they claim to test.

## Red-Green-Refactor Cycle

```
┌─────────────────────────────────────────────────┐
│  1. RED: Write failing test                    │
│     ↓                                           │
│  2. Verify RED: Watch it fail correctly         │
│     ↓                                           │
│  3. GREEN: Write minimal code to pass           │
│     ↓                                           │
│  4. Verify GREEN: Watch it pass                 │
│     ↓                                           │
│  5. REFACTOR: Clean up (stay green)             │
│     ↓                                           │
│  6. Repeat for next behavior                    │
└─────────────────────────────────────────────────┘
```

### Step 1: RED - Write Failing Test

Write one minimal test showing what should happen.

**Good Test**:
```typescript
test('retries failed operations 3 times', async () => {
    let attempts = 0;
    const operation = () => {
        attempts++;
        if (attempts < 3) throw new Error('fail');
        return 'success';
    };

    const result = await retryOperation(operation);

    expect(result).toBe('success');
    expect(attempts).toBe(3);
});
```

**Why This is Good**:
- Clear name describing behavior
- Tests real code, not mocks
- Single responsibility
- Verifiable outcome

**Bad Test**:
```typescript
test('retry works', async () => {
    const mock = jest.fn()
        .mockRejectedValueOnce(new Error())
        .mockRejectedValueOnce(new Error())
        .mockResolvedValueOnce('success');

    await retryOperation(mock);
    expect(mock).toHaveBeenCalledTimes(3);
});
```

**Why This is Bad**:
- Vague name
- Tests mock behavior, not actual code
- No verification of return value

### Step 2: Verify RED - Watch It Fail

**MANDATORY. NEVER SKIP.**

```bash
npm test path/to/test.test.ts
```

**Confirm**:
- ✅ Test fails (not errors)
- ✅ Failure message is expected
- ✅ Fails because feature is missing (not typos)

**Test passes immediately?**
→ You're testing existing behavior. Fix the test.

**Test errors instead of failing?**
→ Fix the error, re-run until it fails correctly.

**Why This Matters**: If you didn't watch it fail, you don't know if the test actually tests the code. Tests written after implementation pass immediately and prove nothing.

### Step 3: GREEN - Minimal Code

Write the simplest code to make the test pass.

**Good Implementation**:
```typescript
async function retryOperation<T>(fn: () => Promise<T>): Promise<T> {
    for (let i = 0; i < 3; i++) {
        try {
            return await fn();
        } catch (e) {
            if (i === 2) throw e;
        }
    }
    throw new Error('unreachable');
}
```

**Why This is Good**:
- Just enough to pass the test
- No extra features
- No premature optimization

**Bad Implementation**:
```typescript
async function retryOperation<T>(
    fn: () => Promise<T>,
    options?: {
        maxRetries?: number;
        backoff?: 'linear' | 'exponential';
        onRetry?: (attempt: number) => void;
        timeout?: number;
    }
): Promise<T> {
    // YAGNI - You Aren't Gonna Need It
    // The test doesn't require any of this
}
```

**Why This is Bad**:
- Over-engineered
- Adding features not tested
- Violates YAGNI principle

**Don't**:
- Add untested features
- Refactor other code (wait for REFACTOR step)
- "Improve" beyond what the test requires

### Step 4: Verify GREEN - Watch It Pass

**MANDATORY.**

```bash
npm test path/to/test.test.ts
```

**Confirm**:
- ✅ New test passes
- ✅ All existing tests pass
- ✅ No errors or warnings

**Test still fails?**
→ Fix implementation, not test.

**Other tests fail?**
→ Fix regression immediately.

### Step 5: REFACTOR - Clean Up

After green only:
- Remove duplication
- Improve names
- Extract helpers
- Simplify logic

**Keep tests green throughout refactoring.**

```typescript
// Before refactoring
async function retryOperation<T>(fn: () => Promise<T>): Promise<T> {
    for (let i = 0; i < 3; i++) {
        try {
            return await fn();
        } catch (e) {
            if (i === 2) throw e;
        }
    }
    throw new Error('unreachable');
}

// After refactoring (still passes same test)
const MAX_RETRIES = 3;

async function retryOperation<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            return await fn();
        } catch (e) {
            lastError = e as Error;
            if (attempt === MAX_RETRIES) throw lastError;
        }
    }

    throw new Error('unreachable');
}
```

### Step 6: Repeat

Write next failing test for next behavior.

## Good Tests Checklist

| Quality | Good Example | Bad Example |
|---------|-------------|-------------|
| **Minimal** | One behavior | `test('validates email and domain and whitespace')` |
| **Clear** | Descriptive name | `test('test1')` or `test('it works')` |
| **Real Code** | Tests actual implementation | Tests mock behavior |
| **Focused** | Single assertion (when practical) | Multiple unrelated assertions |

## Why Order Matters

### "I'll write tests after to verify it works"

**Problem**: Tests written after code pass immediately.

**Why This Fails**:
- Might test wrong thing
- Might test implementation, not behavior
- Might miss edge cases you forgot
- You never saw it catch the bug

**Solution**: Test-first forces you to see the test fail, proving it actually tests something.

### "I already manually tested all the edge cases"

**Problem**: Manual testing is ad-hoc and not repeatable.

**Why This Fails**:
- No record of what you tested
- Can't re-run when code changes
- Easy to forget cases under pressure
- "It worked when I tried it" ≠ comprehensive

**Solution**: Automated tests are systematic and run the same way every time.

### "Deleting X hours of work is wasteful"

**Problem**: Sunk cost fallacy.

**Reality**:
- The time is already gone
- Your choice now:
  - Delete and rewrite with TDD (X hours, high confidence)
  - Keep and add tests after (30 min, low confidence, likely bugs)

**Solution**: Working code without real tests is technical debt. Delete it.

### "TDD is dogmatic, being pragmatic means adapting"

**Counter**: TDD IS pragmatic:
- Finds bugs before commit (faster than debugging after)
- Prevents regressions (tests catch breaks immediately)
- Documents behavior (tests show how to use code)
- Enables refactoring (change freely, tests catch breaks)

"Pragmatic" shortcuts = debugging in production = slower.

## Common Rationalizations

| Excuse | Reality | Action |
|--------|---------|--------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. | Write the test. |
| "I'll test after" | Tests passing immediately prove nothing. | Delete code, test first. |
| "Already manually tested" | Ad-hoc ≠ systematic. Can't re-run. | Write automated tests. |
| "Deleting X hours is wasteful" | Sunk cost fallacy. Unverified code is debt. | Delete and restart. |
| "Keep as reference" | You'll adapt it (testing after). | Delete means delete. |
| "Need to explore first" | Fine. Throw away exploration. | Start fresh with TDD. |
| "Test is hard = design unclear" | Listen to the test. Hard to test = hard to use. | Simplify interface. |
| "TDD will slow me down" | TDD faster than debugging later. | Trust the process. |

## Red Flags - STOP and Start Over

If you find yourself:
- ✋ Writing code before test
- ✋ Writing test after implementation
- ✋ Test passes immediately (didn't watch it fail)
- ✋ Can't explain why test failed
- ✋ Planning to add tests "later"
- ✋ Rationalizing "just this once"
- ✋ "I already manually tested it"
- ✋ "Keep as reference" or "adapt existing code"
- ✋ "This is different because..."

**Action**: Delete code. Start over with TDD.

## Bug Fix Example

**Bug**: Empty email is accepted

**Step 1: RED - Write Failing Test**
```typescript
test('rejects empty email', async () => {
    const result = await submitForm({ email: '' });
    expect(result.error).toBe('Email required');
});
```

**Step 2: Verify RED**
```bash
$ npm test
FAIL: expected 'Email required', got undefined
```

**Step 3: GREEN - Implement Fix**
```typescript
function submitForm(data: FormData) {
    if (!data.email?.trim()) {
        return { error: 'Email required' };
    }
    // ... rest of logic
}
```

**Step 4: Verify GREEN**
```bash
$ npm test
PASS: All tests passing
```

**Step 5: REFACTOR**
Extract validation logic if needed:
```typescript
function validateEmail(email: string): string | null {
    if (!email?.trim()) {
        return 'Email required';
    }
    return null;
}

function submitForm(data: FormData) {
    const emailError = validateEmail(data.email);
    if (emailError) {
        return { error: emailError };
    }
    // ... rest of logic
}
```

## Verification Checklist

Before marking work complete:

- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for expected reason (feature missing, not typo)
- [ ] Wrote minimal code to pass each test
- [ ] All tests pass
- [ ] Output is clean (no errors, warnings)
- [ ] Tests use real code (mocks only if unavoidable)
- [ ] Edge cases and error scenarios covered

**Can't check all boxes?** You skipped TDD. Start over.

## When Stuck

| Problem | Solution |
|---------|----------|
| Don't know how to test | Write wished-for API. Write assertion first. Ask for help. |
| Test too complicated | Design is too complicated. Simplify interface. |
| Must mock everything | Code is too coupled. Use dependency injection. |
| Test setup is huge | Extract test helpers. Still complex? Simplify design. |
| Test is slow | Mock expensive operations. Keep unit tests fast. |

## Integration with Bug Fixes

When a bug is found:

1. **Write failing test** reproducing the bug
2. **Verify RED**: Confirm test fails with current code
3. **Fix bug** with minimal code change
4. **Verify GREEN**: Test now passes
5. **Refactor** if needed (improve code quality)

**Never fix bugs without a test.** The test proves the fix works and prevents regressions.

## The Bottom Line

```
Production code exists → test exists and failed first
Otherwise → not TDD → delete and restart
```

**No exceptions** without explicit approval.

TDD is not about having tests. It's about the order:
1. Test first (defines behavior)
2. Watch fail (proves test works)
3. Implement (minimal code to pass)
4. Watch pass (confirms implementation)
5. Refactor (improve quality)

Tests written after are verification, not TDD. They don't drive design and might not test the right thing.

---

**Skill Type**: Testing - Methodology
**Complexity**: Foundational
**Typical Usage**: Activated for all feature implementation and bug fixes
**Performance**: Prevents bugs early, enables confident refactoring, documents behavior
