---
name: flaky-test-elimination
description: Load when debugging flaky tests, race conditions, or timing-dependent test failures - replaces arbitrary delays with condition-based waiting
trigger_keywords: [flaky test, race condition, timing issue, test timeout, intermittent failure, test reliability, waitfor, async testing]
---

# Flaky Test Elimination

Systematic patterns for eliminating flaky tests by replacing arbitrary timeouts with condition-based waiting and proper synchronization.

## Overview

Flaky tests pass inconsistently due to race conditions, timing dependencies, or environmental variability. The root cause is usually guessing at timing with arbitrary delays instead of waiting for actual conditions.

**Core Principle**: Wait for the actual condition you care about, not a guess about how long it takes.

**When to Use**:
- Tests fail intermittently (pass locally, fail in CI)
- Tests use `setTimeout`, `sleep`, or fixed delays
- Tests depend on async operations completing
- Tests fail under load or in parallel execution
- Tests timeout in CI but pass locally

## Core Concepts

- **Condition-Based Waiting**: Replace arbitrary delays with polling loops that check actual state - flakiness comes from guessing timing instead of waiting for conditions
- **Race Condition Prevention**: Ensure async operations complete with proper `await` before assertions - missing awaits cause intermittent failures
- **Environment Independence**: Tests must pass consistently across fast dev machines and slow CI environments - design for worst-case timing
- **Deterministic State**: Never depend on external state, test order, or system time - each test should produce identical results on every run
- **Debug-First Diagnosis**: Add logging to wait functions, identify the actual timing dependency, then replace the guess with a condition check

## Root Causes of Flaky Tests

### 1. Arbitrary Timeouts

```typescript
// ❌ FLAKY: Guessing at timing
await new Promise(r => setTimeout(r, 50));
const result = getResult();
expect(result).toBeDefined();
```

**Problem**: 50ms might work on fast machine, fail under load or in CI.

**Fix**: Wait for actual condition
```typescript
// ✅ RELIABLE: Wait for condition
await waitFor(() => getResult() !== undefined);
const result = getResult();
expect(result).toBeDefined();
```

### 2. Race Conditions

```typescript
// ❌ FLAKY: Race between API call and assertion
async function test() {
    fetchData();  // Async, no await
    expect(data).toBeDefined();  // Might run before fetch completes
}
```

**Fix**: Proper synchronization
```typescript
// ✅ RELIABLE: Wait for async operation
async function test() {
    await fetchData();
    expect(data).toBeDefined();
}
```

### 3. Timing Dependencies

```python
# ❌ FLAKY: Assumes operation completes in 1 second
time.sleep(1)
assert file_exists("output.txt")
```

**Fix**: Poll for condition
```python
# ✅ RELIABLE: Wait for actual file existence
def wait_for_file(path, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    raise TimeoutError(f"File {path} not created within {timeout}s")

wait_for_file("output.txt")
assert file_exists("output.txt")
```

## Condition-Based Waiting Pattern

### Generic Wait Implementation

```typescript
async function waitFor<T>(
    condition: () => T | undefined | null | false,
    options: {
        description: string;
        timeout?: number;
        interval?: number;
    }
): Promise<T> {
    const { description, timeout = 5000, interval = 10 } = options;
    const startTime = Date.now();

    while (true) {
        const result = condition();
        if (result) return result;

        if (Date.now() - startTime > timeout) {
            throw new Error(
                `Timeout waiting for ${description} after ${timeout}ms`
            );
        }

        await new Promise(r => setTimeout(r, interval));
    }
}
```

### Usage Examples

```typescript
// Wait for element to appear
await waitFor(
    () => document.querySelector('[data-testid="result"]'),
    { description: 'result element' }
);

// Wait for state change
await waitFor(
    () => machine.state === 'ready',
    { description: 'machine ready state' }
);

// Wait for array to populate
await waitFor(
    () => items.length >= 5 ? items : null,
    { description: 'at least 5 items' }
);

// Wait for file to exist
await waitFor(
    () => fs.existsSync(filePath) ? filePath : null,
    { description: 'output file creation' }
);

// Complex condition
await waitFor(
    () => obj.ready && obj.value > 10 ? obj : null,
    { description: 'object ready with value > 10' }
);
```

## Framework-Specific Patterns

### Playwright

```typescript
// ❌ FLAKY: Fixed timeout
await page.waitForTimeout(3000);

// ✅ RELIABLE: Wait for network idle
await page.waitForLoadState('networkidle');

// ✅ RELIABLE: Wait for URL change
await page.waitForURL('/dashboard');

// ✅ RELIABLE: Wait for selector
await page.waitForSelector('[data-testid="profile"]');

// ✅ BEST: Auto-waiting assertions
await expect(page.getByText('Welcome')).toBeVisible();
await expect(page.getByRole('button', { name: 'Submit' })).toBeEnabled();

// Wait for API response
const responsePromise = page.waitForResponse(
    response => response.url().includes('/api/data') &&
                response.status() === 200
);
await page.getByRole('button', { name: 'Load' }).click();
const response = await responsePromise;
```

### Cypress

```typescript
// ❌ FLAKY: Fixed wait
cy.wait(1000);

// ✅ RELIABLE: Wait for element
cy.get('[data-testid="result"]', { timeout: 10000 }).should('exist');

// ✅ RELIABLE: Wait for API call
cy.intercept('GET', '/api/users').as('getUsers');
cy.visit('/users');
cy.wait('@getUsers');

// ✅ RELIABLE: Retry assertions
cy.get('[data-testid="count"]').should('have.text', '5');

// Custom retry logic
cy.waitUntil(() =>
    cy.window().then(win => win.dataReady === true),
    { timeout: 5000, interval: 100 }
);
```

### React Testing Library

```typescript
import { waitFor, screen } from '@testing-library/react';

// ❌ FLAKY: Query immediately
const element = screen.getByText('Loaded');

// ✅ RELIABLE: Wait for element
await waitFor(() => {
    expect(screen.getByText('Loaded')).toBeInTheDocument();
});

// ✅ RELIABLE: findBy queries (auto-wait)
const element = await screen.findByText('Loaded');

// ✅ RELIABLE: Wait for state change
await waitFor(() => {
    expect(screen.getByTestId('status')).toHaveTextContent('ready');
}, { timeout: 3000 });

// Wait for element to disappear
await waitFor(() => {
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
});
```

### Jest/Vitest

```typescript
// ❌ FLAKY: Immediate assertion
expect(asyncValue).toBe('loaded');

// ✅ RELIABLE: Wait for condition
await vi.waitFor(() => {
    expect(asyncValue).toBe('loaded');
}, { timeout: 5000 });

// ✅ RELIABLE: Poll until condition met
await vi.waitUntil(
    () => store.getState().data !== null,
    { timeout: 3000, interval: 100 }
);
```

## Domain-Specific Wait Helpers

### Event-Based Waiting

```typescript
async function waitForEvent<T>(
    emitter: EventEmitter,
    eventName: string,
    options: { timeout?: number } = {}
): Promise<T> {
    const { timeout = 5000 } = options;

    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            emitter.off(eventName, handler);
            reject(new Error(`Timeout waiting for event: ${eventName}`));
        }, timeout);

        const handler = (data: T) => {
            clearTimeout(timer);
            resolve(data);
        };

        emitter.once(eventName, handler);
    });
}

// Usage
const data = await waitForEvent(manager, 'TOOL_STARTED', { timeout: 3000 });
expect(data.toolName).toBe('calculator');
```

### Count-Based Waiting

```typescript
async function waitForEventCount(
    emitter: EventEmitter,
    eventName: string,
    count: number,
    options: { timeout?: number } = {}
): Promise<void> {
    const { timeout = 5000 } = options;
    let receivedCount = 0;

    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            emitter.off(eventName, handler);
            reject(new Error(
                `Timeout: got ${receivedCount}/${count} ${eventName} events`
            ));
        }, timeout);

        const handler = () => {
            receivedCount++;
            if (receivedCount >= count) {
                clearTimeout(timer);
                emitter.off(eventName, handler);
                resolve();
            }
        };

        emitter.on(eventName, handler);
    });
}

// Usage
await waitForEventCount(manager, 'TOOL_OUTPUT', 3);
```

### Predicate-Based Waiting

```typescript
async function waitForEventMatch<T>(
    emitter: EventEmitter,
    eventName: string,
    predicate: (data: T) => boolean,
    options: { timeout?: number } = {}
): Promise<T> {
    const { timeout = 5000 } = options;

    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            emitter.off(eventName, handler);
            reject(new Error(
                `Timeout waiting for ${eventName} matching predicate`
            ));
        }, timeout);

        const handler = (data: T) => {
            if (predicate(data)) {
                clearTimeout(timer);
                emitter.off(eventName, handler);
                resolve(data);
            }
        };

        emitter.on(eventName, handler);
    });
}

// Usage
const errorEvent = await waitForEventMatch(
    manager,
    'TOOL_ERROR',
    (data) => data.errorCode === 'TIMEOUT'
);
```

## When Fixed Timeouts ARE Correct

Sometimes you actually need to test timing behavior:

```typescript
// Testing debounce behavior (100ms debounce)
await waitForEvent(manager, 'SEARCH_STARTED');  // First: wait for trigger
await new Promise(r => setTimeout(r, 150));      // Then: wait for debounce
// 150ms > 100ms debounce - documented and justified

// Testing partial output during streaming
await waitForEvent(manager, 'TOOL_STARTED');
await new Promise(r => setTimeout(r, 200));  // Tool ticks every 100ms
// 200ms = 2 ticks - documented and justified
```

**Requirements for Fixed Timeouts**:
1. First wait for triggering condition
2. Based on known timing (not guessing)
3. Comment explaining WHY timeout is needed
4. Document the timing behavior being tested

## Common Patterns

### Pattern 1: Wait Then Assert

```typescript
// Wait for condition
await waitFor(() => elements.length > 0, {
    description: 'elements to populate'
});

// Assert on stable state
expect(elements.length).toBe(3);
expect(elements[0].text).toBe('Item 1');
```

### Pattern 2: Poll with Interval

```python
def wait_for_condition(check, timeout=5, interval=0.1):
    """Poll condition every interval until timeout"""
    start = time.time()
    while time.time() - start < timeout:
        if check():
            return True
        time.sleep(interval)
    return False

# Usage
assert wait_for_condition(
    lambda: len(get_items()) >= 5,
    timeout=3
)
```

### Pattern 3: Multiple Conditions

```typescript
// Wait for all conditions
await Promise.all([
    page.waitForURL('/success'),
    page.waitForLoadState('networkidle'),
    waitFor(() => store.getState().complete === true, {
        description: 'store completion'
    }),
]);
```

## Debugging Flaky Tests

### Step 1: Identify Timing Dependencies

Look for:
- `setTimeout`, `sleep`, fixed delays
- Async operations without `await`
- Immediate assertions on async state
- Tests that fail in CI but pass locally

### Step 2: Add Logging

```typescript
async function waitFor<T>(
    condition: () => T | undefined | null | false,
    options: { description: string; timeout?: number; debug?: boolean }
): Promise<T> {
    const { description, timeout = 5000, debug = false } = options;
    const startTime = Date.now();

    while (true) {
        const result = condition();

        if (debug) {
            console.log(`[waitFor] ${description}: ${result ? 'MET' : 'waiting...'}`);
        }

        if (result) {
            if (debug) {
                console.log(`[waitFor] ${description}: completed in ${Date.now() - startTime}ms`);
            }
            return result;
        }

        if (Date.now() - startTime > timeout) {
            throw new Error(
                `Timeout waiting for ${description} after ${timeout}ms`
            );
        }

        await new Promise(r => setTimeout(r, 10));
    }
}
```

### Step 3: Replace Fixed Delays

```typescript
// Before: Flaky
await new Promise(r => setTimeout(r, 100));
const result = getResult();

// After: Reliable
const result = await waitFor(
    () => getResult(),
    { description: 'result availability', debug: true }
);
```

### Step 4: Verify Fix

- Run test 100 times locally
- Run test in parallel
- Run test in CI environment
- Monitor for failures over time

## Best Practices

1. **Default to Condition Waiting**: Use `waitFor` instead of fixed timeouts
2. **Poll Frequency**: 10-50ms interval is usually sufficient
3. **Timeout Values**: 3-5 seconds for most operations, 30+ for E2E
4. **Clear Error Messages**: Include context in timeout errors
5. **Fresh Data**: Call getters inside condition, don't cache
6. **Avoid Over-Polling**: Don't poll faster than system can respond

## Common Mistakes

❌ **Polling Too Fast**: `setTimeout(check, 1)` wastes CPU
✅ **Fix**: Poll every 10-50ms

❌ **No Timeout**: Infinite loop if condition never met
✅ **Fix**: Always include timeout with clear error

❌ **Stale Data**: Cache state before loop
✅ **Fix**: Call getter inside loop for fresh data

❌ **Wrong Condition**: Waiting for proxy instead of real state
✅ **Fix**: Wait for actual state change, not side effects

## Quality Standards

- **Flakiness Rate**: Target <0.1% (ideally 0%)
- **Timeout Errors**: Should clearly indicate what was being waited for
- **Test Speed**: Condition-based waiting often faster than fixed delays
- **CI Reliability**: Tests should pass consistently across environments

---

**Skill Type**: Testing - Reliability
**Complexity**: Moderate
**Typical Usage**: Activated when debugging flaky tests or implementing async test patterns
**Performance**: Improves test reliability from 60% → 100% pass rate while reducing execution time
