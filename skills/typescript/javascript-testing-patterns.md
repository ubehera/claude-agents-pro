---
name: javascript-testing-patterns
description: Use when designing reliable JavaScript and TypeScript tests with clear boundaries, deterministic fixtures, and stable async behavior across unit and integration suites.
trigger_keywords: [javascript testing, vitest, jest, testing library, unit test patterns, async test stability, integration testing js]
---

# JavaScript Testing Patterns

Use this skill to improve confidence and reduce flaky tests in JS/TS codebases.

## When to Use This Skill

- Building test suites for services, components, and utilities
- Migrating brittle tests with timing and shared-state failures
- Introducing integration tests around API and data layers
- Standardizing testing practices across teams

## Core Concepts

- **Test behavior, not implementation details**.
- **Use deterministic clocks and fixtures**.
- **Avoid fixed sleeps**; wait on observable conditions.
- **Isolate external dependencies with contract-level mocks**.

## Implementation Patterns

```ts
import { describe, expect, it, vi } from 'vitest';

function createClock(now = new Date('2026-01-01T00:00:00Z')) {
  return { now: () => now };
}

describe('registerUser', () => {
  it('stores user and sends welcome email', async () => {
    const userRepo = { create: vi.fn(async (u) => ({ id: 'u1', ...u })) };
    const emailGateway = { sendWelcome: vi.fn(async () => undefined) };
    const clock = createClock();

    const service = {
      async registerUser(email: string) {
        const user = await userRepo.create({ email, createdAt: clock.now().toISOString() });
        await emailGateway.sendWelcome(user.email);
        return user;
      }
    };

    const user = await service.registerUser('a@example.com');
    expect(userRepo.create).toHaveBeenCalledOnce();
    expect(emailGateway.sendWelcome).toHaveBeenCalledWith('a@example.com');
    expect(user.createdAt).toBe('2026-01-01T00:00:00.000Z');
  });
});
```

## Validation Checklist

- Tests run with `--runInBand` and parallel modes where applicable
- No hard sleeps (`setTimeout`) in assertions
- Fixtures are local to suite and not globally mutated
- Critical paths include both success and failure scenarios
