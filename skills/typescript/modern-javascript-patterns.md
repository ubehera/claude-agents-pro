---
name: modern-javascript-patterns
description: Use when implementing modern JavaScript in Node.js or browser runtimes using ES2020+ patterns, module boundaries, async control flow, and maintainable composition.
trigger_keywords: [modern javascript, esm, async await, promise patterns, module boundaries, composition, es2020, javascript architecture]
---

# Modern JavaScript Patterns

Use this skill to standardize clean JavaScript patterns that scale across services and frontend apps.

## When to Use This Skill

- Building JS modules with clear interfaces
- Replacing callback-heavy logic with structured async flows
- Defining cross-runtime patterns (Node and browser)
- Improving readability and reliability of existing JS code

## Core Concepts

- **Use ESM consistently** and avoid mixed module systems.
- **Design pure core logic** and isolate side effects at boundaries.
- **Prefer composition over inheritance** for reusable behavior.
- **Wrap IO with small adapters** for easier testing.

## Implementation Patterns

```js
// Boundary-first service pattern
export function createUserService({ userRepo, emailGateway, clock }) {
  async function registerUser(input) {
    const now = clock.now();
    const created = await userRepo.create({ ...input, createdAt: now.toISOString() });
    await emailGateway.sendWelcome(created.email);
    return created;
  }

  return { registerUser };
}

// Task fanout with controlled concurrency
export async function mapWithLimit(items, limit, worker) {
  const results = [];
  const queue = [...items];

  async function runWorker() {
    while (queue.length) {
      const item = queue.shift();
      if (item === undefined) break;
      results.push(await worker(item));
    }
  }

  await Promise.all(Array.from({ length: limit }, () => runWorker()));
  return results;
}
```

## Validation Checklist

- Imports/exports use one module style
- Async flows include timeout/cancellation strategy
- Side effects are wrapped behind interfaces
- Shared utilities avoid framework-specific coupling
