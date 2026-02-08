---
name: typescript-advanced-types
description: Use when building or refactoring complex TypeScript type systems with generics, conditional types, inference control, utility types, and compile-time guarantees.
trigger_keywords: [typescript advanced types, generics, conditional types, mapped types, infer, utility types, discriminated union, satisfies, type-level programming]
---

# TypeScript Advanced Types

Use this skill to build safer public APIs and reduce runtime checks by pushing constraints into the type system.

## When to Use This Skill

- Designing reusable libraries and shared packages
- Refactoring `any` and broad union usage into precise types
- Building discriminated unions for state and domain workflows
- Creating type-safe API clients and schema-derived types

## Core Concepts

- **Model domain states explicitly** with discriminated unions.
- **Prefer inference-friendly APIs** over deeply generic call signatures.
- **Use `satisfies` for constraint checking** without widening literals.
- **Make impossible states unrepresentable** using branded and tagged types.

## Implementation Patterns

```ts
// 1) Domain-safe state machine
export type OrderState =
  | { kind: 'draft' }
  | { kind: 'submitted'; submittedAt: string }
  | { kind: 'filled'; fillPrice: number }
  | { kind: 'cancelled'; reason: string };

export function renderState(state: OrderState): string {
  switch (state.kind) {
    case 'draft':
      return 'Draft';
    case 'submitted':
      return `Submitted ${state.submittedAt}`;
    case 'filled':
      return `Filled @ ${state.fillPrice}`;
    case 'cancelled':
      return `Cancelled: ${state.reason}`;
    default: {
      const neverState: never = state;
      return neverState;
    }
  }
}

// 2) Narrow literal config without widening
const endpoints = {
  user: '/api/users',
  trade: '/api/trades'
} as const satisfies Record<string, `/${string}`>;

type EndpointName = keyof typeof endpoints;
type EndpointPath = (typeof endpoints)[EndpointName];

// 3) Conditional utility for API envelopes
export type ApiResult<T, E = { message: string }> =
  | { ok: true; data: T }
  | { ok: false; error: E };
```

## Validation Checklist

- `tsc --noEmit` is clean with strict flags enabled
- No unbounded `any` in new code paths
- Public exported types have stable names and docs
- Exhaustive checks are enforced for tagged unions
