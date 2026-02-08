---
name: type-system-advanced
description: Load when user needs advanced TypeScript type patterns including branded types, conditional types, and type-level programming
trigger_keywords: [branded type, conditional type, template literal type, mapped type, infer, satisfies, discriminated union, type guard, type narrowing, generic constraint]
---

# Advanced TypeScript Type System Skill

Type-level programming patterns for building bulletproof TypeScript applications with compile-time safety guarantees.

## Overview

TypeScript's type system is Turing-complete, enabling patterns that catch entire classes of bugs at compile time. This skill covers advanced patterns beyond basic generics.

**When to Use**:
- Preventing invalid state at the type level
- Building type-safe APIs and libraries
- Encoding business rules in the type system
- Creating self-documenting type contracts

## Core Patterns

### Branded Types (Nominal Typing)

```typescript
// Prevent mixing up semantically different values of the same base type
type Brand<T, B extends string> = T & { readonly __brand: B };

type UserId = Brand<string, 'UserId'>;
type OrderId = Brand<string, 'OrderId'>;
type Email = Brand<string, 'Email'>;
type USD = Brand<number, 'USD'>;
type EUR = Brand<number, 'EUR'>;

// Constructor functions with runtime validation
function UserId(value: string): UserId {
  if (!value.startsWith('usr_')) throw new Error('Invalid user ID');
  return value as UserId;
}

function Email(value: string): Email {
  if (!/.+@.+\..+/.test(value)) throw new Error('Invalid email');
  return value as Email;
}

// Compile-time safety:
function getUser(id: UserId): Promise<User> { /* ... */ }

const userId = UserId('usr_123');
const orderId = OrderId('ord_456');

getUser(userId);   // ✅ Correct type
getUser(orderId);  // ❌ Compile error: OrderId is not assignable to UserId
getUser('raw');    // ❌ Compile error: string is not assignable to UserId
```

### Discriminated Unions (State Machines)

```typescript
// Model all valid states — make invalid states unrepresentable
type AsyncState<T, E = Error> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: E };

// Exhaustive pattern matching
function renderState<T>(state: AsyncState<T>): string {
  switch (state.status) {
    case 'idle':    return 'Ready';
    case 'loading': return 'Loading...';
    case 'success': return `Data: ${state.data}`; // TS knows `data` exists
    case 'error':   return `Error: ${state.error.message}`; // TS knows `error` exists
    // No default needed — TS verifies exhaustiveness
  }
}

// Compile-time exhaustiveness check
function assertNever(x: never): never {
  throw new Error(`Unexpected value: ${x}`);
}
```

### Conditional Types

```typescript
// Extract return type of async functions
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;
type Result = UnwrapPromise<Promise<string>>; // string

// Make specific fields optional
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
type CreateUserInput = PartialBy<User, 'id' | 'createdAt'>;

// Deep readonly
type DeepReadonly<T> = T extends (infer U)[]
  ? readonly DeepReadonly<U>[]
  : T extends object
  ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

// Filter union members
type ExtractByStatus<T, S> = T extends { status: S } ? T : never;
type SuccessState = ExtractByStatus<AsyncState<User>, 'success'>;
// { status: 'success'; data: User }
```

### Template Literal Types

```typescript
// Type-safe event names
type EventName = `on${Capitalize<'click' | 'change' | 'submit'>}`;
// 'onClick' | 'onChange' | 'onSubmit'

// Type-safe route params
type ExtractParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractParams<Rest>
    : T extends `${string}:${infer Param}`
    ? Param
    : never;

type Params = ExtractParams<'/users/:userId/posts/:postId'>;
// 'userId' | 'postId'

// Type-safe CSS properties
type CSSUnit = 'px' | 'rem' | 'em' | '%' | 'vh' | 'vw';
type CSSValue = `${number}${CSSUnit}` | 'auto' | '0';
```

### Builder Pattern with Type Accumulation

```typescript
// Type-safe query builder that tracks selected columns
class QueryBuilder<
  TTable extends string,
  TSelected extends string = never,
> {
  private _table: TTable;
  private _columns: string[] = [];
  private _conditions: string[] = [];

  constructor(table: TTable) { this._table = table; }

  select<C extends string>(
    ...columns: C[]
  ): QueryBuilder<TTable, TSelected | C> {
    this._columns.push(...columns);
    return this as any;
  }

  where(condition: string): this {
    this._conditions.push(condition);
    return this;
  }

  build(): { table: TTable; columns: TSelected[] } {
    return { table: this._table, columns: this._columns as any };
  }
}

// Usage — type tracks selected columns
const query = new QueryBuilder('users')
  .select('id', 'name', 'email')
  .where('active = true')
  .build();
// Type: { table: 'users'; columns: ('id' | 'name' | 'email')[] }
```

### `satisfies` Operator (TypeScript 4.9+)

```typescript
// Validate type while preserving literal types
const config = {
  apiUrl: 'https://api.example.com',
  timeout: 5000,
  retries: 3,
} satisfies Record<string, string | number>;

// config.apiUrl is string (not string | number)
// config.timeout is number (not string | number)
// But it's validated against Record<string, string | number>

// Theme with exact color types
type Theme = Record<string, { bg: string; fg: string }>;

const theme = {
  primary:   { bg: '#3b82f6', fg: '#ffffff' },
  secondary: { bg: '#6b7280', fg: '#ffffff' },
  danger:    { bg: '#ef4444', fg: '#ffffff' },
} satisfies Theme;

// theme.primary.bg is '#3b82f6' (literal), not just string
```

### Type-Safe Event Emitter

```typescript
type EventMap = {
  'user:created': { userId: string; email: string };
  'user:deleted': { userId: string };
  'order:placed': { orderId: string; total: number };
};

class TypedEmitter<T extends Record<string, unknown>> {
  private listeners = new Map<string, Set<Function>>();

  on<K extends keyof T & string>(
    event: K,
    handler: (payload: T[K]) => void
  ): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof T & string>(event: K, payload: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(payload));
  }
}

const bus = new TypedEmitter<EventMap>();
bus.on('user:created', ({ userId, email }) => { /* typed! */ });
bus.emit('user:created', { userId: '1', email: 'a@b.com' }); // ✅
bus.emit('user:created', { userId: '1' }); // ❌ missing email
```

## Best Practices

1. **Branded types** for IDs, money, measurements — prevents mixing semantically different values
2. **Discriminated unions** for state machines — make invalid states unrepresentable
3. **`satisfies`** over type annotations — preserves literal types while validating
4. **`const` assertions** for literal tuples and objects: `as const`
5. **Exhaustiveness checks** with `never` in switch defaults
6. **Avoid `any`** — use `unknown` with type guards instead

---

**Skill Type**: TypeScript — Type System
**Complexity**: Complex
**Typical Usage**: Type-safe API design, domain modeling, library development
