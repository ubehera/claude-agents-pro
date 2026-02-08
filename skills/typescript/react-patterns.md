---
name: react-patterns
description: Load when user needs React 18+ patterns including hooks, Server Components, Suspense, and state management
trigger_keywords: [react, hooks, useEffect, useState, server component, suspense, context, react 18, react 19, rsc, use client, use server]
---

# React Patterns Skill

Production patterns for React 18+ including hooks composition, Server Components (RSC), Suspense boundaries, and modern state management.

## Overview

React 18+ introduced concurrent features, Server Components, and streaming SSR. This skill covers patterns that leverage these capabilities for performant, maintainable applications.

**When to Use**:
- Building React components with complex state or effects
- Implementing Server Components architecture
- Optimizing rendering performance
- Managing state across component trees

## Core Concepts

### Server vs Client Components

```tsx
// Server Component (default in App Router) — runs on server only
// No 'use client' directive, no hooks, no browser APIs
async function UserProfile({ userId }: { userId: string }) {
  const user = await db.users.findUnique({ where: { id: userId } });
  return (
    <div>
      <h1>{user.name}</h1>
      <UserActions user={user} />  {/* Client component for interactivity */}
    </div>
  );
}

// Client Component — runs on both server (SSR) and client
'use client';
import { useState } from 'react';

function UserActions({ user }: { user: User }) {
  const [following, setFollowing] = useState(false);
  return (
    <button onClick={() => setFollowing(!following)}>
      {following ? 'Unfollow' : 'Follow'} {user.name}
    </button>
  );
}
```

**Decision Rule**: Default to Server Components. Use `'use client'` only when you need:
- `useState`, `useEffect`, or other hooks
- Browser APIs (`window`, `document`, `localStorage`)
- Event handlers (`onClick`, `onChange`)
- Class components with lifecycle methods

### Custom Hooks — Composition Pattern

```tsx
// Extract reusable logic into custom hooks
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Compose hooks for complex behavior
function useSearchResults(query: string) {
  const debouncedQuery = useDebounce(query, 300);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!debouncedQuery) { setResults([]); return; }

    const controller = new AbortController();
    setIsLoading(true);

    fetch(`/api/search?q=${debouncedQuery}`, { signal: controller.signal })
      .then(r => r.json())
      .then(data => { setResults(data); setError(null); })
      .catch(e => { if (e.name !== 'AbortError') setError(e); })
      .finally(() => setIsLoading(false));

    return () => controller.abort();
  }, [debouncedQuery]);

  return { results, isLoading, error };
}
```

### Suspense + Error Boundaries

```tsx
import { Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

function Dashboard() {
  return (
    <div className="grid grid-cols-3 gap-4">
      <ErrorBoundary fallback={<ErrorCard />}>
        <Suspense fallback={<Skeleton className="h-48" />}>
          <RevenueChart />  {/* Async server component */}
        </Suspense>
      </ErrorBoundary>

      <ErrorBoundary fallback={<ErrorCard />}>
        <Suspense fallback={<Skeleton className="h-48" />}>
          <RecentOrders />  {/* Independent data stream */}
        </Suspense>
      </ErrorBoundary>
    </div>
  );
}

// Each section streams independently — first to resolve renders first
```

### State Management Patterns

```tsx
// 1. useReducer for complex state machines
type State = { status: 'idle' | 'loading' | 'error' | 'success'; data?: Data; error?: Error };
type Action =
  | { type: 'FETCH' }
  | { type: 'SUCCESS'; data: Data }
  | { type: 'ERROR'; error: Error };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'FETCH':   return { status: 'loading' };
    case 'SUCCESS': return { status: 'success', data: action.data };
    case 'ERROR':   return { status: 'error', error: action.error };
  }
}

// 2. Context + useReducer for shared state (avoid prop drilling)
const AppContext = createContext<{ state: State; dispatch: Dispatch<Action> } | null>(null);

function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}

// 3. External stores (Zustand) for global client state
import { create } from 'zustand';

const useStore = create<Store>((set) => ({
  items: [],
  addItem: (item) => set((s) => ({ items: [...s.items, item] })),
  removeItem: (id) => set((s) => ({ items: s.items.filter(i => i.id !== id) })),
}));
```

## Performance Patterns

### Memoization
```tsx
// React.memo — skip re-render when props haven't changed
const ExpensiveList = memo(function ExpensiveList({ items }: { items: Item[] }) {
  return items.map(item => <ExpensiveItem key={item.id} item={item} />);
});

// useMemo — cache expensive computations
const sortedItems = useMemo(
  () => items.toSorted((a, b) => a.name.localeCompare(b.name)),
  [items]
);

// useCallback — stable function references for child components
const handleClick = useCallback((id: string) => {
  setSelected(id);
}, []); // No deps → stable reference
```

### Virtualization for Large Lists
```tsx
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
  });

  return (
    <div ref={parentRef} style={{ height: '400px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }}>
        {virtualizer.getVirtualItems().map(virtual => (
          <div key={virtual.key} style={{
            position: 'absolute',
            top: 0,
            transform: `translateY(${virtual.start}px)`,
            height: `${virtual.size}px`,
          }}>
            {items[virtual.index].name}
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Best Practices

1. **Server Components first** — default to RSC, add `'use client'` only when needed
2. **Colocate state** — keep state as close to where it's used as possible
3. **Avoid unnecessary effects** — derive values during render, not in `useEffect`
4. **Cleanup effects** — always return cleanup functions for subscriptions/timers
5. **Stable keys** — use unique IDs, never array indices for dynamic lists
6. **Error boundaries per feature** — isolate failures to prevent full-page crashes

---

**Skill Type**: TypeScript — React
**Complexity**: Moderate
**Typical Usage**: React component architecture, hooks patterns, performance optimization
