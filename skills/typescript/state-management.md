---
name: state-management
description: Load when user needs state management patterns for React apps including Zustand, Redux Toolkit, TanStack Query, and Jotai
trigger_keywords: [state management, zustand, redux, redux toolkit, tanstack query, react query, jotai, recoil, global state, server state, client state]
---

# State Management Skill

Modern state management patterns for React applications. Covers the right tool for each category of state.

## Overview

State management in React 18+ has shifted from monolithic stores to purpose-specific solutions. The key insight: **server state** and **client state** are fundamentally different problems requiring different tools.

**When to Use**:
- Choosing a state management approach for a new project
- Migrating from Redux to modern alternatives
- Managing server state (API data) vs client state (UI state)
- Implementing optimistic updates and cache management

## State Categories

| Category | Examples | Best Tool |
|----------|----------|-----------|
| **Server State** | API data, user profile, posts list | TanStack Query |
| **Client State** | Theme, sidebar open, selected tab | Zustand or Context |
| **Form State** | Input values, validation errors | React Hook Form |
| **URL State** | Filters, pagination, search params | nuqs or useSearchParams |
| **Derived State** | Computed from other state | `useMemo` or selectors |

## TanStack Query (Server State)

```tsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Fetch with automatic caching, refetching, and deduplication
function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => fetch('/api/users').then(r => r.json()),
    staleTime: 5 * 60 * 1000,  // 5 min before refetch
    gcTime: 30 * 60 * 1000,    // 30 min in cache
  });
}

// Mutation with optimistic update
function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (newUser: CreateUserInput) =>
      fetch('/api/users', {
        method: 'POST',
        body: JSON.stringify(newUser),
        headers: { 'Content-Type': 'application/json' },
      }).then(r => r.json()),

    // Optimistic update
    onMutate: async (newUser) => {
      await queryClient.cancelQueries({ queryKey: ['users'] });
      const previous = queryClient.getQueryData<User[]>(['users']);
      queryClient.setQueryData<User[]>(['users'], (old) => [
        ...(old ?? []),
        { ...newUser, id: 'temp-id' } as User,
      ]);
      return { previous };
    },

    onError: (_err, _newUser, context) => {
      queryClient.setQueryData(['users'], context?.previous);
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

// Usage in component
function UserList() {
  const { data: users, isLoading, error } = useUsers();
  const createUser = useCreateUser();

  if (isLoading) return <Skeleton />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div>
      {users.map(user => <UserCard key={user.id} user={user} />)}
      <button onClick={() => createUser.mutate({ name: 'New User' })}>
        {createUser.isPending ? 'Creating...' : 'Add User'}
      </button>
    </div>
  );
}
```

## Zustand (Client State)

```tsx
import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

// Simple store — no boilerplate
interface AppStore {
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  selectedIds: Set<string>;
  toggleTheme: () => void;
  toggleSidebar: () => void;
  toggleSelection: (id: string) => void;
  clearSelection: () => void;
}

const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (set) => ({
        theme: 'light',
        sidebarOpen: true,
        selectedIds: new Set(),

        toggleTheme: () => set((s) => ({
          theme: s.theme === 'light' ? 'dark' : 'light',
        })),

        toggleSidebar: () => set((s) => ({
          sidebarOpen: !s.sidebarOpen,
        })),

        toggleSelection: (id) => set((s) => {
          const next = new Set(s.selectedIds);
          next.has(id) ? next.delete(id) : next.add(id);
          return { selectedIds: next };
        }),

        clearSelection: () => set({ selectedIds: new Set() }),
      }),
      { name: 'app-store' } // localStorage key
    )
  )
);

// Usage — components only re-render when their slice changes
function ThemeToggle() {
  const theme = useAppStore((s) => s.theme);         // Only re-renders on theme change
  const toggle = useAppStore((s) => s.toggleTheme);
  return <button onClick={toggle}>{theme}</button>;
}

function Sidebar() {
  const open = useAppStore((s) => s.sidebarOpen);    // Only re-renders on sidebar change
  return open ? <nav>...</nav> : null;
}
```

## When to Use What

```
Need to fetch/cache API data?
  → TanStack Query (handles caching, deduplication, refetching)

Need global UI state (theme, sidebar, modals)?
  → Zustand (simple, performant, no boilerplate)

Need state shared between 2-3 nearby components?
  → React Context + useReducer (no extra dependency)

Need complex form state with validation?
  → React Hook Form + Zod

Need URL-synced state (filters, pagination)?
  → nuqs or useSearchParams

Need state in a single component?
  → useState / useReducer (don't reach for a library)
```

## Anti-Patterns

```
❌ Putting API data in Redux/Zustand (use TanStack Query)
❌ Global state for local concerns (use useState)
❌ Context for frequently changing values (causes re-render cascade)
❌ Multiple sources of truth for the same data
❌ Syncing state between stores manually
```

## Best Practices

1. **Server state ≠ client state** — use different tools for each
2. **Colocate state** — keep it as close to usage as possible
3. **Derive, don't sync** — compute values from source of truth
4. **Selector pattern** — subscribe to slices, not entire stores
5. **URL as state** — filters, pagination, and search belong in the URL

---

**Skill Type**: TypeScript — State Management
**Complexity**: Moderate
**Typical Usage**: React state architecture decisions, TanStack Query setup, Zustand store design
