---
name: monorepo-management
description: Master monorepo management with Turborepo, Nx, and pnpm workspaces for efficient multi-package repositories with optimized builds. Use when setting up monorepos or managing shared dependencies.
---

# Monorepo Management

## Why Monorepos?

**Advantages:** Shared code, atomic commits, consistent tooling, easier refactoring
**Challenges:** Build performance, CI/CD complexity, large Git repo

## Turborepo Setup

```bash
npx create-turbo@latest my-monorepo
# Structure: apps/, packages/, turbo.json
```

```json
// turbo.json
{
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "test": { "dependsOn": ["build"], "outputs": ["coverage/**"] },
    "lint": { "outputs": [] },
    "dev": { "cache": false, "persistent": true }
  }
}
```

## pnpm Workspaces

```yaml
# pnpm-workspace.yaml
packages:
  - 'apps/*'
  - 'packages/*'
```

```bash
pnpm add react --filter @repo/ui      # Install in specific package
pnpm add @repo/ui --filter web         # Add workspace dependency
pnpm --filter web dev                  # Run script in package
pnpm -r build                          # Run in all packages
```

## Code Sharing Patterns

```typescript
// packages/ui/src/button.tsx
export function Button({ variant = 'primary', children }) {
  return <button className={`btn btn-${variant}`}>{children}</button>;
}

// apps/web/src/app.tsx
import { Button } from '@repo/ui';
```

## Shared TypeScript Config

```json
// packages/tsconfig/base.json
{
  "compilerOptions": {
    "strict": true,
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}

// apps/web/tsconfig.json
{ "extends": "@repo/tsconfig/base.json" }
```

## CI/CD (GitHub Actions)

```yaml
- uses: pnpm/action-setup@v2
- run: pnpm install --frozen-lockfile
- run: pnpm turbo run build test lint
```

## Best Practices

1. Lock dependency versions across workspace
2. Centralize ESLint, TypeScript, Prettier configs
3. Keep dependency graph acyclic
4. Configure cache inputs/outputs correctly
5. Share types between frontend/backend
6. Use changesets for versioning

## Common Pitfalls

- Circular dependencies
- Phantom dependencies (using deps not in package.json)
- Incorrect cache inputs
- Over-sharing or under-sharing code
