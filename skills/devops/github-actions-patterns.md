---
name: github-actions-patterns
description: Load when user needs GitHub Actions CI/CD patterns including reusable workflows, matrix builds, and deployment
trigger_keywords: [github actions, workflow, ci/cd pipeline, yaml workflow, reusable workflow, matrix build, github ci, actions runner, workflow dispatch]
---

# GitHub Actions Patterns Skill

Production CI/CD patterns for GitHub Actions including reusable workflows, matrix builds, caching, secrets management, and deployment strategies.

## Overview

GitHub Actions provides native CI/CD integrated with GitHub repositories. This skill covers patterns beyond basic workflows.

**When to Use**:
- Setting up CI/CD pipelines for new projects
- Optimizing existing workflow performance
- Implementing deployment strategies (staging → production)
- Creating reusable workflows for org-wide standards

## Core Patterns

### Reusable Workflow (Org Standard)

```yaml
# .github/workflows/reusable-ci.yml
name: Reusable CI Pipeline

on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      test-command:
        type: string
        default: 'npm test'
    secrets:
      NPM_TOKEN:
        required: false

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4
      - uses: actions/setup-node@1d0ff469b7ec7b3cb9d8673fde0c81c44821de2a # v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'

      - run: npm ci
      - run: npm run lint
      - run: npm run type-check
      - run: ${{ inputs.test-command }}
      - run: npm run build
```

### Caller Workflow

```yaml
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  ci:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      node-version: '20'
      test-command: 'npm test -- --coverage'
    secrets: inherit
```

### Matrix Build (Multi-Version Testing)

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        node: [18, 20, 22]
        exclude:
          - os: macos-latest
            node: 18
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - uses: actions/setup-node@1d0ff469b7ec7b3cb9d8673fde0c81c44821de2a
        with:
          node-version: ${{ matrix.node }}
      - run: npm ci && npm test
```

### Caching Dependencies

```yaml
# Node.js with npm cache
- uses: actions/setup-node@1d0ff469b7ec7b3cb9d8673fde0c81c44821de2a
  with:
    node-version: '20'
    cache: 'npm'  # Built-in npm cache

# Custom cache for larger builds
- uses: actions/cache@5a3ec84eff668545956fd18022155c47e93e2684 # v4
  with:
    path: |
      ~/.npm
      .next/cache
    key: ${{ runner.os }}-nextjs-${{ hashFiles('package-lock.json') }}-${{ hashFiles('**/*.ts') }}
    restore-keys: |
      ${{ runner.os }}-nextjs-${{ hashFiles('package-lock.json') }}-
      ${{ runner.os }}-nextjs-
```

### Deployment with Environments

```yaml
jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - run: ./deploy.sh staging
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.com
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - run: ./deploy.sh production
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

### Security: Pin Actions to SHA

```yaml
# ❌ Mutable tag — vulnerable to supply chain attacks
- uses: actions/checkout@v4

# ✅ Pinned to specific commit SHA
- uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4
```

### Conditional Jobs

```yaml
jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      backend: ${{ steps.filter.outputs.backend }}
      frontend: ${{ steps.filter.outputs.frontend }}
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - uses: dorny/paths-filter@de90cc6fb38fc0963ad72b210f1f284cd68cea36
        id: filter
        with:
          filters: |
            backend:
              - 'server/**'
            frontend:
              - 'client/**'

  test-backend:
    needs: changes
    if: ${{ needs.changes.outputs.backend == 'true' }}
    runs-on: ubuntu-latest
    steps:
      - run: echo "Running backend tests"

  test-frontend:
    needs: changes
    if: ${{ needs.changes.outputs.frontend == 'true' }}
    runs-on: ubuntu-latest
    steps:
      - run: echo "Running frontend tests"
```

## Best Practices

1. **Pin actions to SHAs** — prevent supply chain attacks from mutable tags
2. **Use `workflow_call`** — share CI logic across repos via reusable workflows
3. **Cache aggressively** — npm, pip, Docker layers, build outputs
4. **Path filtering** — only run jobs when relevant files change
5. **Environment protection** — require approvals for production deployments
6. **Fail fast: false** — for matrix builds, let all combinations complete

---

**Skill Type**: DevOps — CI/CD
**Complexity**: Moderate
**Typical Usage**: GitHub Actions pipeline setup, workflow optimization, deployment automation
