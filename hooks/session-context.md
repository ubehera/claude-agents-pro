---
name: session-context
description: Loads project context, recent changes, and active tasks when a new session starts
event: SessionStart
---

# Session Context Hook

Automatically loads relevant project context at session start to reduce cold-start time.

## Context Loading Steps

1. **Git Status**: Check for uncommitted changes, current branch, recent commits
2. **Active Tasks**: Load TodoWrite entries for in-progress work
3. **Project Type**: Detect language/framework from config files (package.json, pyproject.toml, Cargo.toml, etc.)
4. **Recent Changes**: Show files modified in the last 24 hours
5. **Memory Recall**: Search knowledge graph for project-specific decisions and patterns

## Information Gathered

```yaml
Project Context:
  - Project name and description (from README or package.json)
  - Primary language and framework
  - Active git branch and recent commit messages (last 5)
  - Uncommitted changes summary

Active Work:
  - In-progress TodoWrite tasks
  - Open PRs on current branch
  - Recent CI/CD status

Environment:
  - Node/Python/Go/Rust version
  - Available test runners
  - Configured linters and formatters
```

## Output Format

Provide a brief context summary (3-5 lines) at session start:

```
📍 Project: my-app (Next.js 14 + TypeScript)
🌿 Branch: feature/user-auth (3 uncommitted files)
📋 Active: "Implement JWT authentication" (in_progress)
🔄 Last commit: "feat: add login form component" (2h ago)
```

## Notes
- Keep context loading fast (<2 seconds)
- Don't overwhelm with information — summarize concisely
- Skip if no git repository detected
- Respect .gitignore for file discovery
