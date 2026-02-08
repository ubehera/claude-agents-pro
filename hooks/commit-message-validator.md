---
name: commit-message-validator
description: Validates git commit messages follow conventional commit format before commits are created
event: PreToolUse
tools: ["Bash"]
---

# Commit Message Validator Hook

Enforces conventional commit format for all git commits to maintain clean, parseable commit history.

## Validation Rules

### Conventional Commit Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Valid Types
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation changes
- `style` — Code style (formatting, missing semi-colons)
- `refactor` — Code refactoring (no feature/fix)
- `perf` — Performance improvement
- `test` — Adding or updating tests
- `chore` — Maintenance tasks
- `ci` — CI/CD changes
- `build` — Build system changes
- `revert` — Reverting a previous commit

### Rules
1. **Type required**: Must start with a valid type
2. **Description required**: Non-empty description after colon
3. **Lowercase type**: Type must be lowercase
4. **No period**: Description should not end with a period
5. **Max length**: Subject line ≤72 characters
6. **Imperative mood**: Description should use imperative tense ("add" not "added")

## Actions

### On Invalid Commit
1. **BLOCK** the git commit command
2. **Show** what was wrong with the message
3. **Suggest** a corrected version
4. **Provide** the format reminder

### Examples
```
✅ feat: add user authentication with JWT
✅ fix(auth): resolve token refresh race condition
✅ docs: update API documentation for v2 endpoints
✅ refactor(db): extract query builder into separate module

❌ Added new feature        (no type prefix)
❌ FEAT: add login           (uppercase type)
❌ feat: add login.          (trailing period)
❌ feat:add login            (missing space after colon)
```

## Trigger Condition

Activate when Bash tool is called with a command containing `git commit`.
