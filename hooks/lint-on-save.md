---
name: lint-on-save
description: Runs language-appropriate linter after file writes to catch issues early
event: PostToolUse
tools: ["Write", "Edit"]
---

# Lint-on-Save Hook

Runs the appropriate linter after file modifications to catch syntax errors, style issues, and potential bugs immediately.

## Linter Selection

| File Pattern | Linter | Command |
|-------------|--------|---------|
| `*.ts`, `*.tsx` | ESLint + TypeScript | `npx eslint --fix {file}` |
| `*.js`, `*.jsx` | ESLint | `npx eslint --fix {file}` |
| `*.py` | Ruff (preferred) or Flake8 | `ruff check --fix {file}` |
| `*.rs` | Clippy | `cargo clippy --fix -- -W clippy::all` |
| `*.go` | golangci-lint | `golangci-lint run {file}` |
| `*.rb` | RuboCop | `bundle exec rubocop -a {file}` |
| `*.php` | PHP_CodeSniffer | `./vendor/bin/phpcs {file}` |
| `*.java` | Checkstyle | `mvn checkstyle:check` |
| `*.kt` | ktlint | `ktlint --format {file}` |
| `*.sh` | ShellCheck | `shellcheck {file}` |
| `*.yaml`, `*.yml` | yamllint | `yamllint {file}` |
| `*.json` | jsonlint | `jsonlint {file}` |
| `*.md` | markdownlint | `markdownlint {file}` |

## Behavior

1. After Write/Edit completes, detect file type from extension
2. Check if linter is available (skip silently if not installed)
3. Run linter in auto-fix mode where supported
4. Report only errors (suppress warnings for cleaner output)
5. If linter finds unfixable errors, report them concisely

## Configuration

```yaml
# Disable for specific paths
exclude_paths:
  - "**/node_modules/**"
  - "**/vendor/**"
  - "**/dist/**"
  - "**/build/**"
  - "**/.next/**"
  - "**/generated/**"

# Suppress warnings (only show errors)
errors_only: true

# Maximum lint runtime before timeout
timeout_seconds: 15
```

## Notes

- Linting is best-effort — if linter is not installed, skip silently
- Auto-fix mode preferred to minimize manual corrections
- Only lint source files, not generated code or dependencies
- Respect project-local linter configs (.eslintrc, .rubocop.yml, etc.)
