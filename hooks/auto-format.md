---
name: auto-format
description: Automatically formats code after file edits using the appropriate formatter for the file type
event: PostToolUse
tools: ["Write", "Edit"]
---

# Auto-Format Hook

Runs the appropriate code formatter after file modifications to maintain consistent style.

## Formatter Selection

Select formatter based on file extension:

| Extension | Formatter | Command |
|-----------|-----------|---------|
| `.ts`, `.tsx`, `.js`, `.jsx` | Prettier | `npx prettier --write {file}` |
| `.py` | Black + isort | `black {file} && isort {file}` |
| `.go` | gofmt | `gofmt -w {file}` |
| `.rs` | rustfmt | `rustfmt {file}` |
| `.rb` | RuboCop | `rubocop -a {file}` |
| `.php` | Laravel Pint | `./vendor/bin/pint {file}` |
| `.java` | google-java-format | `google-java-format -i {file}` |
| `.kt` | ktlint | `ktlint -F {file}` |
| `.cs` | dotnet format | `dotnet format --include {file}` |
| `.css`, `.scss` | Prettier | `npx prettier --write {file}` |
| `.json`, `.yaml`, `.yml` | Prettier | `npx prettier --write {file}` |
| `.md` | Prettier | `npx prettier --write {file}` |

## Behavior

1. After a Write or Edit tool completes, check the file extension
2. Look for formatter configuration files in project root (`.prettierrc`, `pyproject.toml`, etc.)
3. If formatter is available (check with `which` or `npx`), run it
4. If formatter is not installed, skip silently (don't block workflow)
5. Report formatting changes briefly

## Configuration

```yaml
# Disable for specific paths
exclude_paths:
  - "**/*.min.js"
  - "**/vendor/**"
  - "**/node_modules/**"
  - "**/dist/**"

# Override formatters
overrides:
  "*.ts": "deno fmt {file}"
  "*.py": "ruff format {file}"
```

## Notes
- Only format files that were actually modified (not all project files)
- Respect project-level formatter configuration
- Fail silently if formatter is not available
- Do not format generated files or vendored code
