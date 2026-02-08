---
name: file-protection
description: Prevents modification of critical configuration files without explicit confirmation
event: PreToolUse
tools: ["Write", "Edit"]
---

# File Protection Hook

Guards critical files from accidental modification, requiring explicit user confirmation before changes.

## Protected File Patterns

### Always Protected
- `.env`, `.env.*` — Environment variables and secrets
- `*.pem`, `*.key`, `*.crt` — Certificates and private keys
- `.github/workflows/*.yml` — CI/CD pipelines
- `Dockerfile`, `docker-compose*.yml` — Container definitions
- `*.lock` — Dependency lock files (package-lock.json, yarn.lock, Gemfile.lock, poetry.lock)

### Protected with Warning
- `package.json`, `pyproject.toml`, `Cargo.toml` — Manifest files (warn, don't block)
- `tsconfig.json`, `.eslintrc.*` — Tool configuration
- `CLAUDE.md`, `.claude/*` — Agent configuration
- `*.sql` — Database migration files in `migrations/` directories

## Actions

### On Protected File Write/Edit
1. **BLOCK** the operation
2. **Explain** why the file is protected
3. **Show** what change was attempted
4. **Ask** for explicit confirmation before proceeding

### On Warning-Level File
1. **WARN** that the file is a critical config
2. **Show** the proposed change summary
3. **Proceed** unless the change looks dangerous (deleting all content, removing security configs)

## Configuration

```yaml
# Additional protected patterns
additional_protected:
  - "infrastructure/**/*.tf"
  - "k8s/**/*.yaml"

# Files to never protect (override)
never_protect:
  - "*.test.*"
  - "*.spec.*"
  - "**/fixtures/**"
```

## Notes

- Lock files should generally not be manually edited — suggest running the package manager instead
- CI/CD pipeline changes can have cascading effects — always review carefully
- Environment files may contain secrets — verify no sensitive values are being committed
