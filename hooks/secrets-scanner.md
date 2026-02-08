---
name: secrets-scanner
description: Scans files for hardcoded secrets, API keys, passwords, and credentials before they are committed or written
event: PreToolUse
tools: ["Bash", "Write", "Edit"]
---

# Secrets Scanner Hook

Prevents accidental exposure of sensitive credentials by scanning file content before writes and commits.

## Detection Patterns

Scan for these patterns in file content being written or committed:

### API Keys & Tokens
- `AKIA[0-9A-Z]{16}` — AWS Access Key ID
- `ghp_[a-zA-Z0-9]{36}` — GitHub Personal Access Token
- `sk-[a-zA-Z0-9]{48}` — OpenAI API Key
- `xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}` — Slack Bot Token
- `AIza[0-9A-Za-z_-]{35}` — Google API Key

### Passwords & Secrets
- `password\s*=\s*['"][^'"]+['"]` — Hardcoded passwords
- `secret\s*=\s*['"][^'"]+['"]` — Hardcoded secrets
- `-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----` — Private keys
- `mongodb(\+srv)?://[^:]+:[^@]+@` — MongoDB connection strings with credentials

### Environment Variables in Code
- Inline `.env` values that should be environment variables
- Database connection strings with embedded credentials

## Actions

### On Detection
1. **BLOCK** the operation
2. **Report** which patterns matched and in which files
3. **Suggest** using environment variables, secret managers, or `.env` files
4. **Remind** to add sensitive files to `.gitignore`

### Exceptions
- Test fixtures with obviously fake credentials (e.g., `test-key-12345`)
- Documentation examples with placeholder values
- Files already in `.gitignore`

## Configuration

```yaml
# Optional: paths to exclude from scanning
exclude_paths:
  - "**/*.test.*"
  - "**/fixtures/**"
  - "**/mocks/**"
```
