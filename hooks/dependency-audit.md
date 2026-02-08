---
name: dependency-audit
description: Scans newly added dependencies for known vulnerabilities and license issues
event: PostToolUse
tools: ["Bash"]
---

# Dependency Audit Hook

Automatically scans for security vulnerabilities when new dependencies are installed.

## Trigger Conditions

Activate after Bash commands containing:
- `npm install`, `npm i`, `yarn add`, `pnpm add`
- `pip install`, `poetry add`, `uv add`
- `cargo add`
- `go get`
- `bundle add`, `gem install`
- `composer require`
- `dotnet add package`

## Audit Commands

| Package Manager | Audit Command |
|----------------|---------------|
| npm | `npm audit --omit=dev` |
| yarn | `yarn audit` |
| pnpm | `pnpm audit` |
| pip | `pip-audit` or `safety check` |
| cargo | `cargo audit` |
| go | `govulncheck ./...` |
| bundler | `bundle audit check` |
| composer | `composer audit` |
| dotnet | `dotnet list package --vulnerable` |

## Actions

### On Vulnerability Found
1. **Report** severity (critical, high, medium, low) and affected package
2. **Suggest** fix command if available (e.g., `npm audit fix`)
3. **Warn** about critical/high vulnerabilities — suggest alternative packages
4. **Do not block** — informational only (user decides whether to proceed)

### License Check (Optional)
- Flag copyleft licenses (GPL, AGPL) in commercial projects
- Report license types for new dependencies
- Suggest alternatives for incompatible licenses

## Configuration

```yaml
# Minimum severity to report
min_severity: high

# Ignore specific advisories
ignore_advisories:
  - GHSA-xxxx-xxxx-xxxx

# License allowlist
allowed_licenses:
  - MIT
  - Apache-2.0
  - BSD-2-Clause
  - BSD-3-Clause
  - ISC
```
