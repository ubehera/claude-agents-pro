# Hooks Directory

Event-driven automation hooks that trigger on Claude Code lifecycle events to enforce quality, security, and workflow standards.

## What are Hooks?

Hooks are **automation scripts** that execute in response to Claude Code events. They enable:
- Pre-tool validation (prevent unsafe operations)
- Post-tool verification (auto-format, auto-test)
- Session lifecycle management (context loading, cleanup)
- Notification integration (Slack, Discord, email)

## Hook Events

| Event | When It Fires | Use Cases |
|-------|--------------|-----------|
| `PreToolUse` | Before a tool executes | Block dangerous commands, validate file paths |
| `PostToolUse` | After a tool completes | Auto-format code, run linters, update indexes |
| `SessionStart` | When session begins | Load context, check prerequisites |

## Hook Format

Each hook is a markdown file with YAML frontmatter:

```yaml
---
name: hook-name
description: When and why this hook fires
event: PreToolUse  # Hook event type
tools: ["Bash", "Write"]  # Tools this hook monitors (for PreToolUse/PostToolUse)
---

Hook instructions and logic here.
```

## Available Hooks

### Security Hooks
| Hook | Event | Description |
|------|-------|-------------|
| `secrets-scanner` | PreToolUse | Prevents committing files containing secrets/credentials |
| `file-protection` | PreToolUse | Blocks modifications to protected files (.env, credentials) |
| `dependency-audit` | PostToolUse | Scans new dependencies for known vulnerabilities |

### Quality Hooks
| Hook | Event | Description |
|------|-------|-------------|
| `auto-format` | PostToolUse | Runs formatter after code edits (Prettier, Black, gofmt) |
| `lint-on-save` | PostToolUse | Runs linter after file writes |
| `test-on-change` | PostToolUse | Runs related tests after code changes |

### Git Hooks
| Hook | Event | Description |
|------|-------|-------------|
| `commit-message-validator` | PreToolUse | Enforces conventional commit format |
| `branch-protection` | PreToolUse | Prevents direct commits to main/master |
| `pr-template` | PreToolUse | Enriches PR creation with consistent template and test plan |

### Workflow Hooks
| Hook | Event | Description |
|------|-------|-------------|
| `session-context` | SessionStart | Loads project context and recent changes |

## Installation

Hooks are installed to `.claude/hooks/` (project) or `~/.claude/hooks/` (global):

```bash
# Install specific hook
cp hooks/secrets-scanner.md ~/.claude/hooks/

# Install all hooks
for hook in hooks/*.md; do
  cp "$hook" ~/.claude/hooks/
done
```

## Creating New Hooks

1. Create markdown file in `hooks/` directory
2. Add YAML frontmatter with name, description, event, and tools
3. Write hook logic in the body
4. Test with sample scenarios
5. Document in this README

## Hook Precedence

Project hooks (`.claude/hooks/`) override global hooks (`~/.claude/hooks/`) with the same name.
