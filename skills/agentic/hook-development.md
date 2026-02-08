---
name: hook-development
description: Load when creating or configuring Claude Code hooks for event-driven automation
trigger_keywords: [hook, PreToolUse, PostToolUse, SessionStart, SessionEnd, Stop, Notification, UserPromptSubmit, event hook, automation hook]
---

# Hook Development Skill

Patterns for creating Claude Code hooks that automate quality, security, and workflow enforcement across tool interactions.

## Overview

Hooks are event-driven automation scripts that fire before or after Claude Code tool executions. They enable guardrails, auto-formatting, testing, and notification workflows without manual intervention.

**When to Use**:
- Preventing dangerous operations before they execute
- Automating quality checks after code changes
- Loading context at session start
- Sending notifications on completion or errors

## Hook Events

### PreToolUse — Before Tool Executes

**Purpose**: Validate, block, or modify tool inputs before execution.

```yaml
---
name: my-pretool-hook
description: What this hook does
event: PreToolUse
tools: ["Bash", "Write"]  # Which tools trigger this hook
---

# Logic: Inspect the tool call arguments
# Actions: BLOCK (prevent execution) or ALLOW (proceed)
# Output: Explanation of decision
```

**Common Patterns**:
- **Secrets scanner**: Block commits/writes containing API keys
- **Branch protection**: Block commits to main/master
- **File protection**: Require confirmation for critical file edits
- **Command validation**: Block destructive commands (rm -rf, DROP TABLE)

### PostToolUse — After Tool Completes

**Purpose**: React to tool results, run follow-up actions.

```yaml
---
name: my-posttool-hook
description: What this hook does
event: PostToolUse
tools: ["Write", "Edit"]  # Which tools trigger this hook
---

# Logic: Inspect the completed tool result
# Actions: Run formatters, linters, tests, or notifications
# Output: Results or warnings
```

**Common Patterns**:
- **Auto-format**: Run Prettier/Black after file writes
- **Lint-on-save**: Run ESLint/Ruff after edits
- **Test-on-change**: Run related tests after code modifications
- **Dependency audit**: Scan for vulnerabilities after installs

### SessionStart — When Session Begins

**Purpose**: Load context, check prerequisites, set up environment.

```yaml
---
name: my-session-hook
description: What this hook does
event: SessionStart
---

# Logic: Gather project context
# Actions: Load git status, active tasks, project type
# Output: Context summary for session
```

### Stop — When Agent Completes

**Purpose**: Summary generation, cleanup, state persistence.

```yaml
---
name: my-stop-hook
description: What this hook does
event: Stop
---

# Logic: Summarize work done
# Actions: Update task tracking, send notifications
# Output: Completion summary
```

### UserPromptSubmit — Before Processing User Input

**Purpose**: Enrich or validate user prompts before processing.

```yaml
---
name: my-prompt-hook
description: What this hook does
event: UserPromptSubmit
---

# Logic: Analyze user prompt
# Actions: Add context, suggest clarifications, enforce style
# Output: Enriched prompt context
```

## Hook Design Patterns

### Guard Pattern (PreToolUse)

Block operations that match dangerous patterns:

```markdown
## Detection Rules

1. Scan command/content for patterns:
   - Regex match against known dangerous patterns
   - Check current git branch for protected status
   - Verify file path against protection list

2. Decision:
   - **BLOCK**: Pattern matches → explain why and suggest alternative
   - **ALLOW**: No match → proceed silently

## Important: Fail Open
- If detection logic errors, ALLOW the operation
- Never block legitimate work due to false positives
- Log blocked operations for review
```

### React Pattern (PostToolUse)

Automatically respond to tool completions:

```markdown
## Trigger Logic

1. Check tool type and file extension
2. Determine appropriate action:
   - .ts/.tsx → ESLint + Prettier
   - .py → Ruff + Black
   - .go → gofmt + golangci-lint

3. Execute action:
   - Run silently if tool available
   - Skip silently if tool not installed
   - Report only errors, not warnings

## Important: Non-Blocking
- Don't prevent the user from continuing
- Report issues concisely
- Auto-fix when possible
```

### Context Pattern (SessionStart)

Load relevant context at session beginning:

```markdown
## Context Sources

1. Git: branch, status, recent commits
2. Tasks: active TodoWrite entries
3. Project: language, framework, config
4. Memory: knowledge graph for project patterns
5. Environment: tool versions, available linters

## Output: Brief Summary (3-5 lines)
Keep it fast (<2 seconds) and concise.
Don't overwhelm — summarize, don't dump.
```

## Best Practices

1. **Fail open** — hooks should never block legitimate work
2. **Fast execution** — hooks add latency; keep them under 2 seconds
3. **Silent success** — only output when there's something to report
4. **Idempotent** — hooks may fire multiple times; handle gracefully
5. **Configurable** — support exclude patterns and severity thresholds
6. **Documented** — clear description of when and why the hook fires

---

**Skill Type**: Agentic — Hooks
**Complexity**: Moderate
**Typical Usage**: Creating custom hooks, understanding hook lifecycle, automation patterns
