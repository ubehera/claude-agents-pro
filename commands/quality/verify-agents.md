---
description: Run the repository agent verification suite
args: [--fix]
tools: Bash(./scripts/verify-agents.sh:*)
model: sonnet
---

## Purpose
Lint frontmatter, tool declarations, and required sections across all agents using the maintained verification script.

## Run From Repo Root
!`./scripts/verify-agents.sh $ARGUMENTS`

## Interpreting Results
- Success prints a summary with zero errors.
- Failures include actionable diagnostics—fix them, then rerun the command.
- Add `--fix` to auto-apply supported repairs before rerunning verification.

## Reference Material
- `scripts/verify-agents.sh`
- `agents/AGENT_CHECKLIST.md`
