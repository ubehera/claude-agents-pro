---
description: Install Claude Code agents for user or project scopes
args: [--user|--project] [--select agent1,agent2] [--dry-run]
tools: Bash(./scripts/install-agents.sh:*)
model: sonnet
---

## Objective
Provision this repository's Claude agents by wrapping `./scripts/install-agents.sh` with flexible arguments.

## Before You Run
- Stay at the repository root so relative paths resolve.
- Review available options with `./scripts/install-agents.sh --help` if uncertain.

## Execution
Run the installer with any required flags:

```bash
./scripts/install-agents.sh $ARGUMENTS
```

Typical flows:
- Install everything for the current user: `/install-agents --user`
- Project-only install for automation: `/install-agents --project`
- Select specific agents: `/install-agents --select api-platform-engineer,security-architect`

## Follow Up
- Inspect the output for `[SUCCESS]` messages.
- Re-run with `--dry-run` when you need a no-op preview.
- Consult `agents/AGENT_CHECKLIST.md` for post-install validation expectations.
