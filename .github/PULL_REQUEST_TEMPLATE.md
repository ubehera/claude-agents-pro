## Summary
- What’s changing and why? Keep it concise and specific.

## Linked Issues
- Closes #

## Changes
- 

## Screenshots / Logs (optional)
- 

## Verification Steps
- Ran local install of agents (if applicable):
  ```bash
  mkdir -p .claude/agents && cp agents/*.md .claude/agents/
  ls ~/.claude/agents | rg '(api-platform|aws-cloud|system-design)'
  ```
- Restarted Claude Code and validated automatic agent routing via test prompts.
- Reviewed `.mcp.json` changes (if any) and confirmed no secrets committed.

## Impact / Risks
- 

## Checklist
- [ ] New/updated agent files use `kebab-case.md` and live in `agents/`
- [ ] YAML frontmatter includes `name` and `description`; `tools` minimal and intentional
- [ ] Agent description is specific and action‑oriented for routing
- [ ] Examples compile/are accurate; code blocks render properly
- [ ] Docs updated: `agents/README.md` (matrix/triggers)
- [ ] Local verification done (install, restart, prompt tests)
- [ ] No secrets or credentials added; env vars used for config
- [ ] I agree my contribution is licensed under the Apache License 2.0
