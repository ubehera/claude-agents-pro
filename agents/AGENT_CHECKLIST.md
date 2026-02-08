# Agent Addition/Update Checklist

Use this checklist before opening a PR for any agent in `agents/`.

## File & Location
- [ ] File lives in `agents/`
- [ ] Filename is `kebab-case.md` (e.g., `api-platform-engineer.md`)

## YAML Frontmatter
- [ ] Present at top with `---` fences
- [ ] Includes `name` (kebab-case, unique)
- [ ] Includes concise, specific `description` (drives routing)
- [ ] No `tools:` field required (agents inherit all tools automatically)

## Content Quality
- [ ] Short, actionable guidance; avoids generic filler
- [ ] Practical examples or commands where helpful
- [ ] No secrets or sensitive data

## Validation
- [ ] Install locally via `./scripts/install-agents.sh --user` (use `--project` when needed)
- [ ] Run `./scripts/verify-agents.sh` to confirm frontmatter and filenames
- [ ] Restart Claude Code to load changes
- [ ] Test prompts that should trigger the agent and confirm selection

## Documentation
- [ ] Update `agents/README.md` (matrix/triggers) if adding/renaming
- [ ] Update `../CLAUDE.md` if standards or workflows change
- [ ] Review `SYSTEM_OVERVIEW.md` for alignment
- [ ] Confirm `README.md` remains accurate

## Configuration & Security
- [ ] Review `.mcp.json` if adding servers; no credentials committed

## Tool Philosophy
Agents inherit all available tools automatically and do **not** use an explicit `tools:` field in frontmatter. Tool sets documented in the agent catalog (`agents/README.md`) represent typical usage patterns for reference, not restrictions. If you need least-privilege deployments, fork the agent and add explicit `tools:` fields for your environment.
