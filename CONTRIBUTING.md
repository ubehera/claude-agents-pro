# Contributing

Thanks for improving this repository. Please read this short guide before opening a PR.

## Scope
- Primary contributions target this repository (agents, commands, scripts).

## Start Here
- Read: `README.md` (repo guidelines)
- Agent checklist: `agents/AGENT_CHECKLIST.md`

## Local Validation
```bash
./scripts/install-agents.sh --user   # or --project for repo-scoped installs
./scripts/verify-agents.sh           # ensure frontmatter, names, tools are clean
```
- Restart Claude Code after installing agents.
- Validate the behaviour of any updated agent before submitting a PR.

## Agent Development Standards

### Frontmatter Requirements
Every agent must include valid YAML frontmatter with:
```yaml
---
name: agent-name  # Must match filename (kebab-case, without .md)
description: Clear, concise agent purpose that drives routing
tools: [Read, Write, Bash]  # Minimal set - only what's needed
---
```

**Key Principles**:
- **Least Privilege**: Only declare tools the agent actually needs
- **Specificity**: Description should enable accurate routing by agent-coordinator
- **Uniqueness**: Agent name must be unique across all tiers

### Validation Workflow
Before submitting any agent PR:

1. **Structural Validation**:
   ```bash
   ./scripts/verify-agents.sh  # Checks frontmatter, naming, structure
   ```

2. **Quality Scoring**:
   ```bash
   python3 scripts/quality-scorer.py --agent agents/[tier]/[agent-name].md
   ```
   - **Minimum Score**: 70/100 for new agents
   - **Production Score**: 85/100 for foundation/specialist agents

3. **Functional Testing**:
   - Install locally: `./scripts/install-agents.sh --user`
   - Restart Claude Code
   - Test with prompts matching agent domain
   - Verify tool restrictions work as expected

4. **Documentation Sync**:
   - Update `agents/README.md` with new agent entry
   - Add trigger patterns to agent catalog
   - Update tier counts in main `README.md` if tier changes

### Tool Optimization Guidelines
**Minimize Tools for Performance**:
- Each tool grants permissions and increases context
- Only include tools the agent will actively use
- Common patterns:
  - **Research agents**: `[WebSearch, WebFetch, Read]`
  - **Implementation agents**: `[Read, Write, Edit, Bash, Grep]`
  - **Review agents**: `[Read, Grep, Bash]`
  - **Orchestration agents**: `[TodoWrite, Read, Grep]`

**Tool Selection Strategy**:
```yaml
Read/Write/Edit: File operations
Bash: Command execution, git operations
Grep/Glob: Code search and discovery
WebSearch/WebFetch: External documentation
TodoWrite: Task coordination
```

### Quality Rubric
Agents are scored on:
- **Frontmatter Completeness** (20 pts): Valid YAML, required fields
- **Description Quality** (15 pts): Clear, specific, actionable
- **Tool Optimization** (15 pts): Minimal, justified tool set
- **Content Structure** (20 pts): Organized sections, clear guidance
- **Practical Examples** (15 pts): Code snippets, command examples
- **Specificity** (15 pts): Domain-focused, avoids generic advice

Run `python3 scripts/quality-scorer.py --help` for detailed scoring criteria.

## Pull Requests
- Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Write clear, imperative commit messages (e.g., `agents(api): refine description`).
- Include what changed, why, and verification steps.

## Documentation Updates
- When adding/renaming agents, update:
  - `agents/README.md` (matrix and triggers)

## Security & Configuration
- Review `.mcp.json` changes carefully; never commit secrets. Prefer env vars.
- Keep `tools` minimal per agent to reduce permissions and improve performance.

## Licensing of Contributions
- By contributing, you agree your contributions (code, docs, agents) are licensed under the Apache License 2.0.
