# Contributing

Thanks for improving this repository. Please read this short guide before opening a PR.

## Table of Contents

- [Scope](#scope)
- [Getting Started](#getting-started)
- [Local Validation](#local-validation)
- [Agent Development Standards](#agent-development-standards)
  - [Frontmatter Requirements](#frontmatter-requirements)
  - [Validation Workflow](#validation-workflow)
  - [Tool Optimization Guidelines](#tool-optimization-guidelines)
  - [Quality Rubric](#quality-rubric)
- [Testing](#testing)
  - [Unit Testing Agents](#unit-testing-agents)
  - [Integration Testing](#integration-testing)
  - [Regression Testing](#regression-testing)
  - [Test Scenarios](#test-scenarios)
- [Pull Requests](#pull-requests)
- [Documentation Updates](#documentation-updates)
- [Security & Configuration](#security--configuration)
- [Licensing](#licensing-of-contributions)

## Scope

- Primary contributions target this repository (agents, commands, scripts).

## Getting Started

Before contributing, familiarize yourself with:
- `README.md` - Repository overview and guidelines
- `agents/README.md` - Agent catalog and invocation patterns
- `agents/AGENT_CHECKLIST.md` - Pre-flight checklist for agent updates
- `agents/TESTING.md` - Detailed testing procedures

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
category: foundation  # One of: orchestrator, foundation, development, specialist, expert, platform, integration, quality, finance, security
complexity: moderate  # One of: simple, moderate, complex, expert
model: claude-opus-4-6
capabilities:
  - Capability 1
  - Capability 2
auto_activate:
  keywords: [keyword1, keyword2]
  conditions: [condition1, condition2]
examples:
  - trigger: "Example user request"
    commentary: "What the agent does in response"
---
```

**Key Principles**:
- **Description Quality**: Should enable accurate routing by orchestration-coordinator
- **Uniqueness**: Agent name must be unique across all tiers
- **Capabilities**: List concrete capabilities, not vague claims
- **Examples**: Provide realistic trigger patterns with expected behavior

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
   - Verify expected behavior

4. **Documentation Sync**:
   - Update `agents/README.md` with new agent entry
   - Add trigger patterns to agent catalog
   - Update tier counts in main `README.md` if tier changes

### Tool Optimization Guidelines

**Note**: This repository follows a **full tool inheritance** philosophy. Agents do not declare explicit `tools:` fields and inherit all Claude Code tools by default. The tool sets documented in `agents/README.md` represent typical usage patterns, not restrictions.

**For least-privilege deployments**: Fork agents and add explicit `tools:` fields following the pattern used in VoltAgent.

**Tool Selection Guidance** (for documentation purposes):
```yaml
Read/Write/Edit: File operations
Bash: Command execution, git operations
Grep/Glob: Code search and discovery
WebSearch/WebFetch: External documentation
TodoWrite: Task coordination
Task: Agent delegation
```

### Quality Rubric

Agents are scored on:
- **Frontmatter Completeness** (20 pts): Valid YAML, required fields
- **Description Quality** (15 pts): Clear, specific, actionable
- **Content Structure** (20 pts): Organized sections, clear guidance
- **Practical Examples** (15 pts): Code snippets, command examples
- **Specificity** (15 pts): Domain-focused, avoids generic advice
- **Capabilities & Examples** (15 pts): Realistic triggers and behaviors

Run `python3 scripts/quality-scorer.py --help` for detailed scoring criteria.

## Testing

### Unit Testing Agents

Test individual agents in isolation:

```bash
# 1. Install the agent
./scripts/install-agents.sh --user

# 2. Restart Claude Code
# 3. Open a new conversation

# 4. Test with domain-specific prompts
# Example for code-reviewer:
"Review this pull request for security vulnerabilities"

# 5. Verify:
# - Agent is invoked (not another agent)
# - Response matches expected domain expertise
# - Quality is production-ready
```

### Integration Testing

Test agent interactions and orchestration:

```bash
# Test multi-agent workflows
# 1. Start with orchestration-coordinator
"Help me design and implement a new API endpoint"

# 2. Verify delegation chain:
# - orchestration-coordinator → api-platform-engineer (design)
# - orchestration-coordinator → backend-architect (implementation)
# - orchestration-coordinator → test-engineer (testing)

# 3. Check handoffs maintain context
```

### Regression Testing

Before merging changes, verify existing functionality:

```bash
# Run verification scripts
./scripts/verify-agents.sh

# Test all affected agents
python3 scripts/quality-scorer.py --agents-dir agents

# Verify no score regressions (compare to previous scores)
```

### Test Scenarios

**Minimum test coverage for new agents:**

| Scenario | What to Test |
|----------|-------------|
| **Happy Path** | Agent handles primary use case correctly |
| **Edge Cases** | Ambiguous requests, missing context |
| **Delegation** | Agent correctly delegates to specialists |
| **Boundaries** | Agent stays within its domain |
| **Error Handling** | Agent handles invalid inputs gracefully |

**Example test matrix for `api-platform-engineer`:**

```markdown
1. Happy Path: "Design a REST API for user management"
   Expected: OpenAPI spec, endpoint design, authentication guidance

2. Edge Case: "Help with API"
   Expected: Clarifying questions about type, domain, constraints

3. Delegation: "Build the API and write tests"
   Expected: Delegates to backend-architect and test-engineer

4. Boundaries: "Help me with CSS styling"
   Expected: Redirects to frontend-expert or asks to clarify

5. Error Handling: "Create API for [empty description]"
   Expected: Asks for requirements before proceeding
```

**Testing checklist:**
- [ ] Agent invoked by expected trigger phrases
- [ ] Response quality meets domain expertise level
- [ ] Agent stays within its defined scope
- [ ] Delegation works to appropriate specialists
- [ ] Error cases handled gracefully
- [ ] No hallucinated capabilities or tools

## Pull Requests

- Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Write clear, imperative commit messages (e.g., `agents(api): refine description`).
- Include what changed, why, and verification steps.
- Reference any related issues.

**Commit message format:**
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scopes: agents, scripts, skills, hooks, commands, mcp-servers, config
```

## Documentation Updates

When adding/renaming agents, update:
- `agents/README.md` (matrix and triggers)
- `agents/[tier]/README.md` (tier-level documentation)
- `CHANGELOG.md` (changelog)

## Security & Configuration

- Review `.mcp.json` changes carefully; never commit secrets. Prefer env vars.
- Agents inherit all tools by default; be mindful of security implications.
- Test for prompt injection resistance in agent prompts.

## Licensing of Contributions

By contributing, you agree your contributions (code, docs, agents) are licensed under the Apache License 2.0.
