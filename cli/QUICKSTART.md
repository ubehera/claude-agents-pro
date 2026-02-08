# Claude Agents CLI - Quick Start Guide

Get up and running with the Claude Agents CLI in under 5 minutes.

## 1. Install

```bash
cd claude-agents-pro/cli
pip install -e .
```

## 2. Verify Installation

```bash
claude-agents --version
```

Expected output:
```
claude-agents, version 1.0.0
```

## 3. Install Agents

Install all agents to your user directory (`~/.claude/agents`):

```bash
claude-agents install
```

Expected output:
```
╭─────────────────────────── Installation Plan ───────────────────────────────╮
│ Installation Plan                                                            │
│                                                                              │
│ Scope: user                                                                  │
│ Target: /Users/you/.claude/agents                                           │
│ Agents: 30                                                                   │
│ Dry run: False                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

✓ Successfully installed 30 agents to /Users/you/.claude/agents
✓ Installed CLAUDE.md to /Users/you/.claude/CLAUDE.md
```

## 4. List Installed Agents

```bash
claude-agents list
```

Expected output:
```
┌─────────────────────────────────┬───────────────┬─────────────────────┐
│ Agent                           │ Tier          │ Category            │
├─────────────────────────────────┼───────────────┼─────────────────────┤
│ orchestration-coordinator       │ 00-meta       │ meta                │
│ api-platform-engineer           │ 01-foundation │ foundation          │
│ code-reviewer                   │ 01-foundation │ foundation          │
│ ...                             │ ...           │ ...                 │
└─────────────────────────────────┴───────────────┴─────────────────────┘

Total agents: 30
```

## 5. Search for an Agent

```bash
claude-agents search "API design"
```

Expected output:
```
Found 3 matching agents for: 'API design'

┌──────┬───────┬──────────────────────┬───────────────┬──────────────────────┐
│ Rank │ Score │ Agent                │ Tier          │ Description          │
├──────┼───────┼──────────────────────┼───────────────┼──────────────────────┤
│ 1.   │ 0.92  │ api-platform-engineer│ 01-foundation │ REST API design,...  │
│ 2.   │ 0.68  │ backend-architect    │ 03-specialists│ Backend architect... │
│ 3.   │ 0.52  │ system-design-spec...│ 01-foundation │ System architecture..│
└──────┴───────┴──────────────────────┴───────────────┴──────────────────────┘

Top match: api-platform-engineer
Use 'claude-agents info api-platform-engineer' for details
```

## 6. Get Agent Details

```bash
claude-agents info api-platform-engineer
```

Expected output:
```
╭─────────────────────────────── Agent Information ───────────────────────────╮
│ api-platform-engineer                                                        │
│ Tier: 01-foundation | Category: foundation | Complexity: complex            │
│                                                                              │
│ Expert in REST API design, GraphQL schemas, OpenAPI/Swagger specs, API...   │
╰──────────────────────────────────────────────────────────────────────────────╯

Metadata:
  Model: claude-opus-4-6
  Rationale: Balanced performance for complex analysis
  File: /Users/you/.claude/agents/01-foundation/api-platform-engineer.md

Capabilities:
  • REST API design
  • GraphQL schema design
  • OpenAPI/Swagger specifications
  • API gateway configuration
  • OAuth 2.0 and JWT authentication

Auto-activation:
  Keywords: API, REST, GraphQL, OpenAPI, gateway, OAuth, JWT
```

## 7. Validate Agent Quality

```bash
claude-agents validate
```

Expected output:
```
Validating agents in: /Users/you/claude-agents-pro/agents

Validation Results:

Total agents: 30
Valid: 30
Invalid: 0
Pass rate: 100.0%

✓ All agents passed validation!
```

## 8. Score Agent Quality

```bash
claude-agents score
```

Expected output:
```
╭──────────────────────────── Quality Summary ─────────────────────────────────╮
│ Total agents: 30                                                             │
│ Passing (>=0.7): 28                                                          │
│ Failing (<0.7): 2                                                            │
│ Average score: 0.82                                                          │
│ Pass rate: 93.3%                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Top 5 Performers:
  0.94 - orchestration-coordinator (00-meta)
  0.91 - system-design-specialist (01-foundation)
  0.89 - backend-architect (03-specialists)
  0.87 - api-platform-engineer (01-foundation)
  0.86 - security-architect (07-quality)
```

## Common Commands

### Install Specific Tier
```bash
claude-agents install --tier 01-foundation
```

### Install Specific Agent
```bash
claude-agents install --agent python-expert
```

### Install to Project Scope
```bash
claude-agents install --scope project
```

### List with JSON Output
```bash
claude-agents list --format json
```

### Search with Custom Threshold
```bash
claude-agents search "testing" --min-score 0.6 --limit 5
```

### View Full Agent Content
```bash
claude-agents info python-expert --full
```

### Strict Validation
```bash
claude-agents validate --strict
```

### Score Specific Agent
```bash
claude-agents score api-platform-engineer
```

### Generate Quality Report
```bash
claude-agents score --output quality-report.json
```

## Troubleshooting

### Command Not Found

After installation, if you see "command not found", restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc
```

Alternatively, add Python scripts to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Import Errors

If you see `ModuleNotFoundError`, install dependencies:

```bash
pip install click rich PyYAML
```

### Permission Errors

Use `--user` flag or virtual environment:

```bash
pip install --user -e .
```

or

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Next Steps

1. **Read Full Documentation**: See [README.md](README.md) for complete command reference
2. **Installation Guide**: See [INSTALL.md](INSTALL.md) for detailed installation instructions
3. **Integration**: Use agents in Claude Code via the Task tool
4. **Contribute**: See [../CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines

## Support

- **Issues**: https://github.com/ubehera/claude-agents-pro/issues
- **Documentation**: https://github.com/ubehera/claude-agents-pro
- **Repository**: https://github.com/ubehera/claude-agents-pro

---

You're ready to go! Start using Claude Code agents with the CLI to streamline your development workflow.
