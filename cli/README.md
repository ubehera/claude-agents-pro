# Claude Agents CLI

Production-ready CLI tool for managing Claude Code agents in the claude-agents-pro ecosystem.

## Features

- **Install Agents** - Deploy agents to user or project scope
- **List Agents** - Browse installed agents with filtering
- **Search Agents** - Fuzzy search by capability and domain
- **Agent Info** - Detailed agent metadata and documentation
- **Validate** - Quality validation against standards
- **Score** - Comprehensive quality scoring system

## Installation

### From Source

```bash
cd claude-agents-pro/cli
pip install -e .
```

### System-wide Installation

```bash
cd claude-agents-pro/cli
pip install .
```

### Development Installation

```bash
cd claude-agents-pro/cli
pip install -e ".[dev]"
```

## Usage

### Install Agents

```bash
# Install all agents to user scope (default)
claude-agents install

# Install to project scope
claude-agents install --scope project

# Install specific tier
claude-agents install --tier 01-foundation

# Install specific agent
claude-agents install --agent api-platform-engineer

# Dry run to preview installation
claude-agents install --dry-run
```

### List Agents

```bash
# List all agents (default: table format)
claude-agents list

# List specific tier
claude-agents list --tier 01-foundation

# List from project scope
claude-agents list --scope project

# Output formats
claude-agents list --format table   # Default: formatted table
claude-agents list --format json    # JSON output
claude-agents list --format simple  # Simple text list
```

Example output:

```
┌─────────────────────────────────┬───────────────┬─────────────────────┐
│ Agent                           │ Tier          │ Domain              │
├─────────────────────────────────┼───────────────┼─────────────────────┤
│ orchestration-coordinator       │ 00-meta       │ Orchestration       │
│ api-platform-engineer           │ 01-foundation │ API Design          │
│ code-reviewer                   │ 01-foundation │ Code Quality        │
│ test-engineer                   │ 01-foundation │ Testing             │
│ python-expert                   │ 02-development│ Python Development  │
└─────────────────────────────────┴───────────────┴─────────────────────┘

Total agents: 30
```

### Search Agents

```bash
# Fuzzy search by capability
claude-agents search "database optimization"

# Search with minimum score threshold
claude-agents search "API design" --min-score 0.5

# Limit results
claude-agents search "testing" --limit 5
```

Example output:

```
Found 3 matching agents for: 'database optimization'

┌──────┬───────┬──────────────────────┬───────────────┬──────────────────────┐
│ Rank │ Score │ Agent                │ Tier          │ Description          │
├──────┼───────┼──────────────────────┼───────────────┼──────────────────────┤
│ 1.   │ 0.95  │ database-architect   │ 03-specialists│ Database design &... │
│ 2.   │ 0.72  │ performance-optimiz..│ 01-foundation │ Performance tuning...│
│ 3.   │ 0.45  │ backend-architect    │ 03-specialists│ Service architecture │
└──────┴───────┴──────────────────────┴───────────────┴──────────────────────┘

Top match: database-architect
Use 'claude-agents info database-architect' for details
```

### Agent Info

```bash
# Show agent details
claude-agents info api-platform-engineer

# Show full agent content
claude-agents info api-platform-engineer --full

# Search in project scope
claude-agents info python-expert --scope project
```

Example output:

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

### Validate Agents

```bash
# Validate all agents
claude-agents validate

# Strict validation with additional checks
claude-agents validate --strict

# Validate specific directory
claude-agents validate --agents-dir /path/to/agents
```

Example output:

```
Validating agents in: /Users/you/claude-agents-pro/agents

Validation Results:

Total agents: 30
Valid: 28
Invalid: 2
Pass rate: 93.3%

╭─────────────────────────── Validation Issues ───────────────────────────────╮
│ Agent                  │ Tier          │ Issues                            │
├────────────────────────┼───────────────┼───────────────────────────────────┤
│ old-agent              │ 01-foundation │ • Missing field: model_rationale  │
│                        │               │ • Description too short           │
└────────────────────────┴───────────────┴───────────────────────────────────┘
```

### Quality Scoring

```bash
# Score all agents
claude-agents score

# Score specific agent
claude-agents score api-platform-engineer

# Generate JSON report
claude-agents score --output quality-report.json

# Set minimum score threshold
claude-agents score --min-score 0.8
```

Example output (single agent):

```
╭─────────────────────────── Quality Score ───────────────────────────────────╮
│ Agent: api-platform-engineer                                                │
│ Overall Score: 0.87/1.00 PASS                                               │
│ Tier: Tier 1 - Foundation                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯

Metrics Breakdown:
┌───────────────┬───────┬────────┬────────┐
│ Metric        │ Score │ Weight │ Status │
├───────────────┼───────┼────────┼────────┤
│ Completeness  │ 0.92  │ 25%    │ ✓      │
│ Accuracy      │ 0.88  │ 25%    │ ✓      │
│ Usability     │ 0.85  │ 20%    │ ✓      │
│ Performance   │ 0.82  │ 15%    │ ✓      │
│ Maintainabil..│ 0.86  │ 15%    │ ✓      │
└───────────────┴───────┴────────┴────────┘
```

Example output (all agents):

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

## Configuration

CLI configuration is stored in `~/.claude/cli-config.json`:

```json
{
  "version": "1.0.0",
  "default_scope": "user",
  "marketplace_url": "https://github.com/ubehera/claude-agents-pro",
  "quality_threshold": 0.7,
  "auto_update": false
}
```

## Directory Structure

```
cli/
├── __init__.py              # Package initialization
├── __main__.py              # Main CLI entry point
├── commands/
│   ├── __init__.py
│   ├── install.py           # Agent installation
│   ├── list_agents.py       # List agents
│   ├── search.py            # Fuzzy search
│   ├── info.py              # Agent details
│   ├── validate.py          # Quality validation
│   └── score.py             # Quality scoring
├── utils/
│   ├── __init__.py
│   ├── agent_parser.py      # YAML frontmatter parser
│   └── config.py            # Configuration management
├── pyproject.toml           # Package metadata
└── README.md                # This file
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=cli
```

### Code Quality

```bash
# Format code
black cli/

# Lint code
ruff check cli/

# Type checking
mypy cli/
```

### Creating a Release

```bash
# Build distribution
python -m build

# Install locally
pip install dist/claude_agents_cli-1.0.0-py3-none-any.whl
```

## Integration with Claude Code

Once installed, agents can be invoked in Claude Code using the Task tool:

```python
# Example: Delegate to API platform engineer
Task(
    subagent_type='api-platform-engineer',
    description='Design REST API for user management with OAuth2'
)
```

## Troubleshooting

### Command not found

After installation, restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### Permission errors

For user scope installation:

```bash
pip install --user -e .
```

### Import errors

Ensure dependencies are installed:

```bash
pip install click rich PyYAML
```

## License

Licensed under Apache-2.0. See LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## Links

- **Repository**: https://github.com/ubehera/claude-agents-pro
- **Documentation**: https://github.com/ubehera/claude-agents-pro/blob/main/README.md
- **Issues**: https://github.com/ubehera/claude-agents-pro/issues
