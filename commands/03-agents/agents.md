---
description: Install, validate, and manage Claude Code agents with automated quality checks
tools: Bash(./scripts/install-agents.sh:*), Bash(./scripts/verify-agents.sh:*), Read, Write, Grep, Glob
model: sonnet
args: [action] [agent-name]
---

# Agent Management Command

Comprehensive management of Claude Code agents including installation, validation, and quality monitoring.

## Usage Patterns

### Installation & Setup
```
/agents install --user
/agents install --project
/agents refresh api-platform-engineer
```

### Validation & Quality
```
/agents validate all
/agents validate code-reviewer
/agents quality-report
```

### Discovery & Information
```
/agents list tier-01
/agents info performance-optimization-specialist
/agents search "API design"
```

## Command Logic

1. **Action Routing**: Dispatch to appropriate management function
2. **Validation**: Run structural and frontmatter validation
3. **Quality Scoring**: Generate quality metrics and reports
4. **Installation**: Handle user/project scoped agent installation
5. **Discovery**: Search and information retrieval

## Available Actions

### Installation Actions
- `install --user`: Install all agents to ~/.claude/agents
- `install --project`: Install to .claude/agents (project-scoped)
- `refresh [agent]`: Refresh specific agent or all agents

### Validation Actions
- `validate all`: Run comprehensive validation on all agents
- `validate [agent]`: Validate specific agent structure and tools
- `quality-report`: Generate detailed quality assessment

### Information Actions
- `list [tier]`: List agents by tier or all agents
- `info [agent]`: Show detailed agent information
- `search [term]`: Search agents by capability or domain

## Implementation Details

### Installation Commands
```bash
# User installation
./scripts/install-agents.sh --user

# Project installation
./scripts/install-agents.sh --project

# Validation
./scripts/verify-agents.sh

# Quality scoring
python3 scripts/quality-scorer.py --agents-dir agents --output quality-report.json
```

### Quality Metrics
- **Structure Score**: Frontmatter validation, tool declarations
- **Content Score**: Description quality, usage examples
- **Integration Score**: Tool usage appropriateness, delegation patterns
- **Overall Score**: Weighted combination with tier-specific thresholds

### Error Handling
- Invalid frontmatter: Detailed YAML parsing errors
- Missing tools: Tool permission validation
- File structure: Directory organization compliance
- Quality thresholds: Automated scoring with improvement suggestions