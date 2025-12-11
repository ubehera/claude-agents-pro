# Claude Agents CLI - Setup Guide

Complete installation and testing guide for the `claude-agents` CLI tool.

---

## Prerequisites

- **Python 3.10+** (tested with Python 3.14)
- **pip** package manager
- **Virtual environment** (recommended)

---

## Quick Start (5 minutes)

### Option 1: Automated Setup

```bash
cd /Users/umank/Code/agent-repos/claude-agents-pro/cli

# Run automated test script (creates venv, installs, tests)
./test-cli.sh
```

This script will:
1. Create a virtual environment
2. Install dependencies (click, rich, PyYAML)
3. Install the CLI in development mode
4. Run 6 test commands
5. Display results

### Option 2: Manual Setup

```bash
cd /Users/umank/Code/agent-repos/claude-agents-pro/cli

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install click rich PyYAML

# Install CLI in development mode
pip install -e .

# Test installation
claude-agents --help
```

### Option 3: Using Makefile

```bash
cd /Users/umank/Code/agent-repos/claude-agents-pro/cli

# Create venv and install with dev dependencies
make quickstart

# Activate virtual environment
source venv/bin/activate

# Run tests
make test

# Use CLI
claude-agents --help
```

---

## Installation Verification

After installation, verify the CLI is working:

```bash
# Check version
claude-agents --version

# Show help
claude-agents --help

# List available commands
claude-agents --help | grep "Commands:"

# Test list command
claude-agents list --format simple | head -5

# Test search command
claude-agents search "API design" --limit 3
```

Expected output:
```
claude-agents, version 1.0.0

Usage: claude-agents [OPTIONS] COMMAND [ARGS]...

Options:
  --version   Show the version and exit.
  --help      Show this message and exit.

Commands:
  info      Show detailed agent information
  install   Install Claude Code agents to user or project scope.
  list      List installed agents
  score     Run quality scoring on agents
  search    Search agents by capability using fuzzy matching.
  validate  Validate agents against quality standards
```

---

## Command Reference

### 1. Install Command

Install agents to user or project scope.

```bash
# Install all agents to user scope (default)
claude-agents install

# Install to project scope
claude-agents install --scope project

# Install specific tier
claude-agents install --tier 01-foundation

# Install single agent
claude-agents install --agent api-platform-engineer

# Preview installation (dry-run)
claude-agents install --dry-run

# Install from custom directory
claude-agents install --agents-dir /path/to/agents
```

**Output**:
- Installation plan panel
- Progress spinner
- Success/failure summary
- Installed file count

### 2. List Command

List installed agents with various formats.

```bash
# List all agents (table format)
claude-agents list

# List specific tier
claude-agents list --tier 01-foundation

# JSON output
claude-agents list --format json

# Simple text list
claude-agents list --format simple

# List from project scope
claude-agents list --scope project
```

**Output**:
- Rich table with agent name, tier, domain
- Total agent count
- Color-coded tiers

### 3. Search Command

Fuzzy search agents by capability.

```bash
# Search by keyword
claude-agents search "database optimization"

# Adjust similarity threshold
claude-agents search "API design" --min-score 0.5

# Limit results
claude-agents search "testing" --limit 5

# Search in user scope
claude-agents search "Python" --scope user
```

**Output**:
- Ranked results with similarity scores
- Agent name, tier, description
- Top match recommendation
- Hint for detailed info

### 4. Info Command

Show detailed agent information.

```bash
# Basic info
claude-agents info api-platform-engineer

# Full content
claude-agents info python-expert --full

# Search in project scope
claude-agents info database-architect --scope project
```

**Output**:
- Rich panel with agent metadata
- Tier, category, complexity
- Model and rationale
- Capabilities list
- Auto-activation keywords
- File location

### 5. Validate Command

Validate agents against quality standards.

```bash
# Validate all agents
claude-agents validate

# Strict validation
claude-agents validate --strict

# Custom directory
claude-agents validate --agents-dir /path/to/agents
```

**Output**:
- Validation summary (total, valid, invalid)
- Pass rate percentage
- Detailed error table
- Per-agent issue list

### 6. Score Command

Run comprehensive quality scoring.

```bash
# Score all agents
claude-agents score

# Score specific agent
claude-agents score api-platform-engineer

# Generate JSON report
claude-agents score --output quality-report.json

# Set minimum threshold
claude-agents score --min-score 0.8
```

**Output**:
- Overall quality score (0-1.0)
- Metrics breakdown (completeness, accuracy, usability)
- Pass/fail status
- Top performers list (when scoring all)

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'click'`

**Solution**: Install dependencies

```bash
pip install click rich PyYAML
```

### Issue: `command not found: claude-agents`

**Solution**: Ensure virtual environment is activated and CLI is installed

```bash
source venv/bin/activate
pip install -e .
```

### Issue: `Agents directory not found`

**Solution**: Specify agents directory explicitly

```bash
claude-agents list --agents-dir /Users/umank/Code/agent-repos/claude-agents-pro/agents
```

### Issue: `Permission denied` when installing

**Solution**: Use virtual environment instead of system Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: Import errors in commands

**Solution**: Reinstall in development mode

```bash
cd /Users/umank/Code/agent-repos/claude-agents-pro/cli
pip uninstall claude-agents-cli
pip install -e .
```

---

## Configuration

CLI configuration is stored in `~/.claude/cli-config.json`.

### Default Configuration

```json
{
  "version": "1.0.0",
  "default_scope": "user",
  "marketplace_url": "https://github.com/ubehera/claude-agents-pro",
  "quality_threshold": 0.7,
  "auto_update": false
}
```

### Customization

Edit `~/.claude/cli-config.json` to change defaults:

```json
{
  "default_scope": "project",      // Change default installation scope
  "quality_threshold": 0.85,       // Raise quality bar
  "auto_update": true              // Enable automatic updates
}
```

---

## Development Workflow

### Run Tests

```bash
# Automated test suite
./test-cli.sh

# Or using Makefile
make test
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking (if mypy installed)
mypy cli/
```

### Build Distribution

```bash
# Build packages
make build

# Verify build
ls -la dist/
```

### Local Testing

```bash
# Install in editable mode
pip install -e .

# Make changes to code
vim commands/search.py

# Test immediately (no reinstall needed)
claude-agents search "test"
```

---

## Integration with Claude Code

Once agents are installed, invoke them in Claude Code using the Task tool:

```python
# Example 1: API design task
from task import Task

Task(
    subagent_type='api-platform-engineer',
    description='Design REST API for user management with OAuth2'
)
```

```python
# Example 2: Database optimization
Task(
    subagent_type='database-architect',
    description='Optimize PostgreSQL query performance for user analytics'
)
```

The CLI ensures agents are properly installed and discoverable by Claude Code.

---

## Uninstallation

### Remove CLI

```bash
pip uninstall claude-agents-cli
```

### Remove Installed Agents

```bash
# User scope
rm -rf ~/.claude/agents

# Project scope
rm -rf ./.claude/agents
```

### Remove Configuration

```bash
rm ~/.claude/cli-config.json
```

---

## Next Steps

1. **Install the CLI**
   ```bash
   cd /Users/umank/Code/agent-repos/claude-agents-pro/cli
   ./test-cli.sh
   ```

2. **Install agents**
   ```bash
   claude-agents install --scope user
   ```

3. **Explore agents**
   ```bash
   claude-agents list
   claude-agents search "your domain"
   claude-agents info <agent-name>
   ```

4. **Use in Claude Code**
   - Agents are now available for Task tool delegation
   - Reference agents by name (e.g., 'api-platform-engineer')

5. **Contribute**
   - Create new agents in `/agents/<tier>/`
   - Run `claude-agents validate` before committing
   - Use `claude-agents score` to ensure quality >= 0.7

---

## Support

- **Issues**: https://github.com/ubehera/claude-agents-pro/issues
- **Documentation**: `/Users/umank/Code/agent-repos/claude-agents-pro/cli/README.md`
- **Examples**: Run `python3 demo.py` for detailed examples

---

**CLI Location**: `/Users/umank/Code/agent-repos/claude-agents-pro/cli/`
**Test Script**: `./test-cli.sh`
**Makefile**: Available for quick commands
