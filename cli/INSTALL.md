# Claude Agents CLI - Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation Methods

### Method 1: Development Installation (Recommended)

For development and testing:

```bash
cd claude-agents-pro/cli
pip install -e .
```

This creates an editable installation, allowing you to modify the code and see changes immediately.

### Method 2: User Installation

Install for current user only:

```bash
cd claude-agents-pro/cli
pip install --user .
```

### Method 3: System-wide Installation

Install system-wide (requires admin/sudo):

```bash
cd claude-agents-pro/cli
sudo pip install .
```

### Method 4: Virtual Environment (Isolated)

Create isolated environment:

```bash
cd claude-agents-pro/cli

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install
pip install -e .
```

## Verifying Installation

After installation, verify the CLI is available:

```bash
# Check version
claude-agents --version

# Show help
claude-agents --help

# Test list command
claude-agents list
```

## Dependencies

The CLI requires these Python packages (automatically installed):

- **click** (>=8.1.0) - Command-line interface framework
- **rich** (>=13.0.0) - Terminal formatting and styling
- **PyYAML** (>=6.0) - YAML parsing for agent frontmatter

### Manual Dependency Installation

If you encounter issues, install dependencies manually:

```bash
pip install click rich PyYAML
```

## Quick Start

### 1. Install Agents to User Scope

```bash
# Install all agents
claude-agents install

# Or install specific tier
claude-agents install --tier 01-foundation
```

### 2. List Installed Agents

```bash
claude-agents list
```

### 3. Search for Agents

```bash
claude-agents search "API design"
```

### 4. Get Agent Details

```bash
claude-agents info api-platform-engineer
```

### 5. Validate Agent Quality

```bash
claude-agents validate
```

### 6. Score Agent Quality

```bash
claude-agents score
```

## Installation Locations

### User Scope (Default)
- **Location**: `~/.claude/agents/`
- **Config**: `~/.claude/cli-config.json`
- **Use**: Personal development, per-user configuration

### Project Scope
- **Location**: `./.claude/agents/` (current directory)
- **Use**: Project-specific agents, team collaboration

### Repository Scope
- **Location**: `claude-agents-pro/agents/`
- **Use**: Read-only reference, development

## Troubleshooting

### Issue: Command Not Found

**Problem**: After installation, `claude-agents` command not found.

**Solutions**:

1. Ensure Python's scripts directory is in PATH:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. Restart terminal or reload shell:
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

3. Use full path temporarily:
   ```bash
   python3 -m cli --help
   ```

### Issue: Permission Denied

**Problem**: Permission errors during installation.

**Solutions**:

1. Use `--user` flag:
   ```bash
   pip install --user .
   ```

2. Use virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install .
   ```

### Issue: Import Errors

**Problem**: `ModuleNotFoundError` for click, rich, or yaml.

**Solution**: Install dependencies manually:

```bash
pip install click rich PyYAML
```

### Issue: Agents Directory Not Found

**Problem**: CLI can't find agents directory.

**Solutions**:

1. Run from repository root:
   ```bash
   cd claude-agents-pro
   claude-agents list
   ```

2. Specify agents directory explicitly:
   ```bash
   claude-agents list --agents-dir /path/to/agents
   ```

3. Install agents first:
   ```bash
   claude-agents install
   ```

## Uninstallation

### Remove CLI Package

```bash
pip uninstall claude-agents-cli
```

### Clean Up Installed Agents

```bash
# User scope
rm -rf ~/.claude/agents
rm ~/.claude/cli-config.json

# Project scope
rm -rf .claude/agents
```

## Upgrading

### Development Installation

With editable install (`-e`), just pull latest changes:

```bash
cd claude-agents-pro
git pull origin main
# Changes are immediately available
```

### Standard Installation

```bash
cd claude-agents-pro/cli
pip install --upgrade .
```

## Platform-Specific Notes

### macOS

- Use Homebrew Python: `brew install python3`
- Scripts installed to: `~/.local/bin` or `/usr/local/bin`

### Linux

- Use system package manager: `apt install python3-pip`
- May need `python3-venv`: `apt install python3-venv`

### Windows

- Use Python from python.org or Microsoft Store
- Scripts installed to: `%APPDATA%\Python\Scripts`
- Add to PATH in System Environment Variables

## Development Setup

For contributing to the CLI:

```bash
# Clone repository
git clone https://github.com/ubehera/claude-agents-pro.git
cd claude-agents-pro/cli

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linters
black cli/
ruff check cli/
mypy cli/
```

## Support

- **Issues**: https://github.com/ubehera/claude-agents-pro/issues
- **Documentation**: See [README.md](README.md)
- **Contributing**: See [../CONTRIBUTING.md](../CONTRIBUTING.md)
