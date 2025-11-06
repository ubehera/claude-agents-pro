---
description: Comprehensive health check for Claude Agents Pro infrastructure
args: [--scope agents|scripts|configs] [--fix-issues] [--report-format json|text]
tools: Bash(./scripts/verify-agents.sh:*), Bash(./scripts/quality-scorer.py:*), Read, Grep
model: claude-sonnet-4-5
---

## Objective
Perform comprehensive health checks across the Claude Agents Pro ecosystem including agents, scripts, configurations, and dependencies.

## Before You Run
- Ensure all project dependencies are installed
- Verify you're in the repository root directory
- Check that all required tools are available in PATH

## Execution
Run comprehensive health check:

```bash
# Full system health check
!/health-check

# Focused scope with auto-fix
!/health-check --scope agents --fix-issues

# Generate JSON report
!/health-check --report-format json > health-report.json
```

## Health Check Components

### Agent Health
- **Frontmatter Validation**: YAML syntax and required fields
- **Tool Permissions**: Minimal privilege adherence
- **File Integrity**: Missing files and broken references
- **Quality Metrics**: Scoring against established rubrics

### Script Health
- **Executable Permissions**: Proper script permissions
- **Dependency Validation**: Required tools and libraries
- **Error Handling**: Robust error management
- **Idempotency**: Safe repeated execution

### Configuration Health
- **MCP Server Status**: Connection and functionality
- **Git Configuration**: Repository settings and hooks
- **Environment Variables**: Required variables present
- **File Permissions**: Security and access controls

## Diagnostic Output
- **Green**: All checks passed
- **Yellow**: Warnings that should be addressed
- **Red**: Critical issues requiring immediate attention
- **Blue**: Informational messages and suggestions

## Auto-Fix Capabilities
- Fix file permissions on scripts
- Repair broken symlinks
- Update outdated configurations
- Standardize agent frontmatter

## Follow Up
- Address any critical (red) issues immediately
- Schedule regular health checks in CI/CD pipeline
- Document any persistent warnings
- Update health check criteria as project evolves
