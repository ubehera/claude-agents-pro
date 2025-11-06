# agent-forge

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Agents](https://img.shields.io/badge/agents-34-brightgreen.svg)](./agents/README.md)
[![Commands](https://img.shields.io/badge/commands-35-blue.svg)](./commands/README.md)
[![Quality](https://img.shields.io/badge/quality-validated-success.svg)](./scripts/quality-scorer.py)

Production-ready Claude Code agents with checklists and MCP integrations—build, test, and maintain in one workspace.

## System Architecture

### 🎯 Core Components
- **34 Specialized Agents**: Tiered collection from orchestration to domain experts
- **35 Slash Commands**: Quick access workflows using latest Claude models (Opus 4-1, Sonnet 4-5)
- **Automation Scripts**: Install, verify, and score agent quality
- **MCP Integration**: Memory persistence and advanced reasoning capabilities
- **Quality Framework**: Automated validation, scoring rubric (70+ minimum, 85+ production)

### 📊 Agent Tiers (34 Total)
- **Tier 0 (Meta)**: Multi-agent orchestration and workflow coordination
- **Tier 1 (Foundation)**: Core engineering (API, domain modeling, testing, review, debugging, performance, system design)
- **Tier 2 (Development)**: Language/platform specialists (frontend, mobile, Python, TypeScript)
- **Tier 3 (Specialists)**: Domain experts (cloud, backend, database, DevOps, observability, SRE, data, full-stack)
- **Tier 4 (Experts)**: Machine learning, MLOps, and AI systems
- **Tier 5 (Utilities)**: Search, documentation retrieval, and developer tools
- **Tier 6 (Integration)**: Research, technical documentation, and knowledge management
- **Tier 7 (Quality)**: Security architecture, compliance, and audit

## Quick Start

### For Users
```bash
# Clone and enter the repository
git clone https://github.com/ubehera/agent-forge.git
cd agent-forge

# Install all agents (user-scoped, available across projects)
./scripts/install-agents.sh --user

# Restart Claude Code to load agents
```

### For Developers
```bash
# Install agents locally for testing
./scripts/install-agents.sh --project  # Project-scoped installation

# Validate agent structure and frontmatter
./scripts/verify-agents.sh

# Score agent quality (minimum 70/100, production 85/100)
python3 scripts/quality-scorer.py --agents-dir agents
python3 scripts/quality-scorer.py --agent agents/01-foundation/api-platform-engineer.md

# Test changes
# 1. Restart Claude Code
# 2. Invoke agent with test prompt matching its domain
# 3. Verify tool restrictions and behavior
```

### Agent Discovery
Once installed, agents are automatically selected based on task context:
- **Direct Invocation**: "Use the API platform engineer to design a REST API"
- **Implicit Routing**: "Help me debug this performance issue" → `performance-optimization-specialist`
- **Orchestration**: Complex tasks routed through `agent-coordinator` → specialists

See `agents/README.md` for complete trigger patterns and invocation examples.

## Workflow Patterns

### 🔄 Agent Orchestration
1. **Discovery**: `agent-coordinator` decomposes problems and selects specialists
2. **Implementation**: Foundation/specialist tiers execute domain work
3. **Quality Gates**: Review agents validate readiness (test, security, performance)
4. **Operations**: DevOps agents finalize deployment and monitoring

### 🎯 Common Workflows
- **Feature Development**: `/workflow` → DDD workflow with quality gates
- **API Design**: `/api` → `api-platform-engineer` with OpenAPI/GraphQL
- **Security Review**: `/security-audit` → `security-architect` assessment
- **Performance**: `/performance` → `performance-optimization-specialist`
- **Incident Response**: `sre-incident-responder` → diagnosis → mitigation

## Key Documentation
- `agents/README.md`: Complete agent catalog with invocation triggers
- `agents/AGENT_CHECKLIST.md`: Pre-flight checklist for agent updates
- `agents/TESTING.md`: Comprehensive testing procedures
- `commands/README.md`: Slash command catalog and usage
- `patterns/orchestration/`: Multi-agent coordination patterns
- `prompts/CLAUDE.md`: Operating instructions for Claude Code

## License
- Code, documentation, and agents are licensed under Apache License 2.0. See `LICENSE`.

## MCP Config
- Location: `./.mcp.json` (project-level). Claude Code merges this with your user-level `~/.mcp.json`.
- Included by default: `memory` and `sequential-thinking` servers via `npx`.
- Enable more servers by editing `./.mcp.json`. Example (disabled by default):
  ```json
  {
    "mcpServers": {
      "aws-docs": {
        "command": "uvx",
        "args": ["awslabs.aws-documentation-mcp-server@latest"],
        "env": { "AWS_REGION": "us-east-1" },
        "disabled": true
      }
    }
  }
  ```
- Tips: do not commit secrets; use environment variables. Toggle servers with `"disabled": true|false`.
