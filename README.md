# Claude Agents Pro

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Agents](https://img.shields.io/badge/agents-61-brightgreen.svg)](./agents/README.md)
[![Skills](https://img.shields.io/badge/skills-120-purple.svg)](./skills/README.md)
[![Commands](https://img.shields.io/badge/commands-35-blue.svg)](./commands/README.md)
[![Hooks](https://img.shields.io/badge/hooks-10-orange.svg)](./hooks/README.md)
[![Quality](https://img.shields.io/badge/quality-validated-success.svg)](./scripts/quality-scorer.py)

Personal collection of production-ready Claude Code agents with automated quality validation and multi-tier orchestration.

## System Architecture

### Core Components
- **61 Specialized Agents**: Tiered collection from orchestration to domain experts
- **120 Progressive-Disclosure Skills**: Modular knowledge packages across 17 domains (~70% token savings)
- **35 Slash Commands**: Quick access workflows using latest Claude models (Opus 4.6, Sonnet 4.5)
- **10 Event-Driven Hooks**: Security, quality, git, and workflow automation
- **Automation Scripts**: Install, verify, and score agent quality
- **MCP Integration**: Memory persistence and advanced reasoning capabilities
- **Quality Framework**: Automated validation, scoring rubric (70+ minimum, 85+ production)

### 📊 Agent Tiers (61 Total)
- **Tier 0 (Meta)**: Multi-agent orchestration and workflow coordination
- **Tier 1 (Foundation)**: Core engineering (API, domain modeling, testing, review, debugging, performance, system design)
- **Tier 2 (Development)**: Language/platform specialists (frontend, mobile, Python, TypeScript, Angular, Django, Rails, Laravel, Java, Kotlin, C#)
- **Tier 3 (Specialists)**: Domain experts (AWS/Azure/GCP cloud, backend, database, DevOps, observability, SRE, data, full-stack)
- **Tier 4 (Experts)**: Machine learning, MLOps, LLM architecture, and prompt engineering
- **Tier 5 (Platform)**: Developer experience optimization, Git workflow automation
- **Tier 6 (Integration)**: Research, technical documentation, and knowledge management
- **Tier 7 (Quality)**: Security architecture, compliance, and audit
- **Tier 8 (Finance)**: Trading, risk management, compliance, quantitative analysis, portfolio management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ubehera/claude-agents-pro.git
cd claude-agents-pro

# Install agents (user-scoped, available across all projects)
./scripts/install-agents.sh --user

# Or install project-scoped (only in current directory)
./scripts/install-agents.sh --project

# Restart Claude Code to load agents
```

### Validation & Quality

```bash
# Validate agent structure and frontmatter
./scripts/verify-agents.sh

# Score agent quality (minimum 70/100, production 85/100)
python3 scripts/quality-scorer.py --agents-dir agents
python3 scripts/quality-scorer.py --agent agents/01-foundation/api-platform-engineer.md
```

### Agent Discovery

Once installed, agents are automatically selected based on task context:
- **Direct Invocation**: "Use the API platform engineer to design a REST API"
- **Implicit Routing**: "Help me debug this performance issue" → `performance-optimization-specialist`
- **Orchestration**: Complex tasks routed through `orchestration-coordinator` → specialists

See `agents/README.md` for complete trigger patterns and invocation examples.

## Workflow Patterns

### 🔄 Agent Orchestration
1. **Discovery**: `orchestration-coordinator` decomposes problems and selects specialists
2. **Implementation**: Foundation/specialist tiers execute domain work
3. **Quality Gates**: Review agents validate readiness (test, security, performance)
4. **Operations**: DevOps agents finalize deployment and monitoring

### 🎯 Common Workflows
- **Feature Development**: `/workflows:feature-development` → DDD workflow with quality gates
- **API Design**: `/01-foundation:api` → `api-platform-engineer` with OpenAPI/GraphQL
- **Security Review**: `/quality:security-audit` → `security-architect` assessment
- **Performance**: `/quality:performance` → `performance-optimization-specialist`
- **Debugging**: `/01-foundation:debug` → `error-diagnostician` intelligent debugging

## Key Documentation
- `agents/README.md`: Complete agent catalog with invocation triggers
- `agents/AGENT_CHECKLIST.md`: Pre-flight checklist for agent updates
- `agents/TESTING.md`: Comprehensive testing procedures
- `skills/README.md`: Skills catalog (120 skills across 17 domains)
- `hooks/README.md`: Hook development guide (10 event-driven hooks)
- `mcp-servers/README.md`: MCP server registry (20+ verified servers)
- `commands/README.md`: Slash command catalog and usage
- `patterns/orchestration/`: Multi-agent coordination patterns
- `prompts/CLAUDE.md`: Operating instructions for Claude Code
- `SYSTEM_OVERVIEW.md`: Architecture, tier system, and coordination patterns

### User-Level Configuration Template

**`meta-CLAUDE.md`** - Template for user-level Claude Code configuration (deploys to `~/.claude/CLAUDE.md`)

This file provides:
- Persona and communication style settings
- Memory management and session continuity patterns
- Tool usage optimization and MCP integration
- DDD workflow patterns and quality gates
- Agent orchestration and delegation examples

**Important**: This is NOT project-level documentation. When deployed to `~/.claude/`, it applies to all Claude Code sessions across all projects. The "meta-" prefix is removed during installation.

## MCP Configuration

Location: `./.mcp.json` (project-level). Claude Code merges this with your user-level `~/.mcp.json`.

**Included by default:**
- `memory`: Persistent knowledge graph for session continuity
- `sequential-thinking`: Complex problem decomposition

**Enable additional servers** by editing `./.mcp.json`:
```json
{
  "mcpServers": {
    "aws-docs": {
      "command": "uvx",
      "args": ["awslabs.aws-documentation-mcp-server@latest"],
      "env": { "AWS_REGION": "us-east-1" },
      "disabled": false
    }
  }
}
```

**Tips:**
- Do not commit secrets; use environment variables
- Toggle servers with `"disabled": true|false`
- All agents now inherit MCP tools automatically (no explicit `tools:` field)

## License

Apache License 2.0. See `LICENSE` for details.
