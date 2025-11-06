# Claude Code Custom Slash Commands

> **Production-ready slash commands for Claude Agents Pro**

These custom slash commands orchestrate multi-agent workflows, automate development tasks, and ensure quality standards across Claude Agents Pro.

## Installation

Commands are automatically available in this project. To install globally:
```bash
# Copy to user commands directory (preserves subdirectories)
cp -r commands/* ~/.claude/commands/

# Or create a symlink for automatic updates
ln -s $(pwd)/commands ~/.claude/commands/ubehera

# Restart Claude Code to load commands
```

## Command Architecture

### Security & Design Principles
- **Least Privilege**: Minimal tool permissions with specific restrictions
- **Agent Delegation**: Complex operations delegated to specialist agents
- **Input Validation**: All user inputs sanitized and validated
- **Quality Gates**: Progressive validation at each workflow stage
- **Error Handling**: Graceful failures with actionable guidance

### Command Structure
```yaml
Frontmatter Fields:
  description: Brief command purpose
  args: <required> [optional] [--flag value]
  tools: Specific tools with restrictions (e.g., Bash(npm test:*))
  model: claude-opus-4-1 or claude-sonnet-4-5
```

## Command Catalog

### 🎯 Meta & Orchestration (`00-meta/`)
- `/workflow` - Execute complete DDD workflow with quality gates
- `/orchestrate` - Coordinate multi-agent workflows
- `/parallel-execution` - Run agents in parallel
- `/team-formation` - Form specialized agent teams
- `/workflow-patterns` - Common workflow patterns

### 🏗️ Foundation (`01-foundation/`)
- `/api` - Design APIs with api-platform-engineer
- `/debug` - Intelligent debugging assistance

### 🤖 Agent Commands (`03-agents/`)
- `/agent <name> <task>` - Direct agent invocation
- `/agents` - List and manage agents

### ✅ Quality & Testing (`quality/`)
- `/test [scope]` - Comprehensive testing with coverage gates
- `/review [scope]` - Code review with multi-level validation
- `/performance [target]` - Performance analysis and optimization
- `/security-audit` - Security assessment and remediation
- `/quality-gates` - Multi-dimensional quality validation
- `/verify-agents` - Validate agent configuration
- `/score-agents` - Quality scoring for agents

### 🔄 Workflows (`workflows/`)
- `/feature-development` - Full-stack feature pipeline
- `/full-stack-feature` - Complete feature implementation
- `/review-and-deploy` - Review and deployment with safety checks

### 🔧 Utilities (`utils/` & `05-utilities/`)
- `/quick-fix <issue>` - Quick fixes for common issues
- `/docs <library>` - Fetch documentation
- `/search` - Multi-source intelligent search
- `/search-agents` - Search for specific agents

### 🚀 Automation & DevOps
- **automation/**
  - `/deploy-pipeline` - CI/CD pipeline setup
  - `/health-check` - Infrastructure health monitoring
  - `/batch-update` - Batch operations with rollback
- **ops/**
  - `/release-prepare` - Automated release preparation
  - `/sync-upstream` - Repository synchronization
- **git/**
  - `/workflow-automation` - Git workflow automation
- **maintenance/**
  - `/cleanup-automation` - Repository cleanup
- **monitoring/**
  - `/performance-monitor` - Performance monitoring
- **setup/**
  - `/install-agents` - Install repository agents
- **review/**
  - `/agent-checklist` - Agent review checklist

## Usage Examples

### Feature Development
```bash
# Complete feature with DDD workflow
/workflow "user authentication system" --level enterprise

# Quick feature implementation
/feature "payment processing" --api-first

# API-first development
/api payment-service --spec openapi --version v2
```

### Quality Assurance
```bash
# Comprehensive testing
/test src/auth --coverage-target 90 --type all

# Strict code review
/review PR-123 --level strict

# Performance optimization
/performance api/users --metric latency --optimize

# Security audit
/security-audit --comprehensive
```

### Agent Coordination
```bash
# Direct agent invocation
/agent python-expert "Optimize data pipeline"

# Multi-agent orchestration
/orchestrate "microservices migration" --quality enterprise

# Parallel agent execution
/parallel-execution "comprehensive refactoring"
```

### Development Utilities
```bash
# Debug complex issues
/debug "TypeError: Cannot read property 'x' of undefined"

# Quick fixes
/quick-fix "eslint errors" --auto-fix

# Documentation lookup
/docs react hooks
```

## Quality Levels

### MVP (Minimum Viable Product)
- 70% quality gates
- Basic validation and testing
- Fast iteration focus

### Standard (Production Ready)
- 85% quality gates
- Comprehensive testing
- Performance validated
- Security reviewed

### Enterprise (Mission Critical)
- 95% quality gates
- Full compliance audits
- Complete documentation
- Extensive monitoring

## Command Development

### Creating New Commands

1. **Choose appropriate directory**:
   - `00-meta/` - Orchestration and coordination
   - `01-foundation/` - Core development commands
   - `03-agents/` - Agent-specific commands
   - `quality/` - Testing and validation
   - `workflows/` - Multi-step processes
   - `utils/` - Helper utilities
   - `automation/` - CI/CD and automation
   - `ops/` - Operations and deployment

2. **Use standard frontmatter**:
   ```yaml
   ---
   description: Clear, concise description
   args: <required> [optional] [--flag value]
   tools: Task, Read  # Minimal permissions
   model: claude-sonnet-4-5  # Or claude-opus-4-1 for complex orchestration
   ---
   ```

3. **Follow security principles**:
   - Validate all inputs
   - Use least privilege for tools
   - Delegate dangerous operations to agents
   - Include error handling

4. **Test thoroughly**:
   ```bash
   # Validate command structure
   ./scripts/verify-commands.sh

   # Test with sample inputs
   /your-command "test input"
   ```

## Integration with Agent Ecosystem

Commands seamlessly integrate with 23 specialized agents:
- **Foundation Tier**: Core development (API, review, test, debug)
- **Development Tier**: Language specialists (Python, TypeScript, frontend)
- **Specialist Tier**: Domain experts (cloud, database, DevOps)
- **Expert Tier**: Advanced capabilities (ML, security, research)

## Best Practices

1. **Start with `/workflow`** for complex multi-phase features
2. **Use `/agent`** for direct specialist invocation
3. **Run `/test`** and `/review`** before deployments
4. **Leverage `/performance`** for optimization
5. **Apply `/quick-fix`** for common issues

## Troubleshooting

### Command Not Found
- Ensure Claude Code is restarted after adding commands
- Check command file has `.md` extension
- Verify frontmatter is valid YAML

### Permission Errors
- Review `tools` declaration in frontmatter
- Use specific tool restrictions (e.g., `Bash(npm:*)`)
- Delegate privileged operations to agents

### Execution Failures
- Check argument format matches `args` pattern
- Verify required agents are installed
- Review error messages for guidance

## Contributing

When adding or modifying commands:
1. Follow the established patterns
2. Update this README
3. Test with various inputs
4. Run validation scripts
5. Document usage examples
