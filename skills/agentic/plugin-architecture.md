---
name: plugin-architecture
description: Load when designing or creating Claude Code plugins with commands, skills, agents, and hooks
trigger_keywords: [plugin, plugin.json, plugin architecture, claude code plugin, plugin structure, plugin manifest, plugin commands, plugin development]
---

# Plugin Architecture Skill

Patterns for designing and building Claude Code plugins that package commands, skills, agents, hooks, and MCP servers as distributable extensions.

## Overview

Claude Code plugins are self-contained packages that extend Claude Code with new capabilities. They follow a manifest-driven architecture with auto-discovery of components.

**When to Use**:
- Building reusable tooling for teams or organizations
- Packaging domain-specific workflows
- Creating shareable automation suites
- Extending Claude Code with custom capabilities

## Plugin Structure

```
my-plugin/
├── plugin.json              # Manifest (required)
├── README.md                # Documentation
├── commands/                # Slash commands
│   ├── deploy.md
│   └── review.md
├── skills/                  # Knowledge modules
│   ├── domain-patterns.md
│   └── api-standards.md
├── agents/                  # Subagent definitions
│   └── domain-expert.md
├── hooks/                   # Event-driven automation
│   ├── pre-commit.md
│   └── post-deploy.md
├── mcp-servers/             # MCP server configs
│   └── .mcp.json
└── my-plugin.local.md       # User-local settings (gitignored)
```

## Plugin Manifest (plugin.json)

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "Domain-specific automation for my team",
  "author": "team-name",

  "commands": {
    "deploy": {
      "path": "commands/deploy.md",
      "description": "Deploy to staging or production"
    },
    "review": {
      "path": "commands/review.md",
      "description": "Run comprehensive code review"
    }
  },

  "skills": {
    "domain-patterns": {
      "path": "skills/domain-patterns.md"
    }
  },

  "agents": {
    "domain-expert": {
      "path": "agents/domain-expert.md"
    }
  },

  "hooks": {
    "pre-commit": {
      "path": "hooks/pre-commit.md",
      "event": "PreToolUse"
    },
    "post-deploy": {
      "path": "hooks/post-deploy.md",
      "event": "PostToolUse"
    }
  }
}
```

## Component Design

### Commands (User-Invokable)

```markdown
---
name: deploy
description: Deploy application to target environment
arguments:
  - name: environment
    description: Target environment (staging, production)
    required: true
  - name: version
    description: Version tag to deploy
    required: false
---

# Deploy Command

Deploy the application to the specified environment with safety checks.

## Steps

1. Verify current branch is clean (no uncommitted changes)
2. Run test suite: `npm test`
3. Build application: `npm run build`
4. If environment is production:
   - Require explicit confirmation
   - Check that staging tests passed
   - Create git tag with version
5. Deploy: `./scripts/deploy.sh {{ environment }} {{ version }}`
6. Verify deployment health check
7. Notify team via Slack
```

### Skills (Knowledge Modules)

```markdown
---
name: domain-patterns
description: Load when working with our domain-specific business logic
trigger_keywords: [order, payment, inventory, shipping, customer]
---

# Domain Patterns

## Order Processing Rules
- Orders under $100 auto-approve
- Orders $100-$1000 require manager approval
- Orders >$1000 require VP approval and fraud check

## Payment Integration
- Stripe for US customers
- Adyen for EU customers
- PayPal as fallback

## Business Invariants
- Inventory must be reserved before payment
- Shipping address validated against USPS API
- Tax calculated per destination state
```

### Agents (Autonomous Specialists)

```markdown
---
name: domain-expert
description: Expert in our business domain logic, data models, and integration patterns
---

You are a domain expert for [Company Name]'s e-commerce platform.

## Domain Knowledge
- Order lifecycle: draft → submitted → paid → fulfilled → shipped → delivered
- Inventory: real-time sync with warehouse API
- Payments: Stripe primary, Adyen failover

## When Activated
- Questions about business rules
- Data model design for domain entities
- Integration patterns with external services

## Delegation
- For infrastructure → delegate to cloud architect
- For frontend → delegate to frontend expert
- For security → delegate to security architect
```

### Hooks (Event-Driven)

```markdown
---
name: pre-commit
description: Enforce team standards before git commits
event: PreToolUse
tools: ["Bash"]
---

# Pre-Commit Hook

## Trigger
Activate when Bash contains `git commit`.

## Checks
1. Conventional commit format required
2. No TODO comments in staged files
3. No console.log in production code
4. All imports sorted (auto-fix)

## Actions
- BLOCK if commit message invalid
- WARN if TODOs found
- Auto-fix import sorting
```

## Local Settings Pattern

```markdown
<!-- my-plugin.local.md (gitignored) -->
---
environment: staging
slack_channel: "#team-deploys"
auto_deploy: false
review_strictness: high
---

Local configuration for my-plugin.
This file is gitignored and contains user-specific settings.
```

## Plugin Installation

```bash
# From directory
claude plugins add ./my-plugin

# From GitHub
claude plugins add github:org/my-plugin

# List installed
claude plugins list

# Remove
claude plugins remove my-plugin
```

## Best Practices

1. **Single responsibility** — one plugin per domain/workflow
2. **Manifest first** — define plugin.json before implementing components
3. **Local settings** — use .local.md for user-specific configuration (gitignore it)
4. **`${CLAUDE_PLUGIN_ROOT}`** — use this variable for relative paths in commands
5. **Progressive disclosure** — skills load on-demand, not all at once
6. **Test commands** — verify each command works in isolation before packaging
7. **Document triggers** — make it clear when each component activates

---

**Skill Type**: Agentic — Plugins
**Complexity**: Moderate
**Typical Usage**: Plugin design, component packaging, team tooling distribution
