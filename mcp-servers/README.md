# MCP Server Registry

Curated Model Context Protocol (MCP) server configurations for extending Claude Code with external tools, data sources, and services.

## What are MCP Servers?

MCP servers expose tools, resources, and prompts to Claude Code through a standardized protocol. They enable Claude to interact with databases, APIs, file systems, browsers, and other external services beyond its built-in capabilities.

## Configuration

MCP servers are configured in `.mcp.json` at the project root or `~/.claude/.mcp.json` for global scope.

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@package/mcp-server"],
      "env": {
        "API_KEY": "..."
      }
    }
  }
}
```

## Server Catalog

### Knowledge & Memory

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `memory` | stdio | Persistent knowledge graph for session continuity | `@anthropic/memory-server` |
| `sequential-thinking` | stdio | Complex problem decomposition and multi-step reasoning | `@anthropic/sequential-thinking` |

**memory** — Stores entities, relations, and observations in a persistent knowledge graph. Essential for maintaining project context across sessions.

```json
"memory": {
  "command": "npx",
  "args": ["-y", "@anthropic/memory-server"],
  "env": {
    "MEMORY_FILE": "/path/to/memory.json"
  }
}
```

**sequential-thinking** — Enables structured multi-step reasoning with branching, revision, and hypothesis testing. Ideal for complex debugging, architecture decisions, and multi-factor analysis.

```json
"sequential-thinking": {
  "command": "npx",
  "args": ["-y", "@anthropic/sequential-thinking"]
}
```

### Development Tools

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `playwright` | stdio | Browser automation, testing, screenshots | `@anthropic/playwright-mcp` |
| `context7` | stdio | Library documentation and code examples | `@anthropic/context7-mcp` |
| `Ref` | stdio | Documentation search and URL reading | `@anthropic/ref-mcp` |

**playwright** — Browser automation for E2E testing, screenshot capture, form interaction, and web scraping. Supports Chromium, Firefox, and WebKit.

```json
"playwright": {
  "command": "npx",
  "args": ["-y", "@anthropic/playwright-mcp"]
}
```

**context7** — Retrieves up-to-date documentation and code examples for any library. Resolve library IDs first, then query docs.

```json
"context7": {
  "command": "npx",
  "args": ["-y", "@anthropic/context7-mcp"]
}
```

### Database & Storage

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `postgres` | stdio | PostgreSQL query execution and schema inspection | `@modelcontextprotocol/server-postgres` |
| `sqlite` | stdio | SQLite database operations | `@modelcontextprotocol/server-sqlite` |
| `redis` | stdio | Redis key-value operations | `@modelcontextprotocol/server-redis` |

**postgres** — Execute queries, inspect schemas, and manage PostgreSQL databases directly from Claude Code.

```json
"postgres": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-postgres"],
  "env": {
    "DATABASE_URL": "postgresql://user:pass@localhost:5432/db"
  }
}
```

### Cloud & Infrastructure

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `aws` | stdio | AWS service interaction via CLI | `@modelcontextprotocol/server-aws` |
| `github` | stdio | GitHub API for repos, PRs, issues | `@modelcontextprotocol/server-github` |
| `docker` | stdio | Container management and inspection | `@modelcontextprotocol/server-docker` |

**github** — Full GitHub API access for repository management, PR workflows, issue tracking, and code search.

```json
"github": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_TOKEN": "ghp_..."
  }
}
```

### Search & Web

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `brave-search` | stdio | Web search via Brave Search API | `@anthropic/brave-search-mcp` |
| `fetch` | stdio | URL fetching with markdown conversion | `@anthropic/fetch-mcp` |
| `exa` | stdio | AI-powered semantic web search | `@anthropic/exa-mcp` |

**brave-search** — Web search with domain filtering, freshness controls, and local business search.

```json
"brave-search": {
  "command": "npx",
  "args": ["-y", "@anthropic/brave-search-mcp"],
  "env": {
    "BRAVE_API_KEY": "..."
  }
}
```

### AI & ML

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `jupyter` | stdio | Jupyter notebook execution | `@modelcontextprotocol/server-jupyter` |
| `huggingface` | stdio | HuggingFace model and dataset access | `@modelcontextprotocol/server-huggingface` |

### Communication & Productivity

| Server | Transport | Purpose | Package |
|--------|-----------|---------|---------|
| `slack` | stdio | Slack workspace messaging | `@modelcontextprotocol/server-slack` |
| `linear` | stdio | Linear issue tracking and project management | `@modelcontextprotocol/server-linear` |
| `notion` | stdio | Notion workspace and database access | `@modelcontextprotocol/server-notion` |

## Recommended Configurations

### Minimal (Essential)
```json
{
  "mcpServers": {
    "memory": { "command": "npx", "args": ["-y", "@anthropic/memory-server"] },
    "sequential-thinking": { "command": "npx", "args": ["-y", "@anthropic/sequential-thinking"] }
  }
}
```

### Development
```json
{
  "mcpServers": {
    "memory": { "command": "npx", "args": ["-y", "@anthropic/memory-server"] },
    "sequential-thinking": { "command": "npx", "args": ["-y", "@anthropic/sequential-thinking"] },
    "playwright": { "command": "npx", "args": ["-y", "@anthropic/playwright-mcp"] },
    "context7": { "command": "npx", "args": ["-y", "@anthropic/context7-mcp"] },
    "postgres": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres"] }
  }
}
```

### Full Stack
```json
{
  "mcpServers": {
    "memory": { "command": "npx", "args": ["-y", "@anthropic/memory-server"] },
    "sequential-thinking": { "command": "npx", "args": ["-y", "@anthropic/sequential-thinking"] },
    "playwright": { "command": "npx", "args": ["-y", "@anthropic/playwright-mcp"] },
    "context7": { "command": "npx", "args": ["-y", "@anthropic/context7-mcp"] },
    "brave-search": { "command": "npx", "args": ["-y", "@anthropic/brave-search-mcp"] },
    "postgres": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres"] },
    "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] },
    "docker": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-docker"] }
  }
}
```

## Adding Custom MCP Servers

### Stdio Transport (Local)
```json
{
  "my-server": {
    "command": "node",
    "args": ["./path/to/server.js"],
    "env": { "CONFIG_VAR": "value" }
  }
}
```

### SSE Transport (Remote)
```json
{
  "my-remote-server": {
    "url": "https://my-server.example.com/sse",
    "headers": { "Authorization": "Bearer token" }
  }
}
```

## Best Practices

1. **Environment Variables**: Never hardcode secrets — use `env` field with references to shell variables
2. **Scope Appropriately**: Global (`~/.claude/.mcp.json`) for personal tools, project (`.mcp.json`) for team tools
3. **Minimal Servers**: Only enable servers you actively use — each adds startup time
4. **Version Pinning**: Pin `npx -y package@version` for reproducible setups
5. **Health Checks**: Test server connectivity before relying on it in workflows

---

**Registry Status**: Curated catalog of verified MCP servers
**Categories**: 7 (Knowledge, Development, Database, Cloud, Search, AI/ML, Communication)
**Recommended Minimum**: memory + sequential-thinking
