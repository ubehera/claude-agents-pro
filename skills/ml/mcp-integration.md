---
name: mcp-integration
description: Integrate Model Context Protocol (MCP) servers into Claude Code plugins for external tool and service integration. Use when adding MCP servers, configuring external services, or setting up tool access.
---

# MCP Integration for Claude Code Plugins

## Overview

Model Context Protocol (MCP) enables Claude Code plugins to integrate with external services and APIs by providing structured tool access.

**Key capabilities:**
- Connect to external services (databases, APIs, file systems)
- Provide multiple related tools from a single service
- Handle OAuth and complex authentication flows
- Bundle MCP servers with plugins for automatic setup

## MCP Server Types

| Type | Transport | Best For | Auth |
|------|-----------|----------|------|
| stdio | Process | Local tools, custom servers | Env vars |
| SSE | HTTP | Hosted services, cloud APIs | OAuth |
| HTTP | REST | API backends, token auth | Tokens |
| ws | WebSocket | Real-time, streaming | Tokens |

## Configuration Methods

### Method 1: Dedicated .mcp.json (Recommended)

```json
{
  "database-tools": {
    "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",
    "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
    "env": {
      "DB_URL": "${DB_URL}"
    }
  }
}
```

### Method 2: Inline in plugin.json

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "mcpServers": {
    "plugin-api": {
      "command": "${CLAUDE_PLUGIN_ROOT}/servers/api-server",
      "args": ["--port", "8080"]
    }
  }
}
```

## Server Type Examples

### stdio (Local Process)
```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"],
    "env": { "LOG_LEVEL": "debug" }
  }
}
```

### SSE (Server-Sent Events)
```json
{
  "asana": {
    "type": "sse",
    "url": "https://mcp.asana.com/sse"
  }
}
```

### HTTP (REST API)
```json
{
  "api-service": {
    "type": "http",
    "url": "https://api.example.com/mcp",
    "headers": {
      "Authorization": "Bearer ${API_TOKEN}"
    }
  }
}
```

### WebSocket (Real-time)
```json
{
  "realtime-service": {
    "type": "ws",
    "url": "wss://mcp.example.com/ws"
  }
}
```

## Tool Naming Convention

**Format:** `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`

Pre-allow specific MCP tools in commands:
```markdown
---
allowed-tools: [
  "mcp__plugin_asana_asana__asana_create_task",
  "mcp__plugin_asana_asana__asana_search_tasks"
]
---
```

## Security Best Practices

**DO:**
- Use ${CLAUDE_PLUGIN_ROOT} for portable paths
- Use environment variables for tokens
- Use secure connections (HTTPS/WSS)
- Pre-allow specific MCP tools, not wildcards

**DON'T:**
- Hardcode absolute paths or credentials
- Commit tokens to git
- Use HTTP instead of HTTPS
- Pre-allow all tools with wildcards

## Debugging

```bash
claude --debug
```

Use `/mcp` command to see all servers and available tools.

## Resources

- **Official MCP Docs**: https://modelcontextprotocol.io/
- **Claude Code MCP Docs**: https://docs.claude.com/en/docs/claude-code/mcp
