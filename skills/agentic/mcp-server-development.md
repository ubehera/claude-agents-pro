---
name: mcp-server-development
description: Load when building custom MCP servers or integrating existing MCP servers into Claude Code workflows
trigger_keywords: [mcp server, model context protocol, mcp tool, build mcp, custom mcp, mcp integration, stdio transport, sse transport, mcp.json]
---

# MCP Server Development Skill

Patterns for building custom MCP servers and integrating existing servers into Claude Code workflows.

## Overview

Model Context Protocol (MCP) servers expose tools, resources, and prompts to LLM applications through a standardized protocol. This skill covers building custom servers and configuring pre-built ones.

**When to Use**:
- Building custom tools for Claude Code
- Integrating external APIs/databases as MCP servers
- Configuring `.mcp.json` for project workflows
- Understanding MCP server capabilities and limitations

## MCP Architecture

```
Claude Code ←→ MCP Client ←→ MCP Server ←→ External Service

Transport Options:
  stdio:  Local process (node, python, binary)
  SSE:    Remote HTTP server
  HTTP:   Streamable HTTP endpoint
```

### Server Capabilities

| Capability | Purpose | Example |
|-----------|---------|---------|
| **Tools** | Actions the model can invoke | `execute_query`, `send_email` |
| **Resources** | Read-only data sources | `file://`, `postgres://`, `https://` |
| **Prompts** | Reusable prompt templates | `summarize_code`, `review_pr` |

## Building a Custom MCP Server (TypeScript)

```typescript
// server.ts
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const server = new McpServer({
  name: 'my-custom-server',
  version: '1.0.0',
});

// Register a tool
server.tool(
  'search_tickets',
  'Search support tickets by keyword or status',
  {
    query: z.string().describe('Search query'),
    status: z.enum(['open', 'closed', 'in_progress']).optional(),
    limit: z.number().default(10),
  },
  async ({ query, status, limit }) => {
    const tickets = await ticketDb.search({ query, status, limit });
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(tickets, null, 2),
      }],
    };
  }
);

// Register a resource
server.resource(
  'ticket',
  'ticket://{id}',
  async (uri) => {
    const id = uri.pathname.replace('//', '');
    const ticket = await ticketDb.getById(id);
    return {
      contents: [{
        uri: uri.href,
        mimeType: 'application/json',
        text: JSON.stringify(ticket),
      }],
    };
  }
);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

## Building a Custom MCP Server (Python)

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-custom-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_tickets",
            description="Search support tickets",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "closed"]},
                },
                "required": ["query"],
            },
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_tickets":
        results = await ticket_db.search(arguments["query"])
        return [TextContent(type="text", text=str(results))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Configuration in `.mcp.json`

```json
{
  "mcpServers": {
    "my-custom-server": {
      "command": "node",
      "args": ["./mcp-servers/my-server/dist/server.js"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/tickets",
        "API_KEY": "${TICKET_API_KEY}"
      }
    },
    "remote-server": {
      "url": "https://mcp.example.com/sse",
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN}"
      }
    }
  }
}
```

## Integration Patterns

### Database Query Tool
```typescript
server.tool('query', 'Execute read-only SQL query', {
  sql: z.string().describe('SQL SELECT query'),
}, async ({ sql }) => {
  // Safety: only allow SELECT
  if (!sql.trim().toUpperCase().startsWith('SELECT')) {
    return { content: [{ type: 'text', text: 'Error: Only SELECT queries allowed' }] };
  }
  const result = await pool.query(sql);
  return { content: [{ type: 'text', text: JSON.stringify(result.rows) }] };
});
```

### API Wrapper Tool
```typescript
server.tool('create_issue', 'Create a GitHub issue', {
  title: z.string(),
  body: z.string(),
  labels: z.array(z.string()).optional(),
}, async ({ title, body, labels }) => {
  const response = await octokit.rest.issues.create({
    owner: REPO_OWNER, repo: REPO_NAME,
    title, body, labels,
  });
  return { content: [{ type: 'text', text: `Created issue #${response.data.number}` }] };
});
```

## Best Practices

1. **Validate inputs** — use Zod schemas with clear descriptions
2. **Read-only by default** — require explicit confirmation for writes
3. **No secrets in code** — use `env` field in `.mcp.json` config
4. **Descriptive tool names** — Claude uses the name and description to decide when to use tools
5. **Error handling** — return clear error messages, not stack traces
6. **Timeout handling** — set reasonable timeouts for external calls
7. **Scope minimally** — project `.mcp.json` for project-specific, `~/.claude/.mcp.json` for global

---

**Skill Type**: Agentic — MCP
**Complexity**: Moderate
**Typical Usage**: Building custom MCP servers, configuring MCP integrations
