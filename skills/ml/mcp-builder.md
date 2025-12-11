---
name: mcp-builder
description: Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services.
---

# MCP Server Development Guide

## Overview

Create high-quality MCP servers that enable LLMs to effectively interact with external services. The quality of an MCP server is measured by how well it enables LLMs to accomplish real-world tasks.

## High-Level Workflow

### Phase 1: Deep Research and Planning

#### Agent-Centric Design Principles

**Build for Workflows, Not Just API Endpoints:**
- Don't simply wrap existing API endpoints - build thoughtful, high-impact workflow tools
- Consolidate related operations (e.g., `schedule_event` that both checks availability and creates event)
- Focus on tools that enable complete tasks, not just individual API calls

**Optimize for Limited Context:**
- Agents have constrained context windows - make every token count
- Return high-signal information, not exhaustive data dumps
- Provide "concise" vs "detailed" response format options
- Default to human-readable identifiers over technical codes

**Design Actionable Error Messages:**
- Error messages should guide agents toward correct usage patterns
- Suggest specific next steps: "Try using filter='active_only' to reduce results"
- Make errors educational, not just diagnostic

**Use Evaluation-Driven Development:**
- Create realistic evaluation scenarios early
- Let agent feedback drive tool improvements
- Prototype quickly and iterate based on actual agent performance

#### Study MCP Protocol Documentation

Use WebFetch to load: `https://modelcontextprotocol.io/llms-full.txt`

#### Create Implementation Plan

**Tool Selection:**
- List the most valuable endpoints/operations to implement
- Prioritize tools that enable the most common use cases
- Consider which tools work together to enable complex workflows

**Input/Output Design:**
- Define input validation models (Pydantic for Python, Zod for TypeScript)
- Design consistent response formats (JSON or Markdown)
- Plan for large-scale usage and character limits

### Phase 2: Implementation

#### Set Up Project Structure

**For Python:**
```python
from mcp.server import Server
from pydantic import BaseModel, Field

server = Server("my-mcp-server")

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")

@server.tool()
async def search(input: SearchInput) -> str:
    """Search for items matching the query."""
    # Implementation
    pass
```

**For Node/TypeScript:**
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { z } from "zod";

const server = new Server({ name: "my-mcp-server", version: "1.0.0" });

const SearchInputSchema = z.object({
  query: z.string().describe("Search query string"),
  limit: z.number().min(1).max(100).default(10).describe("Maximum results")
}).strict();

server.registerTool({
  name: "search",
  description: "Search for items matching the query",
  inputSchema: SearchInputSchema,
  handler: async (input) => {
    // Implementation
  }
});
```

#### Implement Core Infrastructure First

- API request helper functions
- Error handling utilities
- Response formatting functions (JSON and Markdown)
- Pagination helpers
- Authentication/token management

#### Tool Implementation Guidelines

**Write Comprehensive Docstrings:**
- One-line summary of what the tool does
- Detailed explanation of purpose and functionality
- Explicit parameter types with examples
- Complete return type schema
- Usage examples (when to use, when not to use)
- Error handling documentation

**Add Tool Annotations:**
- `readOnlyHint`: true (for read-only operations)
- `destructiveHint`: false (for non-destructive operations)
- `idempotentHint`: true (if repeated calls have same effect)
- `openWorldHint`: true (if interacting with external systems)

### Phase 3: Review and Refine

#### Code Quality Review

- **DRY Principle**: No duplicated code between tools
- **Composability**: Shared logic extracted into functions
- **Consistency**: Similar operations return similar formats
- **Error Handling**: All external calls have error handling
- **Type Safety**: Full type coverage
- **Documentation**: Every tool has comprehensive docstrings

#### Testing

**Important:** MCP servers are long-running processes. Running them directly will cause your process to hang indefinitely.

**Safe testing approaches:**
- Use the evaluation harness (recommended)
- Run the server in tmux to keep it outside your main process
- Use a timeout when testing: `timeout 5s python server.py`

### Phase 4: Create Evaluations

Create 10 evaluation questions that test whether LLMs can effectively use your MCP server.

**Each question must be:**
- Independent: Not dependent on other questions
- Read-only: Only non-destructive operations required
- Complex: Requiring multiple tool calls
- Realistic: Based on real use cases
- Verifiable: Single, clear answer

**Output Format:**
```xml
<evaluation>
  <qa_pair>
    <question>Find discussions about AI model launches...</question>
    <answer>3</answer>
  </qa_pair>
</evaluation>
```

## Best Practices

1. **Design for agents first** - Think about what the LLM needs
2. **Consolidate related operations** - One tool per workflow, not per endpoint
3. **Return actionable data** - High signal, not exhaustive dumps
4. **Handle errors gracefully** - Guide agents to correct usage
5. **Test with real agents** - Evaluation-driven development
6. **Document thoroughly** - Every tool needs comprehensive docs
7. **Respect context limits** - Implement truncation and pagination
