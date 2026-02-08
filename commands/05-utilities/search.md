---
description: Intelligent code and documentation search with context-aware results
tools: Grep, Glob, Read, WebSearch, Task
model: sonnet
args: [pattern] [scope]
---

# Intelligent Search Command

Context-aware search across codebase, documentation, and external resources with smart result ranking.

## Usage Patterns

### Code Search
```
/search "authentication patterns" agents
/search "GraphQL resolvers" typescript
/search "error handling" --type=function
```

### Documentation Search
```
/search "API versioning" docs
/search "deployment patterns" infrastructure
/search "testing strategies" --scope=project
```

### External Research
```
/search "microservices best practices" --external
/search "Node.js security" --recent
/search "AWS Lambda patterns" --docs=aws
```

## Command Logic

1. **Pattern Analysis**: Parse search terms and intent
2. **Scope Detection**: Determine search boundaries and context
3. **Multi-Source Search**: Combine local and external results
4. **Result Ranking**: Context-aware relevance scoring
5. **Synthesis**: Provide actionable insights and references

## Search Scopes

### Local Scopes
- **agents**: Search agent definitions and capabilities
- **code**: Source code and implementation patterns
- **docs**: Documentation and specifications
- **config**: Configuration files and infrastructure
- **tests**: Test files and testing patterns

### External Scopes
- **web**: General web search for latest information
- **docs**: Official documentation sites
- **standards**: Technical standards and RFCs
- **research**: Academic papers and technical reports

## Search Types

### Pattern Matching
- **Exact**: Quoted strings for precise matches
- **Fuzzy**: Similar terms and concepts
- **Regex**: Pattern-based matching for complex queries
- **Semantic**: Intent-based search with context

### Result Filtering
- **File Types**: Filter by extension or file category
- **Date Range**: Recent changes or historical context
- **Quality Score**: Rank by relevance and authority
- **Project Scope**: Limit to specific modules or domains

## Integration Points

- **research-librarian**: Delegate complex research queries
- **MCP Memory**: Leverage stored project context
- **WebSearch**: External information retrieval
- **Agent Knowledge**: Tap into specialist agent expertise

## Output Formats

### Summary Format
- Top 3-5 most relevant results
- Context snippets with line numbers
- Related concepts and suggestions
- Quick action recommendations

### Detailed Format
- Complete search results with scores
- File paths and exact locations
- Cross-references and dependencies
- External resource links and citations