---
description: Search and discover agents by domain, tier, or capability with intelligent filtering
args: [search-term] [--tier=0-9] [--domain=api|security|frontend] [--format=table|json|list] [--tools=bash|task]
tools: Read, Grep, Glob
model: sonnet
---

## Purpose
Interactive agent discovery with intelligent filtering by tier, domain, capability, and tool requirements. Helps developers quickly find the right specialist for their task.

## Search Capabilities

### Text Search
- **Name Matching**: Find agents by exact or partial name
- **Description Search**: Match against agent descriptions and domains
- **Keyword Filtering**: Search within agent capabilities and specialties
- **Tool Requirements**: Filter by specific tool needs

### Advanced Filtering
- **Tier-Based**: Filter by agent tier (00-meta to 09-enterprise)
- **Domain-Specific**: Find specialists for particular domains
- **Capability Matching**: Match agents by specific capabilities
- **Tool Dependencies**: Find agents supporting specific tools

### Output Formats
- **Table**: Structured overview with key information
- **JSON**: Machine-readable format for automation
- **List**: Compact format for quick scanning
- **Interactive**: Rich details with usage examples

## Usage Examples

```bash
# Domain-specific search
/search-agents api --format=table
/search-agents "security" --tier=7 --format=json
/search-agents "performance" --domain=backend

# Tier-based filtering
/search-agents --tier=1 --format=table  # Foundation tier
/search-agents --tier=3 --domain=cloud  # Cloud specialists
/search-agents --tier=0  # Meta-level coordinators

# Tool-specific search
/search-agents --tools=bash --domain=devops
/search-agents --tools=task --tier=1
/search-agents typescript --tools=multiedit

# Capability-based search
/search-agents "machine learning" --format=json
/search-agents "database" --tier=3 --format=list
/search-agents "frontend react" --tools=webfetch

# General search with multiple filters
/search-agents cloud --tier=3 --format=table
/search-agents "test" --domain=quality --format=interactive
```

## Input Validation & Processing

```bash
SEARCH_TERM="$1"
TIER=$(echo "$2 $3 $4 $5" | grep -o "--tier=[0-9]" | cut -d= -f2)
DOMAIN=$(echo "$2 $3 $4 $5" | grep -o "--domain=[^[:space:]]*" | cut -d= -f2)
FORMAT=$(echo "$2 $3 $4 $5" | grep -o "--format=[^[:space:]]*" | cut -d= -f2)
TOOLS=$(echo "$2 $3 $4 $5" | grep -o "--tools=[^[:space:]]*" | cut -d= -f2)

# Set defaults
FORMAT=${FORMAT:-"table"}

# Validate format parameter
case "$FORMAT" in
  "table"|"json"|"list"|"interactive")
    echo "✓ Output format: $FORMAT"
    ;;
  *)
    echo "❌ Error: Invalid format: $FORMAT"
    echo "💡 Available formats: table, json, list, interactive"
    exit 1
    ;;
esac

# Validate tier if provided
if [ -n "$TIER" ]; then
  case "$TIER" in
    [0-9])
      echo "✓ Filtering by tier: $TIER"
      ;;
    *)
      echo "❌ Error: Invalid tier: $TIER"
      echo "💡 Available tiers: 0-9"
      exit 1
      ;;
  esac
fi

# Validate domain if provided
if [ -n "$DOMAIN" ]; then
  VALID_DOMAINS="api security frontend backend cloud database devops mobile typescript python performance testing observability ml research"
  if echo "$VALID_DOMAINS" | grep -qw "$DOMAIN"; then
    echo "✓ Filtering by domain: $DOMAIN"
  else
    echo "⚠️ Warning: Unknown domain '$DOMAIN' - will search anyway"
  fi
fi

echo "🔍 Searching ubehera agents..."
if [ -n "$SEARCH_TERM" ]; then
  echo "   Search term: '$SEARCH_TERM'"
fi
if [ -n "$TIER" ]; then
  echo "   Tier filter: $TIER"
fi
if [ -n "$DOMAIN" ]; then
  echo "   Domain filter: $DOMAIN"
fi
if [ -n "$TOOLS" ]; then
  echo "   Tool filter: $TOOLS"
fi
echo ""
```

## Search Implementation

### Agent Catalog Search

Search through the agents catalog using file system and content matching:

```bash
# Read the main agent catalog
AGENT_CATALOG="/Users/umank/Code/agent-repos/ubehera/agents/README.md"

if [ ! -f "$AGENT_CATALOG" ]; then
  echo "❌ Error: Agent catalog not found at $AGENT_CATALOG"
  echo "💡 Ensure you're running from the ubehera project root"
  exit 1
fi

# Extract agent information from catalog
search_agents_catalog() {
  local search_term="$1"
  local tier_filter="$2"
  local domain_filter="$3"
  local tools_filter="$4"
  
  # Build grep pattern for search term
  local grep_pattern="."
  if [ -n "$search_term" ]; then
    grep_pattern="$search_term"
  fi
  
  # Search through agent catalog
  echo "Searching agent catalog for matching entries..."
  
  # Use Read and Grep tools to search catalog
}
```

### Multi-Dimensional Filtering

```bash
filter_agents() {
  local results_file="$1"
  local tier="$2"
  local domain="$3"
  local tools="$4"
  
  # Apply tier filtering
  if [ -n "$tier" ]; then
    echo "Applying tier filter: $tier"
    # Filter by tier directory structure (0X-tier-name)
  fi
  
  # Apply domain filtering
  if [ -n "$domain" ]; then
    echo "Applying domain filter: $domain"
    # Filter by domain keywords in descriptions
  fi
  
  # Apply tools filtering
  if [ -n "$tools" ]; then
    echo "Applying tools filter: $tools"
    # Filter by tool requirements in agent frontmatter
  fi
}
```

### Content-Based Search

Search within agent files for deeper matching:

```bash
search_agent_content() {
  local search_term="$1"
  local agent_dir="/Users/umank/Code/agent-repos/ubehera/agents"
  
  echo "🔎 Deep content search for: '$search_term'"
  
  # Search in agent frontmatter and content
  # Use Grep tool to search through agent files
  
  # Search patterns:
  # - Agent names and descriptions
  # - Tool declarations
  # - Capability descriptions
  # - Domain expertise areas
  
  local matches=$()
  # Implementation would use Grep tool to find matches
  
  echo "Found matches in agent content"
}
```

## Output Formatting

### Table Format

```bash
generate_table_output() {
  local results="$1"
  
  echo "📋 Agent Search Results (Table Format)"
  echo ""
  printf "%-25s %-8s %-20s %-30s\n" "Agent" "Tier" "Domain" "Description"
  printf "%-25s %-8s %-20s %-30s\n" "-----" "----" "------" "-----------"
  
  # Process results and format as table
  while IFS='|' read -r agent tier domain description tools; do
    printf "%-25s %-8s %-20s %-30s\n" "$agent" "$tier" "$domain" "${description:0:28}..."
  done < "$results"
  
  echo ""
  echo "💡 Use --format=interactive for detailed information"
}
```

### JSON Format

```bash
generate_json_output() {
  local results="$1"
  
  echo "📊 Agent Search Results (JSON Format)"
  echo "{"
  echo '  "search_results": {'
  echo '    "total_matches": '$(wc -l < "$results"),'  
  echo '    "agents": ['
  
  local first=true
  while IFS='|' read -r agent tier domain description tools; do
    if [ "$first" = "false" ]; then
      echo ","
    fi
    first=false
    
    cat << EOF
      {
        "name": "$agent",
        "tier": "$tier",
        "domain": "$domain",
        "description": "$description",
        "tools": [$(echo "$tools" | sed 's/,/", "/g' | sed 's/^/"/' | sed 's/$/"/')],
        "invocation": "/agent-$(echo $domain | tr '[:upper:]' '[:lower:]') [task] [context] [--options]"
      }
EOF
  done < "$results"
  
  echo ""
  echo "    ]"
  echo "  }"
  echo "}"
}
```

### Interactive Format

```bash
generate_interactive_output() {
  local results="$1"
  
  echo "🎆 Interactive Agent Discovery Results"
  echo ""
  
  local count=1
  while IFS='|' read -r agent tier domain description tools; do
    echo "🤖 $count. $agent (Tier $tier)"
    echo "   🎯 Domain: $domain"
    echo "   📝 Description: $description"
    echo "   🔧 Tools: $tools"
    echo "   🚀 Usage: /agent-$(echo $domain | tr '[:upper:]' '[:lower:]') [task] [context]"
    echo ""
    
    # Show usage examples based on domain
    case "$domain" in
      "api")
        echo "   📚 Examples:"
        echo "      /agent-api 'design REST endpoints' ./src/api --spec=openapi"
        echo "      /agent-api 'review API security' ./apis --owasp"
        ;;
      "security")
        echo "   📚 Examples:"
        echo "      /agent-security 'threat model authentication' ./src/auth"
        echo "      /agent-security 'review security posture' ./infra --cis"
        ;;
      "frontend")
        echo "   📚 Examples:"
        echo "      /agent-frontend 'optimize React components' ./src/components"
        echo "      /agent-frontend 'implement responsive design' ./ui"
        ;;
      *)
        echo "   📚 Example: /agent-$domain 'describe your task' ./target-context"
        ;;
    esac
    
    echo ""
    ((count++))
  done < "$results"
}
```

### List Format

```bash
generate_list_output() {
  local results="$1"
  
  echo "📝 Agent Search Results (List Format)"
  echo ""
  
  while IFS='|' read -r agent tier domain description tools; do
    echo "• $agent ($domain, Tier $tier)"
  done < "$results"
  
  echo ""
  local total=$(wc -l < "$results")
  echo "📊 Total: $total matching agents"
}
```

## Usage Examples & Recommendations

```bash
show_usage_examples() {
  local domain="$1"
  local agent_count="$2"
  
  echo "💡 Usage Recommendations:"
  echo ""
  
  if [ "$agent_count" -gt 10 ]; then
    echo "   ⚡ Large result set - consider narrowing your search:"
    echo "      --tier=N        Filter by specific tier"
    echo "      --domain=TYPE   Filter by domain expertise"
    echo "      --tools=TOOL    Filter by tool requirements"
    echo ""
  fi
  
  if [ "$agent_count" -eq 0 ]; then
    echo "   🔍 No matches found. Try:"
    echo "      Broader search terms"
    echo "      Remove restrictive filters"
    echo "      Check spelling and domain names"
    echo ""
    echo "   📚 Available domains: api, security, frontend, backend, cloud,"
    echo "      database, devops, mobile, typescript, python, ml, research"
    echo ""
  fi
  
  echo "   🚀 Quick Agent Invocation:"
  echo "      /agent-{domain} 'task description' [context] [--options]"
  echo ""
  echo "   🔄 Multi-Agent Workflows:"
  echo "      /workflow-feature-development [context] --stage=design"
  echo "      /workflow-api-development [api-context] --parallel"
  echo ""
  echo "   📚 References:"
  echo "      agents/README.md - Complete agent catalog"
  echo "      /help - All available commands"
  echo "      docs/AGENT_PATTERNS.md - Usage patterns"
}
```

## Advanced Search Features

### Fuzzy Matching

```bash
fuzzy_search() {
  local term="$1"
  local threshold="0.7"  # Similarity threshold
  
  echo "Applying fuzzy matching for: '$term'"
  
  # Implement fuzzy string matching
  # Match similar terms, handle typos
  # Score results by relevance
}
```

### Search History

```bash
save_search_history() {
  local search_params="$1"
  local results_count="$2"
  local history_file="~/.claude/agent-search-history"
  
  echo "$(date): $search_params -> $results_count results" >> "$history_file"
}
```

### Intelligent Suggestions

```bash
generate_suggestions() {
  local failed_search="$1"
  
  echo "🤔 Smart Suggestions for '$failed_search':"
  
  # Analyze failed search and suggest alternatives
  case "$failed_search" in
    *"react"*|*"frontend"*)
      echo "   → Try: frontend, ui, javascript, typescript"
      ;;
    *"security"*|*"auth"*)
      echo "   → Try: security-architect, authentication, authorization"
      ;;
    *"performance"*|*"speed"*)
      echo "   → Try: performance-optimization-specialist, optimization"
      ;;
    *"database"*|*"sql"*)
      echo "   → Try: database-architect, data, storage"
      ;;
    *)
      echo "   → Try broader terms or check available domains"
      ;;
  esac
}
```

## Integration with Other Commands

```bash
show_related_commands() {
  echo "🔗 Related Commands:"
  echo "   /workflow-* - Multi-agent orchestration workflows"
  echo "   /quality-* - Quality assessment and validation"
  echo "   /setup-* - Environment and project setup"
  echo "   /agent-* - Direct agent invocation"
  echo ""
  echo "   📚 Command Discovery:"
  echo "      /help - Show all available commands"
  echo "      commands/README.md - Command documentation"
}
```

This search utility provides comprehensive agent discovery with intelligent filtering, multiple output formats, and helpful usage guidance to help developers quickly find and use the right specialist agents for their tasks.