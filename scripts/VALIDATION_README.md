# Agent Validation Infrastructure

Comprehensive validation system for Claude Agents Pro agents, ensuring quality, consistency, and adherence to best practices.

## Overview

The validation infrastructure consists of three core components:

1. **agent-schema.json** - JSON Schema for frontmatter validation
2. **validate-agents.ts** - TypeScript validation script with custom rules
3. **quality-scorer.py** - Python-based quality assessment system

## Quick Start

```bash
# Install dependencies
npm install

# Validate all agents
npm run validate

# Validate specific agent
npm run validate:agent agents/01-foundation/api-platform-engineer.md

# Run quality scoring
npm run quality:score

# Run full test suite (validation + quality)
npm test
```

## Components

### 1. JSON Schema Validation (`agent-schema.json`)

Defines the structure and constraints for agent frontmatter:

**Required Fields:**
- `name` - Lowercase, hyphen-separated identifier matching filename (pattern: `^[a-z0-9-]+$`)
- `description` - Clear description (50-500 characters)

**Optional Fields:**
- `tools` - Comma-separated tool list (e.g., "Read, Write, Bash")
- `model` - Specific Claude model (haiku|sonnet|opus)
- `model_rationale` - Justification for model choice (required if model specified)
- `category` - Organizational category (orchestrator|foundation|specialist|etc.)
- `complexity` - Task complexity level (simple|moderate|complex)
- `capabilities` - Array of primary capabilities
- `auto_activate` - Auto-activation configuration (keywords, conditions)

**Validation Rules:**
- Name must match filename (without `.md`)
- If `model` is specified, `model_rationale` is required (min 20 chars)
- Description must be actionable and clear

### 2. TypeScript Validator (`validate-agents.ts`)

Performs schema validation plus custom business logic checks:

#### Validations

**Frontmatter Checks:**
- ✅ YAML structure and completeness
- ✅ Filename matches `name` field
- ✅ Description quality (length, action verbs)
- ✅ Tool usage optimization (≤7 tools recommended)
- ✅ Model specification consistency
- ✅ Duplicate name detection

**Content Checks:**
- ✅ "You are" opening statement presence
- ✅ Markdown section structure (## headings)
- ✅ Minimum content length (500 words)
- ✅ Capabilities documentation for complex agents

**Tool Optimization:**
- Warns if >7 tools specified (least-privilege principle)
- Detects redundant tool combinations (WebSearch + WebFetch)
- Validates against known Claude Code tools

#### Usage

```bash
# Validate all agents
npm run validate

# Validate specific agent
npm run validate:agent agents/path/to/agent.md

# Generate detailed JSON report
npm run validate:verbose

# Show help
npx tsx scripts/validate-agents.ts --help
```

#### Exit Codes

- `0` - All validations passed
- `1` - Validation errors found (warnings don't fail the build)

#### Output Format

**Console Report:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Agent Validation Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files Processed: 34
Errors: 2
Warnings: 5

❌ Errors:
  agents/path/to/agent.md
    • Filename 'agent.md' doesn't match name field 'different-name'

⚠️  Warnings:
  agents/path/to/agent.md
    • Description should include action verbs

✅ No errors found (5 warnings)

Detailed report: validation-report.json
```

**JSON Report (`validation-report.json`):**
```json
{
  "timestamp": "2025-11-05T10:30:00.000Z",
  "summary": {
    "total": 34,
    "errors": 2,
    "warnings": 5
  },
  "issues": [
    {
      "file": "/path/to/agent.md",
      "message": "Description too short (45 chars, minimum 50 recommended)",
      "severity": "warning"
    }
  ]
}
```

### 3. Quality Scorer (`quality-scorer.py`)

Python-based comprehensive quality assessment using weighted metrics.

#### Quality Metrics (Weighted)

| Metric | Weight | Evaluates |
|--------|--------|-----------|
| **Completeness** | 25% | Content coverage, required sections, word count, code examples |
| **Accuracy** | 25% | Technical correctness, terminology, realistic examples |
| **Usability** | 20% | Description clarity, structure, actionable instructions |
| **Performance** | 15% | Tool optimization, MCP integration, efficiency patterns |
| **Maintainability** | 15% | Naming conventions, collaboration patterns, documentation |

#### Quality Tiers

- **Tier 0 - Meta** (9.0+): Orchestration-level agents
- **Tier 1 - Foundation** (8.0+): Core engineering agents
- **Tier 2 - Specialist** (7.5+): Domain specialists
- **Tier 3 - Expert** (7.0+): Advanced expertise
- **Tier 4 - Professional** (6.5+): Production-ready
- **Tier 5 - Developing** (<6.5): In development

#### Usage

```bash
# Score all agents
npm run quality:score

# Score specific agent
npm run quality:agent agents/path/to/agent.md

# Generate full JSON report
npm run quality:full
```

#### Output

```
Quality Score: 8.45
Tier: Tier 1 - Foundation

Detailed Metrics:
  Completeness: 0.87
  Accuracy: 0.91
  Usability: 0.84
  Performance: 0.78
  Maintainability: 0.82

Recommendations:
  - Optimize tool selection and add MCP integration patterns
```

## CI/CD Integration

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

npm run validate
if [ $? -ne 0 ]; then
  echo "Agent validation failed. Fix errors before committing."
  exit 1
fi
```

### GitHub Actions

```yaml
name: Validate Agents

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: npm install

      - name: Validate agents
        run: npm run validate

      - name: Quality scoring
        run: npm run quality:score
```

## Best Practices

### Agent Frontmatter

```yaml
---
name: my-specialist-agent
description: Expert in domain-specific tasks including design, implementation, and optimization. Use for complex workflows requiring deep domain knowledge.
tools: Read, Write, Bash, Task
---
```

**Do:**
- ✅ Use lowercase-hyphenated names matching filename
- ✅ Include action verbs in description (design, implement, optimize)
- ✅ Specify minimal tool set (least-privilege principle)
- ✅ Document model choice with rationale if overriding default

**Don't:**
- ❌ Include >7 tools unless genuinely needed
- ❌ Use CamelCase or underscores in name
- ❌ Write vague descriptions ("Helps with things")
- ❌ Specify model without rationale

### Content Structure

```markdown
---
frontmatter
---

You are a [role] specializing in [domain]. [Brief expertise statement].

## Core Expertise

### Technical Domains
- Domain 1
- Domain 2

### Capabilities
- Capability 1
- Capability 2

## Approach & Philosophy

[Your methodology]

## Technical Implementation

[Implementation patterns]

## Quality Standards

[Quality expectations]

## Collaboration Patterns

[How you work with other agents]
```

## Common Issues & Solutions

### Issue: "Filename doesn't match name field"

**Problem:** `api-engineer.md` with `name: api-platform-engineer`

**Solution:** Rename file to match name field:
```bash
mv agents/api-engineer.md agents/api-platform-engineer.md
```

### Issue: "Description too short"

**Problem:** Description < 50 characters

**Solution:** Expand description with specific use cases:
```yaml
description: Expert in API design, gateway configuration, developer portals, and API governance. Use for REST/GraphQL schema design, OpenAPI specs, and API lifecycle management.
```

### Issue: "Tools list is long"

**Problem:** >7 tools specified

**Solution:** Review necessity of each tool. Consider inheriting tools (omit `tools` field) or reduce to essentials:
```yaml
# Before (11 tools)
tools: Read, Write, Edit, MultiEdit, Bash, WebSearch, WebFetch, Task, TodoWrite, Grep, Glob

# After (4 tools)
tools: Read, Write, Bash, Task
```

### Issue: "Model specified without model_rationale"

**Problem:**
```yaml
model: claude-opus-4-1-20250805
```

**Solution:** Add justification:
```yaml
model: claude-opus-4-1-20250805
model_rationale: Complex multi-agent orchestration requires highest reasoning capability for dependency management and workflow optimization
```

## Validation Report Formats

### Console Output

- Color-coded (errors: red, warnings: yellow, success: green)
- Grouped by file
- Summary statistics
- Detailed issue descriptions

### JSON Output (`validation-report.json`)

```json
{
  "timestamp": "2025-11-05T10:30:00.000Z",
  "summary": {
    "total": 34,
    "errors": 0,
    "warnings": 3
  },
  "issues": [
    {
      "file": "/absolute/path/to/agent.md",
      "message": "Description should include action verbs",
      "severity": "warning",
      "details": { }
    }
  ]
}
```

### Quality Report (`quality-report.json`)

```json
{
  "summary": {
    "total_agents": 34,
    "average_score": 7.85,
    "highest_score": 9.2,
    "lowest_score": 6.4,
    "tier_distribution": {
      "Tier 0 - Meta": 1,
      "Tier 1 - Foundation": 8,
      "Tier 2 - Specialist": 12,
      "Tier 3 - Expert": 10,
      "Tier 4 - Professional": 3,
      "Tier 5 - Developing": 0
    }
  },
  "agents": {
    "01-foundation/api-platform-engineer.md": {
      "metrics": {
        "completeness": 0.87,
        "accuracy": 0.91,
        "usability": 0.84,
        "performance": 0.78,
        "maintainability": 0.82
      },
      "overall_score": 8.45,
      "tier_classification": "Tier 1 - Foundation"
    }
  }
}
```

## Extending Validation

### Add Custom Validation Rule

Edit `scripts/validate-agents.ts`:

```typescript
function validateCustomRule(filePath: string, frontmatter: any, content: string): void {
  // Your custom logic
  if (someCondition) {
    issues.push({
      file: filePath,
      message: 'Custom validation message',
      severity: 'warning'
    });
  }
}

// Add to validateAgentFile function
function validateAgentFile(filePath: string): void {
  // ... existing validations
  validateCustomRule(filePath, frontmatter, content);
}
```

### Add Schema Field

Edit `scripts/agent-schema.json`:

```json
{
  "properties": {
    "new_field": {
      "type": "string",
      "description": "Your new field"
    }
  }
}
```

## Troubleshooting

### Validation script fails to run

**Check:**
1. Node.js version ≥18.0.0: `node --version`
2. Dependencies installed: `npm install`
3. TypeScript compiler: `npx tsc --version`

### Quality scorer fails

**Check:**
1. Python version ≥3.9: `python3 --version`
2. YAML library: `pip3 install pyyaml`
3. File paths are absolute in script

### False positives

**Report issues:**
1. Open GitHub issue with example agent
2. Include validation output
3. Suggest expected behavior

## Performance

- **Validation Speed**: ~34 agents in <2s
- **Quality Scoring**: ~34 agents in <5s
- **Memory Usage**: <100MB for full repository scan

## Dependencies

### Node.js (TypeScript)
- `ajv` (8.12.0) - JSON Schema validation
- `gray-matter` (4.0.3) - YAML frontmatter parsing
- `glob` (10.3.10) - File pattern matching
- `tsx` (4.7.0) - TypeScript execution
- `typescript` (5.3.3) - TypeScript compiler

### Python
- `pyyaml` - YAML parsing
- `pathlib` - Path handling
- `dataclasses` - Data structures

## License

Apache-2.0 (matches repository license)

## Contributing

1. Test changes: `npm run validate && npm run quality:score`
2. Update schema for new fields
3. Add validation rules with rationale
4. Document in this README
5. Submit PR with validation report

## Support

- Issues: GitHub Issues
- Docs: `agents/README.md`, `SYSTEM_OVERVIEW.md`
- Examples: `agents/01-foundation/` for reference implementations
