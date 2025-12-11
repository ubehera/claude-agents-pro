# add-examples.py

Batch-add contextual example sections to agent markdown files.

## Installation

Requires PyYAML:
```bash
pip3 install --user --break-system-packages PyYAML
```

## Usage

### Preview Changes (Dry Run)
```bash
python3 scripts/add-examples.py --dry-run
```

### Preview with Verbose Output
```bash
python3 scripts/add-examples.py --dry-run --verbose
```

### Apply Changes
```bash
python3 scripts/add-examples.py
```

### Apply with Detailed Output
```bash
python3 scripts/add-examples.py --verbose
```

## What It Does

1. **Scans** all `.md` files in `agents/` subdirectories (00-meta through 08-finance)
2. **Parses** YAML frontmatter from each file
3. **Detects** agent domain based on name and description
4. **Generates** 2-3 contextual examples using domain-specific templates
5. **Updates** frontmatter with examples field
6. **Writes** changes back to files (preserves all existing fields)

## Domain Detection

The script intelligently detects agent domains and generates appropriate examples:

| Domain | Example Triggers |
|--------|------------------|
| **orchestration** | "Coordinate multi-agent workflow for building a payment processing system" |
| **algorithmic-trading** | "Implement order execution system with TWAP and VWAP algorithms" |
| **portfolio** | "Design portfolio optimization system with risk constraints" |
| **quant** | "Develop statistical arbitrage strategy using cointegration" |
| **market-data** | "Set up real-time market data pipeline with tick-level granularity" |
| **code-review** | "Review pull request for security vulnerabilities and code quality" |
| **domain-modeling** | "Design domain model for e-commerce order fulfillment system" |
| **system-design** | "Design scalable architecture for social media platform" |
| **api** | "Design a REST API for user authentication with OAuth 2.0" |
| **database** | "Design database schema for multi-tenant SaaS application" |
| **security** | "Conduct security audit of authentication and authorization flows" |
| **testing** | "Create comprehensive test suite for payment processing module" |
| **frontend** | "Build responsive dashboard with real-time data visualization" |
| **backend** | "Design event-driven architecture for order processing system" |
| **performance** | "Investigate memory leak causing OOM crashes in production" |
| **devops** | "Set up CI/CD pipeline with automated testing and deployment" |
| **observability** | "Design monitoring and alerting strategy for microservices" |
| **ml** | "Design ML pipeline for fraud detection model" |

## Example Output Format

Generated examples are added to frontmatter as:

```yaml
examples:
  - trigger: "Design a REST API for user authentication with OAuth 2.0"
    commentary: "Triggers API design expertise with authentication focus, expecting OpenAPI spec and security best practices"
  - trigger: "Review our GraphQL schema for performance bottlenecks"
    commentary: "Engages GraphQL expertise for optimization analysis, expecting N+1 query detection and resolver improvements"
```

## Skipped Files

The script automatically skips:
- Files that already have an `examples:` field
- Files without valid YAML frontmatter
- Non-agent files (README.md, AGENT_CHECKLIST.md, TESTING.md, finance-glossary.md)

## Output Summary

After running, you'll see a summary:
```
======================================================================
SUMMARY
======================================================================
Total files processed:        42
  Skipped (has examples):     4
  Skipped (no frontmatter):   0
  Skipped (non-agent):        5
  Updated:                    33
  Errors:                     0
======================================================================
```
