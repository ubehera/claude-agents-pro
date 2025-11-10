# Skills Directory

Progressive disclosure knowledge modules that activate on-demand to reduce baseline token usage while providing deep expertise when needed.

## What are Skills?

Skills are **modular knowledge packages** that extend agent capabilities without loading all details into every conversation. Instead of baking 2000+ lines of technical content into agent prompts, skills load progressively:

1. **Tier 1 (Metadata)**: Always loaded (~10 tokens) - name, description, trigger keywords
2. **Tier 2 (Instructions)**: Loaded when activated (~200-400 tokens) - core concepts, patterns, best practices
3. **Tier 3 (Resources)**: Loaded on-demand (~1000+ tokens) - complete code examples, detailed implementations

## Benefits

**Token Efficiency**:
- Agent baseline: Lean prompt without detailed implementations
- Skills activate only when needed
- Average conversation: ~70% token reduction vs monolithic agents

**Maintainability**:
- Update skills independently without touching agent prompts
- Share skills across multiple agents
- Version skills separately

**Clarity**:
- Agents focus on orchestration and high-level strategy
- Skills provide deep technical knowledge
- Clear separation of concerns

## Skills Catalog

### Finance Domain

**Location**: `skills/finance/`

| Skill | Triggers | Purpose | Lines |
|-------|----------|---------|-------|
| `technical-indicators` | RSI, MACD, Bollinger, ATR, moving averages | Vectorized technical analysis indicators | ~400 |
| `options-greeks` | Delta, Gamma, Theta, Vega, Black-Scholes, IV | Options pricing and Greeks calculations | ~450 |
| `statistical-models` | Cointegration, GARCH, stationarity, time-series | Statistical analysis and feature engineering | ~500 |

**Total**: ~1350 lines of finance expertise available on-demand

## Usage Pattern

### Agent Configuration

Add skills to agent frontmatter:

```yaml
---
name: quantitative-analyst
skills:
  - technical-indicators
  - options-greeks
  - statistical-models
---
```

### Automatic Activation

Skills activate based on trigger keywords in user requests:

```
User: "Calculate RSI and MACD for this price series"
→ technical-indicators skill activates automatically
→ Loads implementations, code examples, validation patterns
```

```
User: "Calculate option Greeks for this call option"
→ options-greeks skill activates automatically
→ Loads Black-Scholes pricing, Greeks formulas, IV solver
```

```
User: "Test if these two stocks are cointegrated"
→ statistical-models skill activates automatically
→ Loads Engle-Granger test, z-score calculations, pairs trading patterns
```

### Skill Invocation (Claude Code Native Support)

Claude Code supports skills natively via the `Skill` tool. When an agent has skills configured, Claude automatically:
1. Checks trigger keywords in the user's request
2. Loads relevant skill metadata
3. Activates full skill content when patterns match
4. Returns to baseline after task completion

## Token Savings Example

### Before Skills (Monolithic Agent)

**quantitative-analyst.md**: 334 lines
- Frontmatter: 24 lines
- Philosophy & delegation: 40 lines
- Technical indicators: ~120 lines (RSI, MACD, BB, ATR code)
- Options Greeks: ~100 lines (Black-Scholes, all Greeks)
- Statistical analysis: ~100 lines (ADF, cointegration, GARCH)
- Quality standards: ~50 lines

**Token usage per invocation**: ~2000 tokens baseline

### After Skills (Modular Design)

**quantitative-analyst.md**: 240 lines (28% reduction)
- Frontmatter with skills reference: 27 lines
- Philosophy & delegation: 40 lines
- Skill activation guides: ~60 lines (pointers to skills)
- Quality standards: ~50 lines
- No detailed implementations (moved to skills)

**Token usage**:
- Baseline (no skills activated): ~600 tokens
- With technical-indicators skill: ~1000 tokens
- With all 3 skills: ~2100 tokens

**Average savings**: ~70% for typical invocations (most don't need all skills)

## Creating New Skills

### Skill Template

```markdown
---
name: skill-name
description: Load when user needs [specific capability description]
trigger_keywords: [keyword1, keyword2, keyword3]
---

# Skill Name

Brief overview of what this skill provides.

## Core Concepts

High-level concepts and when to use this skill.

## Implementation Patterns

Detailed patterns with code examples.

## Best Practices

Guidelines for using this skill effectively.

## Quality Standards

Quality metrics and validation criteria.

---

**Skill Type**: [Domain - Subdomain]
**Complexity**: [Simple/Moderate/Complex]
**Typical Usage**: [When this skill activates]
```

### Guidelines

1. **Single Responsibility**: One skill = one cohesive capability
2. **Self-Contained**: Skills should work independently
3. **Progressive Detail**: Start broad, get specific
4. **Production-Ready**: Include complete, tested code examples
5. **Clear Triggers**: Specific keywords that indicate when skill is needed

### When to Extract a Skill

Extract knowledge into a skill when:
- Agent prompt >300 lines
- Dense technical content (>100 lines of code/formulas)
- Content needed <50% of the time
- Knowledge is reusable across agents
- Updates happen frequently

**Don't extract if**:
- Content <100 lines total
- Needed in >80% of invocations
- Tightly coupled to agent logic
- Constantly referenced

## Skill Maintenance

### Updating Skills

1. **Edit skill file directly** in `skills/` directory
2. **No agent updates needed** - skills load dynamically
3. **Test activation** by invoking with trigger keywords
4. **Version skills** if making breaking changes

### Sharing Skills Across Agents

Multiple agents can reference the same skill:

```yaml
# quantitative-analyst frontmatter
skills:
  - technical-indicators
  - options-greeks

# trading-strategy-architect frontmatter
skills:
  - technical-indicators  # Shared skill
```

### Skill Dependencies

Skills can reference other skills or suggest delegation:

```markdown
## When to Use Other Skills

- For GARCH volatility modeling → Activate `statistical-models` skill
- For ML feature engineering → Activate `feature-engineering` skill
```

## Roadmap

### Phase 1 (Complete)
- ✅ Finance domain skills (technical-indicators, options-greeks, statistical-models)
- ✅ quantitative-analyst integration
- ✅ Token efficiency validation

### Phase 2 (Planned)
- ML/AI skills (pytorch-patterns, hyperparameter-tuning, mlops-deployment)
- Python skills (async-patterns, testing-strategies, packaging-tools)
- Backend skills (api-design-patterns, microservices-patterns, caching-strategies)

### Phase 3 (Future)
- Cross-domain skill sharing
- Skill versioning and compatibility
- Skill marketplace/registry

## Best Practices for Skill Usage

### For Users

1. **Request what you need naturally** - skills activate automatically
2. **Don't mention skills explicitly** - they're an implementation detail
3. **Expect complete answers** - skills provide production-ready code

### For Agent Developers

1. **Reference skills in agent frontmatter** clearly
2. **Remove detailed implementations** from agent prompts
3. **Add skill activation guides** in agent body
4. **Test skill activation** with various trigger patterns
5. **Monitor token usage** to validate savings

## Quality Standards

All skills must meet:
- **Completeness**: Self-contained with all necessary code/patterns
- **Accuracy**: Production-ready, tested implementations
- **Clarity**: Clear concepts, well-documented code
- **Performance**: Optimized patterns (vectorization, type hints)
- **Safety**: Error handling, input validation, no hardcoded secrets

---

**Skills System**: Progressive Disclosure Architecture
**Status**: Production (Phase 1 Complete)
**Token Savings**: ~70% average reduction vs monolithic agents
**Domains**: Finance (3 skills), ML/AI (planned), Python (planned)
