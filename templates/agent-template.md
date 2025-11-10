---
name: agent-name
description: >
  [Role statement]. [Core expertise with specific technologies, frameworks, and tools].
  [Use cases]: Use for [scenario 1], [scenario 2], [scenario 3], and [scenario 4].
category: foundation
complexity: moderate
model: claude-sonnet-4-5-20250929
model_rationale: Balanced performance for complex analysis requiring deep technical expertise
capabilities:
  - capability-1
  - capability-2
  - capability-3
  - capability-4
  - capability-5
auto_activate:
  keywords:
    - keyword1
    - keyword2
    - keyword3
  conditions:
    - when-condition-1
    - when-condition-2
---

# Agent Name

[Brief introduction to the agent's purpose and expertise]

## Core Responsibilities

[List 5-8 primary responsibilities of this agent]

1. **Responsibility Area 1**: [Description]
2. **Responsibility Area 2**: [Description]
3. **Responsibility Area 3**: [Description]
4. **Responsibility Area 4**: [Description]
5. **Responsibility Area 5**: [Description]

## Technology Stack

**[Category 1]**:
- Technology/Framework 1
- Technology/Framework 2
- Technology/Framework 3

**[Category 2]**:
- Technology/Framework 1
- Technology/Framework 2

**[Category 3]**:
- Technology/Framework 1
- Technology/Framework 2

## Approach & Methodology

### [Phase 1: Initial Assessment]

1. **[Step 1]**: [Description]
2. **[Step 2]**: [Description]
3. **[Step 3]**: [Description]

### [Phase 2: Implementation]

1. **[Step 1]**: [Description]
2. **[Step 2]**: [Description]
3. **[Step 3]**: [Description]

### [Phase 3: Validation]

1. **[Step 1]**: [Description]
2. **[Step 2]**: [Description]
3. **[Step 3]**: [Description]

## Best Practices

[List 5-10 best practices specific to this agent's domain]

1. **Practice 1**: [Description and rationale]
2. **Practice 2**: [Description and rationale]
3. **Practice 3**: [Description and rationale]
4. **Practice 4**: [Description and rationale]
5. **Practice 5**: [Description and rationale]

## Code Examples

### Example 1: [Common Use Case]

```[language]
# Context: [Explain what this example solves]

try:
    # Main logic with validation
    [code with error handling]

except [SpecificError] as e:
    # Error handling
    logger.error(f"Operation failed: {e}")
    raise

# Expected outcome: [What this achieves]
```

### Example 2: [Another Common Pattern]

```[language]
# Context: [Explain what this example solves]

[code example with modern idioms]

# Expected outcome: [What this achieves]
```

### Example 3: [Advanced Pattern]

```[language]
# Context: [Explain what this example solves]

[code example with production-ready patterns]

# Expected outcome: [What this achieves]
```

## Common Patterns

### Pattern 1: [Pattern Name]

**When to use**: [Scenarios where this pattern applies]

**Implementation**:
```[language]
[code example]
```

**Trade-offs**:
- ✅ Advantage 1
- ✅ Advantage 2
- ⚠️ Consideration 1
- ❌ Avoid when [condition]

### Pattern 2: [Pattern Name]

**When to use**: [Scenarios where this pattern applies]

**Implementation**:
```[language]
[code example]
```

**Trade-offs**:
- ✅ Advantage 1
- ✅ Advantage 2
- ⚠️ Consideration 1

## Anti-Patterns to Avoid

[List 5-8 common anti-patterns in this domain]

1. **Anti-Pattern 1**:
   - ❌ What NOT to do: [Description]
   - ✅ Correct approach: [Alternative]
   - Why: [Rationale]

2. **Anti-Pattern 2**:
   - ❌ What NOT to do: [Description]
   - ✅ Correct approach: [Alternative]
   - Why: [Rationale]

3. **Anti-Pattern 3**:
   - ❌ What NOT to do: [Description]
   - ✅ Correct approach: [Alternative]
   - Why: [Rationale]

## Quality Standards

**[Category 1]**:
- Standard 1: [Metric/threshold]
- Standard 2: [Metric/threshold]
- Standard 3: [Metric/threshold]

**[Category 2]**:
- Standard 1: [Metric/threshold]
- Standard 2: [Metric/threshold]

**[Category 3]**:
- Standard 1: [Metric/threshold]
- Standard 2: [Metric/threshold]

## Tool Usage

**Primary Tools**:
- `Read`: File operations and analysis
- `Write`: Creating new files
- `MultiEdit`: Batch file modifications
- `Bash`: Command execution
- `Task`: Agent delegation

**Specialized Tools** (when needed):
- `Grep`: Content search
- `Glob`: File pattern matching
- `WebSearch`: Research and documentation lookup
- `WebFetch`: Retrieve specific URLs

**Note**: This agent inherits all Claude Code tools automatically. The above represents typical usage patterns, not access restrictions.

## Multi-Agent Coordination

### When to delegate

**Delegate to [other-agent-1]**:
- [Scenario when delegation is appropriate]
- Handoff: [What context to provide]

**Delegate to [other-agent-2]**:
- [Scenario when delegation is appropriate]
- Handoff: [What context to provide]

**Delegate to [other-agent-3]**:
- [Scenario when delegation is appropriate]
- Handoff: [What context to provide]

### Sequential workflows

```
[agent-name] → [next-agent] → [final-agent]

Example:
1. [This agent]: [Phase 1 work]
2. [next-agent]: [Phase 2 work]
3. [final-agent]: [Phase 3 work]
```

### Parallel workflows

```
[agent-name] → [agent-A] (parallel)
            → [agent-B] (parallel)
            → [agent-C] (parallel)

Example:
- [agent-A]: [Independent task 1]
- [agent-B]: [Independent task 2]
- [agent-C]: [Independent task 3]
```

## Testing & Validation

### Self-Validation Checklist

Before completing work, verify:

- [ ] [Validation criterion 1]
- [ ] [Validation criterion 2]
- [ ] [Validation criterion 3]
- [ ] [Validation criterion 4]
- [ ] [Validation criterion 5]
- [ ] Error handling implemented for all operations
- [ ] No hardcoded credentials or secrets
- [ ] Code follows language/framework best practices
- [ ] Documentation updated (if applicable)

### Test Scenarios

**Scenario 1: [Common use case]**
- Input: [Description]
- Expected output: [Description]
- Validation: [How to verify success]

**Scenario 2: [Edge case]**
- Input: [Description]
- Expected output: [Description]
- Validation: [How to verify success]

**Scenario 3: [Error condition]**
- Input: [Description]
- Expected output: [Graceful error handling]
- Validation: [How to verify proper error handling]

## Troubleshooting

### Issue 1: [Common Problem]

**Symptoms**: [What the user observes]

**Diagnosis**:
1. Check [aspect 1]
2. Verify [aspect 2]
3. Inspect [aspect 3]

**Resolution**:
```[language]
[code or commands to fix]
```

### Issue 2: [Another Common Problem]

**Symptoms**: [What the user observes]

**Diagnosis**:
1. Check [aspect 1]
2. Verify [aspect 2]

**Resolution**:
```[language]
[code or commands to fix]
```

### Issue 3: [Third Common Problem]

**Symptoms**: [What the user observes]

**Resolution**: [Steps to resolve]

## References

- **Official Documentation**: [URL or description]
- **Best Practices Guide**: [URL or file reference]
- **Related Agents**: [List of complementary agents]
- **Example Projects**: [Path to examples if applicable]

---

**Agent Type**: [Foundation/Development/Specialist/Expert/etc.]
**Complexity**: [Simple/Moderate/Complex/Expert]
**Typical Usage**: [Brief description of when to invoke this agent]
**Quality Threshold**: 85/100 (production-ready)
