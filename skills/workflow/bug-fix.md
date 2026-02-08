---
name: bug-fix
description: Load when user needs to diagnose, fix, and verify bug fixes with proper testing and documentation
trigger_keywords: [bug, fix, debug, error, issue, broken, not working, regression, defect, crash, exception, failure]
---

# Bug Fix Workflow

Systematic bug diagnosis, fixing, and verification workflow to ensure reliable fixes without introducing regressions.

## Core Concepts

### Bug Fix Lifecycle

```yaml
1. Reproduce
   - Confirm the bug exists
   - Identify exact steps to trigger
   - Document expected vs actual behavior

2. Isolate
   - Narrow down to root cause
   - Identify affected code paths
   - Check for related issues

3. Fix
   - Make minimal targeted change
   - Avoid unrelated refactoring
   - Consider edge cases

4. Verify
   - Write failing test first (TDD)
   - Run full test suite
   - Manual verification

5. Document
   - Clear commit message
   - Update any affected docs
   - Close related issues
```

### Bug Categories

| Type | Priority | Approach |
|------|----------|----------|
| **Crash/Data Loss** | P0 | Hotfix immediately, minimal change |
| **Feature Broken** | P1 | Fix in next release, full testing |
| **Performance** | P2 | Profile first, fix bottleneck |
| **Edge Case** | P3 | Add test, fix when convenient |
| **UI/UX** | P4 | Schedule with design review |

## Implementation Patterns

### 1. Bug Investigation Template

```markdown
## Bug Investigation: [Brief Title]

### Reported Behavior
- **What happened**: [Exact error/behavior]
- **Expected behavior**: [What should happen]
- **Reproduction steps**:
  1. [Step 1]
  2. [Step 2]
  3. [Step 3]
- **Frequency**: [Always/Sometimes/Rare]
- **Environment**: [OS, browser, version]

### Investigation Notes
- **Root cause hypothesis**: [What you think is wrong]
- **Code location**: [file:line]
- **Related code**: [Other affected areas]
- **Dependencies involved**: [External libs, APIs]

### Fix Plan
- **Proposed change**: [Brief description]
- **Files to modify**: [List of files]
- **Test strategy**: [How to verify fix]
- **Risk assessment**: [Low/Medium/High]
```

### 2. Systematic Debugging Flow

```python
"""Systematic bug debugging workflow."""
from dataclasses import dataclass
from enum import Enum

class BugSeverity(Enum):
    CRITICAL = "P0"  # System down, data loss
    HIGH = "P1"      # Feature broken, workaround exists
    MEDIUM = "P2"    # Degraded experience
    LOW = "P3"       # Minor issue

@dataclass
class BugReport:
    title: str
    severity: BugSeverity
    reproduction_steps: list[str]
    expected: str
    actual: str
    environment: dict

    def to_markdown(self) -> str:
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.reproduction_steps))
        return f"""
## Bug: {self.title}

**Severity**: {self.severity.value}

### Reproduction Steps
{steps}

### Expected Behavior
{self.expected}

### Actual Behavior
{self.actual}

### Environment
{self.environment}
"""

# Debugging checklist
DEBUGGING_CHECKLIST = [
    "Can you reproduce the bug consistently?",
    "What's the minimal reproduction case?",
    "When did this start happening? (git bisect)",
    "What changed recently in this area?",
    "Are there related error logs/stack traces?",
    "Does it happen in all environments?",
    "Is there a workaround?",
]
```

### 3. Git Bisect for Regression Finding

```bash
#!/bin/bash
# Find the commit that introduced a bug

# Start bisect
git bisect start

# Mark current state as bad
git bisect bad HEAD

# Mark last known good state
git bisect good v1.2.0  # or specific commit

# Git will checkout middle commits
# Test each one and mark:
git bisect good  # if bug not present
git bisect bad   # if bug present

# Or automate with a test script:
git bisect run npm test -- --grep="specific test"

# When done, git shows the culprit commit
git bisect reset  # Return to original state
```

### 4. Test-First Bug Fix

```typescript
// Step 1: Write failing test that captures the bug
describe('UserService', () => {
  it('should handle empty email gracefully (bug #1234)', () => {
    // This test captures the exact bug scenario
    const result = userService.validateEmail('');

    // Expected: graceful handling with error message
    // Actual (bug): throws unhandled exception
    expect(result.isValid).toBe(false);
    expect(result.error).toBe('Email is required');
  });
});

// Step 2: Fix the code to make test pass
function validateEmail(email: string): ValidationResult {
  // Bug fix: Handle empty string explicitly
  if (!email || email.trim() === '') {
    return { isValid: false, error: 'Email is required' };
  }

  // ... rest of validation logic
}

// Step 3: Run full test suite to check for regressions
// npm test
```

### 5. Commit Message Format

```
fix: prevent crash when email is empty (#1234)

Root cause: validateEmail() assumed non-null input but received
empty string from form submission when user cleared the field.

Changes:
- Add explicit empty string check in validateEmail()
- Return structured error instead of throwing

Testing:
- Added unit test for empty email case
- Verified fix manually in staging
- All existing tests pass

Closes #1234
```

## Best Practices

### Before Fixing

```yaml
DO:
  - Reproduce the bug locally first
  - Write a failing test that captures the bug
  - Check for duplicate issues
  - Understand the full code path
  - Consider if this is a symptom of larger issue

DON'T:
  - Fix blindly without understanding root cause
  - Make unrelated changes in the same PR
  - Skip writing tests "because it's obvious"
  - Assume the bug report is accurate without verifying
```

### While Fixing

```yaml
DO:
  - Make the minimal change needed
  - Add defensive checks for similar edge cases
  - Update type definitions if relevant
  - Add inline comments explaining non-obvious fixes

DON'T:
  - Refactor surrounding code (separate PR)
  - Fix "while you're there" issues (separate PR)
  - Remove "unnecessary" code that might be load-bearing
  - Change behavior beyond the bug scope
```

### After Fixing

```yaml
DO:
  - Run full test suite
  - Test manually in staging
  - Update documentation if behavior changed
  - Close related issues with fix reference

DON'T:
  - Merge without code review
  - Skip QA for "simple" fixes
  - Forget to update changelog
  - Leave debug statements in code
```

## Debugging Tools

### JavaScript/TypeScript

```typescript
// Strategic logging
console.log('[DEBUG] Function entry:', { args, state });
console.trace('Call stack');
console.time('operation'); /* ... */ console.timeEnd('operation');

// Debugger statement (stops in DevTools)
debugger;

// Node.js debugging
// node --inspect-brk script.js
```

### Python

```python
# pdb debugger
import pdb; pdb.set_trace()

# Rich debugging
from rich import inspect
inspect(obj, methods=True)

# Logging with context
import logging
logging.debug(f"Processing {item=}, {state=}")
```

### General

```bash
# Git blame to find who changed code
git blame -L 50,60 path/to/file.ts

# Git log for specific function
git log -p -S "functionName" -- "*.ts"

# Find when test started failing
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
git bisect run npm test
```

## Quality Standards

- **Reproduction**: Bug can be reliably reproduced
- **Root Cause**: Actual cause identified (not just symptoms)
- **Minimal Fix**: Changes only what's necessary
- **Test Coverage**: Failing test added before fix
- **No Regressions**: All existing tests pass
- **Documentation**: Clear commit message and issue closure

---

**Skill Type**: Workflow - Development
**Complexity**: Moderate
**Typical Usage**: Activated when debugging or fixing reported issues
**Tools**: Git, debuggers, testing frameworks, issue trackers
