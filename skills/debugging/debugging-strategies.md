---
name: debugging-strategies
description: Master systematic debugging techniques, profiling tools, and root cause analysis to efficiently track down bugs across any codebase or technology stack. Use when investigating bugs, performance issues, or unexpected behavior.
---

# Debugging Strategies

Transform debugging from frustrating guesswork into systematic problem-solving with proven strategies, powerful tools, and methodical approaches.

## When to Use This Skill

- Tracking down elusive bugs
- Investigating performance issues
- Understanding unfamiliar codebases
- Debugging production issues
- Analyzing crash dumps and stack traces
- Profiling application performance
- Investigating memory leaks

## Core Principles

### The Scientific Method

1. **Observe**: What's the actual behavior?
2. **Hypothesize**: What could be causing it?
3. **Experiment**: Test your hypothesis
4. **Analyze**: Did it prove/disprove your theory?
5. **Repeat**: Until you find the root cause

### Debugging Mindset

**Don't Assume:**
- "It can't be X" - Yes it can
- "I didn't change Y" - Check anyway
- "It works on my machine" - Find out why

**Do:**
- Reproduce consistently
- Isolate the problem
- Keep detailed notes
- Question everything

## Systematic Debugging Process

### Phase 1: Reproduce

1. **Can you reproduce it?** Always? Sometimes? Randomly?
2. **Create minimal reproduction** - Simplify to smallest example
3. **Document steps** - Write down exact steps

### Phase 2: Gather Information

- Full stack trace
- Environment (OS, runtime version, dependencies)
- Recent changes (git history)
- Scope (all users or specific ones?)

### Phase 3: Form Hypothesis

Ask:
- What changed?
- What's different between working vs broken?
- Where could this fail?

### Phase 4: Test & Verify

**Binary Search:** Comment out half the code, narrow down
**Add Logging:** Strategic console.log/print
**Isolate Components:** Test each piece separately
**Compare Working vs Broken:** Diff configurations/environments

## Debugging Tools

### JavaScript/TypeScript
```typescript
debugger;  // Execution pauses here

console.log('Value:', value);
console.table(arrayOfObjects);
console.time('operation'); /* code */ console.timeEnd('operation');
console.trace();  // Stack trace
```

### Python
```python
import pdb
pdb.set_trace()  # Debugger starts here

# Python 3.7+
breakpoint()

# Profiling
import cProfile
cProfile.run('slow_function()', 'profile_stats')
```

## Advanced Techniques

### Git Bisect (Find Regression)
```bash
git bisect start
git bisect bad                    # Current commit is bad
git bisect good v1.0.0            # v1.0.0 was good
# Git checks out middle commit - test it
git bisect good   # if it works
git bisect bad    # if it's broken
git bisect reset  # when done
```

### Differential Debugging

| Aspect | Working | Broken |
|--------|---------|--------|
| Environment | Development | Production |
| Node version | 18.16.0 | 18.15.0 |
| Data | Empty DB | 1M records |

### Memory Leak Detection
```typescript
if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) {
    console.warn('High memory usage:', process.memoryUsage());
    require('v8').writeHeapSnapshot();
}
```

## Debugging Patterns by Issue Type

### Intermittent Bugs
1. Add extensive logging
2. Look for race conditions
3. Check timing dependencies
4. Stress test

### Performance Issues
1. Profile first - don't optimize blindly
2. Common culprits: N+1 queries, unnecessary re-renders, synchronous I/O
3. Tools: DevTools Performance, Lighthouse, cProfile

### Production Bugs
1. Gather evidence (error tracking, logs)
2. Reproduce locally with production data
3. Don't change production - test fixes in staging

## Quick Debugging Checklist

When stuck, check:
- [ ] Spelling errors (typos in variable names)
- [ ] Case sensitivity
- [ ] Null/undefined values
- [ ] Array index off-by-one
- [ ] Async timing (race conditions)
- [ ] Scope issues
- [ ] Type mismatches
- [ ] Missing dependencies
- [ ] Environment variables
- [ ] File paths (absolute vs relative)
- [ ] Cache issues
- [ ] Stale data

## Best Practices

1. **Reproduce First**: Can't fix what you can't reproduce
2. **Isolate the Problem**: Remove complexity until minimal case
3. **Read Error Messages**: They're usually helpful
4. **Check Recent Changes**: Most bugs are recent
5. **Use Version Control**: Git bisect, blame, history
6. **Take Breaks**: Fresh eyes see better
7. **Document Findings**: Help future you
8. **Fix Root Cause**: Not just symptoms
