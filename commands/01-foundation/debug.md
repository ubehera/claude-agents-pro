---
description: Intelligent debugging assistance for errors and issues
args: <error-description-or-log>
tools: Task, Read, Grep
model: sonnet
---

# Intelligent Debugging Assistant

Debug and resolve: **$ARGUMENTS**

## Issue Analysis

### Step 1: Error Classification
Identify the type of issue:
- **Compilation/Syntax Error**: Code won't compile/parse
- **Runtime Error**: Crashes, exceptions, panics
- **Logic Error**: Incorrect behavior, wrong results
- **Performance Issue**: Slow execution, high resource usage
- **Integration Problem**: API failures, connection issues
- **Flaky/Intermittent**: Inconsistent failures

### Step 2: Context Gathering

Collect relevant information:
1. **Error Messages & Stack Traces**
   - Parse error output for key information
   - Identify error origin file and line number
   - Extract error codes if present

2. **Code Context**
   - Use Read to examine error location
   - Check recent changes (git diff)
   - Review related files and dependencies

3. **Environment Factors**
   - System resources (memory, CPU, disk)
   - Environment variables
   - Configuration files
   - Network connectivity

### Step 3: Root Cause Analysis

Delegate to error-diagnostician via Task for:
- Deep stack trace analysis
- Pattern matching against known issues
- Correlation with recent changes
- Identification of contributing factors

## Debugging Strategies

### For Compilation Errors
1. Check syntax at error location
2. Verify import/include statements
3. Validate type definitions
4. Review build configuration

### For Runtime Errors
1. Add strategic logging/debugging output
2. Use debugger with breakpoints
3. Validate input data and edge cases
4. Check for null/undefined references

### For Performance Issues
1. Profile code execution
2. Identify bottlenecks and hot paths
3. Analyze memory usage patterns
4. Review algorithm complexity

### For Integration Problems
1. Verify API endpoints and contracts
2. Check authentication/authorization
3. Validate request/response formats
4. Test network connectivity

## Systematic Debugging Process

1. **Reproduce the Issue**
   ```bash
   # Create minimal reproduction case
   # Document steps to trigger error
   # Capture full error output
   ```

2. **Isolate the Problem**
   - Binary search to narrow scope
   - Comment out code sections
   - Use test cases to isolate

3. **Form Hypotheses**
   - List potential causes
   - Rank by likelihood
   - Design tests to validate

4. **Test Solutions**
   - Implement fixes incrementally
   - Verify each change
   - Ensure no regressions

5. **Document Resolution**
   - Record root cause
   - Document fix
   - Add regression test

## Common Issue Patterns

### Memory Leaks
- Unclosed resources (files, connections)
- Circular references
- Growing collections
- Event listener accumulation

### Race Conditions
- Concurrent access to shared state
- Missing synchronization
- Incorrect lock ordering
- Timing-dependent logic

### Configuration Issues
- Environment-specific problems
- Missing/incorrect settings
- Permission problems
- Path resolution issues

## Advanced Debugging Tools

```bash
# Language-specific debuggers
gdb, lldb              # C/C++
pdb, ipdb             # Python
node --inspect        # Node.js
dlv                   # Go

# Profilers
perf, valgrind        # System level
py-spy               # Python
clinic.js            # Node.js

# Tracing
strace, dtrace       # System calls
tcpdump, wireshark   # Network
```

## Error Prevention

After resolution, implement:
1. **Defensive Coding**
   - Input validation
   - Error boundaries
   - Graceful degradation

2. **Testing**
   - Unit test for the fix
   - Integration test for the scenario
   - Regression test suite

3. **Monitoring**
   - Add metrics for the issue
   - Set up alerts
   - Track error rates

## Examples

```
/debug "TypeError: Cannot read property 'x' of undefined"
/debug "API returns 403 forbidden intermittently"
/debug "Application crashes with OOM after 2 hours"
/debug "Tests fail only in CI environment"
```

## Follow-up Actions

1. Add fix to knowledge base
2. Update error handling
3. Improve logging around issue
4. Share learnings with team
5. Consider preventive refactoring