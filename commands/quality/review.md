---
description: Comprehensive code review with quality gates
args: [scope] [--level standard|strict|security]
tools: Task, Read, Grep, Glob
model: sonnet
---

# Code Review Orchestration

Execute comprehensive code review for: **$ARGUMENTS**

## Review Scope

### Scope Detection
Determine what to review:
- **No arguments**: Review recent changes (git diff)
- **File/directory path**: Review specific code
- **PR number**: Review pull request changes
- **Feature name**: Review related components

### Review Level
- **standard** (default): Code quality, best practices, bugs
- **strict**: Includes performance, accessibility, documentation
- **security**: Focus on security vulnerabilities and compliance

## Review Process

### Phase 1: Static Analysis
Delegate to code-reviewer via Task for:
- Code quality assessment
- Best practices validation
- Bug detection
- Complexity analysis
- Documentation coverage

### Phase 2: Domain Review
Based on code type, invoke specialists:
- **Frontend**: frontend-expert
- **Backend**: backend-architect
- **Database**: database-architect
- **Infrastructure**: aws-cloud-architect
- **Security**: security-architect

### Phase 3: Quality Gates

#### Standard Level (Default)
- ✅ No critical bugs
- ✅ Code follows conventions
- ✅ Tests included
- ✅ Documentation present
- ✅ No obvious performance issues

#### Strict Level
All standard checks plus:
- ✅ >80% test coverage
- ✅ Performance benchmarks pass
- ✅ Accessibility standards met
- ✅ API documentation complete
- ✅ Error handling comprehensive

#### Security Level
All strict checks plus:
- ✅ No security vulnerabilities
- ✅ OWASP Top 10 compliance
- ✅ No secrets/credentials
- ✅ Input validation complete
- ✅ Authentication/authorization correct

## Review Feedback

### Categorized Issues
- 🔴 **Critical**: Bugs, security issues, data loss risks
- 🟡 **Major**: Performance problems, missing tests, bad patterns
- 🔵 **Minor**: Style issues, typos, optimizations
- 💡 **Suggestions**: Improvements, refactoring opportunities

### Actionable Feedback
Each issue includes:
1. **Location**: File path and line number
2. **Issue**: Clear description of problem
3. **Impact**: Why it matters
4. **Solution**: How to fix it
5. **Example**: Code snippet if helpful

## Integration Points

### Pull Request Reviews
```
/review PR-123 --level strict
```
Automatically:
- Fetches PR changes
- Runs comprehensive review
- Posts comments on PR
- Updates PR status checks

### Pre-commit Reviews
```
/review staged --level standard
```
Reviews staged changes before commit

### Continuous Integration
```
/review --level security
```
Integrates with CI/CD pipelines

## Review Artifacts

Generate:
1. **Review summary** with pass/fail status
2. **Issue report** with categorized findings
3. **Metrics dashboard** with code quality trends
4. **Suggested fixes** for common issues
5. **Learning notes** for team improvement

## Examples

```
/review                          # Review recent changes
/review src/auth --level strict  # Strict review of auth module
/review PR-456 --level security  # Security review of PR
/review feature/payments         # Review feature branch
```

## Follow-up Actions

After review:
1. Fix critical issues immediately
2. Create tasks for major issues
3. Schedule minor improvements
4. Update coding standards if needed
5. Share learnings with team