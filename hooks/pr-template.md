---
name: pr-template
description: Enriches PR creation with consistent template, related issues, and test plan
event: PreToolUse
tools: ["Bash"]
---

# PR Template Hook

Ensures pull requests follow a consistent format with summary, related issues, test plan, and breaking changes when using `gh pr create`.

## Trigger Condition

Activate when Bash tool is called with a command containing `gh pr create`.

## Template Structure

When a PR is being created, ensure the body includes:

```markdown
## Summary
<!-- 1-3 bullet points describing what changed and why -->

## Changes
<!-- List of key changes, grouped by area -->

## Related Issues
<!-- Link to related issues: Fixes #123, Relates to #456 -->

## Test Plan
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
<!-- Add specific test scenarios -->

## Breaking Changes
<!-- List any breaking changes, or "None" -->

## Screenshots
<!-- If UI changes, include before/after screenshots -->
```

## Actions

### On PR Create
1. **Inspect** the current `--body` content (if provided)
2. **Enrich** with missing template sections
3. **Auto-detect** related issues from commit messages and branch name
4. **Suggest** test plan items based on files changed
5. **Flag** if breaking changes detected (API signature changes, schema migrations, config format changes)

### Auto-Detection Rules

**Related Issues**: Extract from:
- Commit messages containing `#123`, `fixes #`, `closes #`, `relates to #`
- Branch name patterns: `fix/issue-123`, `feature/JIRA-456`

**Breaking Changes Detection**:
- API route changes in controller/router files
- Database migration files added
- Package major version bumps
- Config file format changes
- Removed public exports

## Notes

- Don't block PR creation — enrich the template instead
- If user provides a complete body, respect it and only add missing sections
- Keep auto-generated content clearly marked so user can edit
