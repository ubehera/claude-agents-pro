---
name: branch-protection
description: Prevents direct commits and force-pushes to main/master branches
event: PreToolUse
tools: ["Bash"]
---

# Branch Protection Hook

Prevents accidental direct commits and destructive operations on protected branches.

## Protected Branches
- `main`
- `master`
- `production`
- `release/*`

## Blocked Operations

1. **Direct commits to protected branches**: `git commit` when on main/master
2. **Force push**: `git push --force` or `git push -f` to protected branches
3. **Hard reset**: `git reset --hard` on protected branches
4. **Branch deletion**: `git branch -D main` or similar

## Actions

### On Blocked Operation
1. **BLOCK** the command
2. **Explain** why the operation is blocked
3. **Suggest** the correct workflow:
   - Create feature branch: `git checkout -b feature/my-change`
   - Make changes and commit on feature branch
   - Create PR: `gh pr create`

## Trigger Condition

Activate when Bash tool is called with commands matching:
- `git commit` while current branch is protected
- `git push --force` or `git push -f` targeting protected branches
- `git reset --hard` while on protected branches
- `git branch -D` targeting protected branches
