---
description: Automated Git workflow management with branch strategies and PR automation
args: [--workflow gitflow|github|gitlab] [--action init|feature|hotfix|release] [--auto-pr]
tools: Bash(git:*), Bash(gh pr:*), Bash(glab mr:*), Read, Grep
model: claude-sonnet-4-5
---

## Objective
Automate Git workflows with intelligent branch management, automated PR creation, and workflow-specific conventions.

## Before You Run
- Verify Git configuration: `git config --list`
- Ensure working directory is clean: `git status`
- Check remote repository access
- Review current branch strategy

## Execution
Automate Git workflows:

```bash
# Initialize GitFlow workflow
!/workflow-automation --workflow gitflow --action init

# Create feature branch with automated setup
!/workflow-automation --workflow github --action feature --auto-pr

# Hotfix workflow with automatic PR
!/workflow-automation --workflow gitflow --action hotfix --auto-pr
```

## Workflow Strategies

### GitFlow Workflow
- **Main Branches**: `main` (production), `develop` (integration)
- **Feature Branches**: `feature/feature-name`
- **Release Branches**: `release/version`
- **Hotfix Branches**: `hotfix/issue-description`
- **Support Branches**: Long-term maintenance branches

### GitHub Flow
- **Main Branch**: `main` (always deployable)
- **Feature Branches**: `feature/feature-name`
- **Direct Integration**: PR directly to main
- **Continuous Deployment**: Automatic deployment on merge

### GitLab Flow
- **Production Branch**: `production`
- **Main Branch**: `main` (pre-production)
- **Feature Branches**: `feature/feature-name`
- **Environment Branches**: `staging`, `production`

## Automated Actions

### Feature Development
```bash
# Feature branch creation
git checkout develop  # GitFlow
git pull origin develop
git checkout -b feature/new-agent-improvements

# Automated setup
echo "Feature: New Agent Improvements" > .git/BRANCH_DESCRIPTION
git push -u origin feature/new-agent-improvements

# Pre-commit hooks setup
git config core.hooksPath .githooks
```

### Pull Request Automation
```bash
# GitHub PR creation
gh pr create \
  --title "Feature: New Agent Improvements" \
  --body "$(cat .git/BRANCH_DESCRIPTION)" \
  --assignee "@me" \
  --label "enhancement" \
  --draft

# GitLab MR creation
glab mr create \
  --title "Feature: New Agent Improvements" \
  --description "$(cat .git/BRANCH_DESCRIPTION)" \
  --assignee "@me" \
  --label "enhancement"
```

### Release Management
```bash
# GitFlow release branch
git checkout develop
git checkout -b release/v2.1.0

# Version bump and changelog
echo "2.1.0" > VERSION
git add VERSION
git commit -m "chore: bump version to 2.1.0"

# Merge to main and tag
git checkout main
git merge --no-ff release/v2.1.0
git tag -a v2.1.0 -m "Release version 2.1.0"
```

## Branch Protection Rules

### Main Branch Protection
- **Require PR reviews**: At least 1 reviewer
- **Require status checks**: CI/CD pipeline passes
- **Require branches up to date**: Prevent outdated merges
- **Restrict push access**: Only through PRs
- **Require signed commits**: GPG signature validation

### Develop Branch Protection
- **Require PR reviews**: At least 1 reviewer for GitFlow
- **Allow force push**: For feature branch rebasing
- **Require status checks**: Automated testing

### Automation Rules
```bash
# Set up branch protection via GitHub CLI
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}'
```

## Commit Conventions

### Conventional Commits
- **feat**: New features
- **fix**: Bug fixes
- **docs**: Documentation changes
- **style**: Code formatting
- **refactor**: Code refactoring
- **test**: Test additions/modifications
- **chore**: Maintenance tasks

### Automated Commit Validation
```bash
# Pre-commit hook for commit message validation
#!/bin/bash
commit_regex='^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format"
    echo "Use: type(scope): description"
    exit 1
fi
```

## Automated Quality Gates

### Pre-merge Checks
- **Agent Validation**: `/verify-agents` passes
- **Quality Scoring**: Minimum quality threshold met
- **Security Scan**: No critical vulnerabilities
- **Performance Test**: No significant regressions
- **Documentation**: Changes documented

### Continuous Integration
```yaml
# GitHub Actions workflow
name: Quality Gates
on:
  pull_request:
    branches: [main, develop]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Agents
        run: ./scripts/verify-agents.sh
      - name: Quality Check
        run: python scripts/quality-scorer.py --threshold 80
      - name: Security Audit
        run: /security-audit --scope all --level basic
```

## Conflict Resolution

### Automated Merge Strategies
- **Fast-forward**: Clean linear history
- **Merge commit**: Preserve branch history
- **Squash merge**: Combine commits into one
- **Rebase merge**: Linear history with original commits

### Conflict Prevention
- **Regular sync**: Frequent upstream synchronization
- **Small changes**: Smaller, focused commits
- **Communication**: Team coordination on changes
- **Branch hygiene**: Regular cleanup of stale branches

### Conflict Resolution Tools
```bash
# Interactive rebase for conflict resolution
git rebase -i HEAD~3

# Merge tool configuration
git config merge.tool vimdiff
git mergetool

# Automated conflict resolution patterns
git config merge.ours.driver true  # Always use "ours"
```

## Workflow Metrics

### Development Velocity
- **Lead Time**: Time from commit to deployment
- **Cycle Time**: Time from start to finish
- **Throughput**: Features delivered per iteration
- **Failure Rate**: Percentage of failed deployments

### Quality Metrics
- **Defect Escape Rate**: Bugs found in production
- **Code Review Coverage**: Percentage of reviewed code
- **Test Coverage**: Automated test coverage percentage
- **Technical Debt**: Outstanding technical debt items

### Process Efficiency
```bash
# Git statistics for metrics
git log --since="1 month ago" --oneline | wc -l  # Commit frequency
git log --since="1 month ago" --merges | wc -l   # Merge frequency
git branch -r --merged | wc -l                    # Merged branches
```

## Multi-Repository Management

### Repository Synchronization
```bash
# Sync across multiple repositories
for repo in claude-agents-pro agent-templates agent-docs; do
    cd "$repo"
    git fetch --all
    git checkout main
    git pull origin main
    cd ..
done
```

### Cross-Repository Dependencies
- **Submodules**: Git submodule management
- **Subtrees**: Git subtree for shared code
- **Package Management**: Dependency version coordination
- **Release Coordination**: Synchronized releases

## Follow Up
- Review workflow execution logs
- Monitor branch health and cleanup
- Update workflow rules based on team feedback
- Analyze workflow metrics for optimization
- Train team on new workflow procedures
- Document workflow decisions and rationale

## Troubleshooting

### Common Issues
- **Merge Conflicts**: Use interactive rebase and merge tools
- **Branch Divergence**: Regular synchronization and communication
- **Failed CI/CD**: Review and fix quality gate failures
- **Permission Issues**: Verify repository access and branch protection

### Recovery Procedures
```bash
# Emergency rollback
git revert HEAD~1  # Revert last commit
git push origin main

# Branch recovery
git reflog  # Find lost commits
git checkout -b recovery-branch <commit-hash>

# Force sync (use with caution)
git reset --hard origin/main
```
