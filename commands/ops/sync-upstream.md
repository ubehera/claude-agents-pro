---
description: Synchronize with upstream repositories and manage dependency updates
args: [--source upstream|origin] [--strategy merge|rebase] [--auto-resolve] [--dry-run]
tools: Bash(git fetch:*), Bash(git merge:*), Bash(git rebase:*), Bash(git status:*), Grep, Read
model: sonnet
---

## Objective
Automatically synchronize with upstream repositories, manage dependency updates, and resolve conflicts with intelligent merge strategies.

## Before You Run
- Verify current branch status: `git status`
- Ensure you have upstream remotes configured
- Create backup branch: `git checkout -b sync-backup-$(date +%Y%m%d)`
- Review pending local changes

## Execution
Synchronize with upstream sources:

```bash
# Basic upstream sync with merge strategy
!/sync-upstream --source upstream --strategy merge

# Rebase local changes on upstream
!/sync-upstream --source upstream --strategy rebase --auto-resolve

# Preview changes without applying
!/sync-upstream --source upstream --dry-run
```

## Sync Strategies

### Merge Strategy (Default)
- **Preserves History**: Maintains complete commit history
- **Conflict Resolution**: Interactive conflict resolution
- **Safe Operation**: Minimal risk of data loss
- **Merge Commits**: Creates explicit merge points

### Rebase Strategy
- **Linear History**: Clean, linear commit progression
- **Conflict Handling**: Per-commit conflict resolution
- **Advanced Option**: Requires Git expertise
- **History Rewriting**: Modifies commit history

### Auto-Resolution Features
- **Intelligent Merging**: Automatic resolution of trivial conflicts
- **Preference Rules**: Prioritize upstream or local changes
- **Backup Creation**: Automatic state preservation
- **Conflict Reporting**: Detailed conflict analysis

## Dependency Management

### Package Dependencies
```bash
# Check for dependency updates
npm outdated || pip list --outdated || gem outdated
```

### Git Submodules
```bash
# Update all submodules recursively
git submodule update --init --recursive --remote
```

### Configuration Sync
- **MCP Servers**: Update server configurations
- **Agent Dependencies**: Sync agent tool requirements
- **Script Updates**: Merge script improvements
- **Documentation**: Synchronize README and docs

## Conflict Resolution

### Automatic Resolution
- **Whitespace Conflicts**: Auto-resolve formatting differences
- **Documentation**: Prefer upstream documentation updates
- **Configuration**: Merge configuration changes intelligently
- **Version Files**: Update version numbers appropriately

### Manual Resolution Required
- **Code Logic**: Requires human review and decision
- **Breaking Changes**: API or interface modifications
- **Custom Modifications**: Local customizations vs upstream
- **Security Patches**: Critical security-related changes

### Conflict Analysis
```bash
# Analyze conflict patterns
git diff --name-only --diff-filter=U
git diff --check
```

## Quality Assurance

### Post-Sync Validation
- **Agent Verification**: Run full agent validation suite
- **Script Testing**: Verify all scripts execute correctly
- **Configuration Check**: Validate configuration integrity
- **Quality Scoring**: Ensure quality standards maintained

### Regression Testing
- **Functionality Tests**: Core feature validation
- **Performance Tests**: Benchmark comparison
- **Integration Tests**: Cross-component compatibility
- **Security Tests**: Vulnerability assessment

## Monitoring & Reporting

### Sync Report Generation
- **Change Summary**: Files modified, added, deleted
- **Conflict Report**: Conflicts encountered and resolved
- **Dependency Updates**: Package and library changes
- **Quality Impact**: Before/after quality metrics

### Notification Integration
- **Slack/Teams**: Sync status notifications
- **Email Reports**: Detailed sync summaries
- **Issue Tracking**: Auto-create issues for conflicts
- **Dashboard Updates**: Project health metrics

## Rollback & Recovery

### Automatic Rollback Triggers
- **Build Failures**: CI/CD pipeline failures
- **Quality Degradation**: Significant quality score drops
- **Test Failures**: Critical test suite failures
- **Security Issues**: New vulnerability introductions

### Manual Rollback Options
```bash
# Quick rollback to backup branch
git checkout sync-backup-$(date +%Y%m%d)
git branch -D main
git checkout -b main
```

### Recovery Procedures
- **State Restoration**: Return to last known good state
- **Selective Revert**: Undo specific problematic changes
- **Manual Merge**: Hand-craft resolution for complex conflicts
- **Fresh Sync**: Start sync process from clean state

## Follow Up
- Review sync report for any issues
- Run comprehensive validation: `/health-check`
- Update local development environments
- Notify team of significant changes
- Schedule next sync operation
- Document any manual interventions

## Best Practices
- **Regular Scheduling**: Weekly or bi-weekly sync cycles
- **Incremental Sync**: Smaller, more frequent updates
- **Testing Strategy**: Comprehensive post-sync validation
- **Communication**: Team awareness of sync operations
- **Documentation**: Record sync procedures and decisions
