---
description: Batch update agents with version control and rollback capability
args: [--pattern glob] [--action update|validate|rollback] [--dry-run] [--parallel]
tools: Bash(git:*), Read, Write, MultiEdit, Grep
model: opus
---

## Objective
Perform batch operations across multiple agents with atomic transactions, version control, and automated rollback capabilities.

## Before You Run
- Create a backup branch: `git checkout -b batch-update-$(date +%Y%m%d)`
- Ensure working directory is clean: `git status`
- Review agents that match your pattern
- Test operations on a single agent first

## Execution
Batch update operations:

```bash
# Dry-run to preview changes
!/batch-update --pattern "01-foundation/*.md" --action update --dry-run

# Update all agents in parallel
!/batch-update --pattern "agents/**/*.md" --action update --parallel

# Validate specific tier
!/batch-update --pattern "03-specialists/*.md" --action validate

# Rollback if issues detected
!/batch-update --action rollback
```

## Batch Operations

### Update Actions
- **Frontmatter Standardization**: Ensure consistent YAML structure
- **Tool Permission Audit**: Remove excessive permissions
- **Description Normalization**: Standardize format and style
- **Version Bumping**: Update agent versions systematically
- **Quality Improvements**: Apply automated fixes

### Validation Actions
- **Syntax Checking**: YAML and Markdown validation
- **Reference Verification**: Check internal and external links
- **Dependency Analysis**: Verify tool and pattern usage
- **Compliance Checking**: Policy adherence validation
- **Performance Analysis**: Resource usage optimization

### Rollback Actions
- **Git Reset**: Revert to last known good state
- **Selective Rollback**: Undo specific changes only
- **Backup Restoration**: Restore from automatic backups
- **State Verification**: Validate post-rollback integrity

## Safety Features

### Atomic Operations
- All changes succeed or all fail
- Automatic rollback on any failure
- Transaction log for change tracking
- Checkpoint creation before major operations

### Parallel Execution
- Configurable concurrency limits
- Progress tracking across operations
- Error isolation between parallel tasks
- Resource contention management

### Quality Gates
- Pre-update validation
- Post-update verification
- Quality score thresholds
- Regression detection

## Monitoring & Reporting
- Real-time progress indicators
- Detailed operation logs
- Success/failure metrics
- Performance benchmarks
- Change impact analysis

## Follow Up
- Review batch operation report
- Run full validation suite: `/verify-agents`
- Execute quality scoring: `/score-agents`
- Commit successful changes with descriptive message
- Tag release if appropriate: `git tag v1.2.3`

## Error Recovery
1. **Partial Failure**: Automatically rollback failed operations
2. **Complete Failure**: Restore from backup branch
3. **Corruption Detection**: Run health check and repair
4. **State Inconsistency**: Manual intervention with guided recovery
