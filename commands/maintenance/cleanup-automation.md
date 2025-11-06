---
description: Automated cleanup and maintenance for optimal repository health
args: [--scope git|files|cache|logs] [--aggressive] [--schedule daily|weekly|monthly]
tools: Bash(git gc:*), Bash(git clean:*), Bash(find:*), Read, Write, Grep
model: claude-sonnet-4-5
---

## Objective
Automated repository maintenance with intelligent cleanup, performance optimization, and health monitoring.

## Before You Run
- Create backup: `git stash push -u -m "pre-cleanup backup"`
- Review current disk usage: `du -sh .`
- Check for important uncommitted changes
- Verify no critical processes are using target files

## Execution
Run maintenance operations:

```bash
# Full cleanup operation
!/cleanup-automation --scope all

# Aggressive cleanup with advanced options
!/cleanup-automation --scope all --aggressive

# Schedule regular cleanup
!/cleanup-automation --scope git,cache --schedule weekly
```

## Cleanup Scope

### Git Repository Cleanup
```bash
# Remove merged branches (local)
git branch --merged main | grep -v "main\|develop" | xargs -n 1 git branch -d

# Remove remote-tracking branches for deleted remotes
git remote prune origin

# Clean up unreachable objects
git gc --aggressive --prune=now

# Optimize repository size
git repack -ad

# Remove stale reflog entries
git reflog expire --expire=30.days --all
```

### File System Cleanup
```bash
# Remove temporary files
find . -name "*.tmp" -type f -mtime +7 -delete
find . -name "*.bak" -type f -mtime +30 -delete
find . -name ".DS_Store" -type f -delete

# Clean up build artifacts
rm -rf node_modules/.cache/
rm -rf dist/ build/ .next/
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Remove empty directories
find . -type d -empty -not -path "./.git/*" -delete 2>/dev/null
```

### Cache and Log Cleanup
```bash
# Clear application caches
rm -rf ~/.cache/claude-agents-pro/
rm -rf /tmp/claude-agents-pro-*

# Rotate and compress logs
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;
find logs/ -name "*.log.gz" -mtime +90 -delete

# Clear system-specific caches
if [[ "$OSTYPE" == "darwin"* ]]; then
    rm -rf ~/Library/Caches/claude-agents-pro/
fi
```

### Dependency Cleanup
```bash
# Clean npm dependencies
npm prune
npm dedupe
npm cache clean --force

# Clean Python dependencies
pip cache purge 2>/dev/null || true
find . -name "*.pyc" -delete

# Clean Ruby dependencies
bundle clean 2>/dev/null || true
gem cleanup 2>/dev/null || true
```

## Automated Maintenance Tasks

### Daily Maintenance
- **Log Rotation**: Compress old log files
- **Temp File Cleanup**: Remove temporary files
- **Cache Validation**: Verify cache integrity
- **Health Checks**: Basic system health monitoring

### Weekly Maintenance
- **Git Cleanup**: Remove merged branches and optimize repository
- **Dependency Updates**: Check for and apply minor updates
- **Performance Analysis**: Monitor system performance
- **Security Scan**: Basic security vulnerability check

### Monthly Maintenance
- **Deep Cleanup**: Aggressive cleanup operations
- **Dependency Audit**: Full security audit of dependencies
- **Performance Optimization**: System-wide performance tuning
- **Backup Verification**: Validate backup integrity
- **Documentation Update**: Refresh outdated documentation

## Smart Cleanup Rules

### Retention Policies
```bash
# Age-based cleanup
FILE_RETENTION_DAYS=30
LOG_RETENTION_DAYS=90
BACKUP_RETENTION_DAYS=180

# Size-based cleanup (when disk usage > 80%)
DISK_THRESHOLD=80
if [ $(df . | tail -1 | awk '{print $5}' | sed 's/%//') -gt $DISK_THRESHOLD ]; then
    echo "Disk usage high, performing aggressive cleanup"
    AGGRESSIVE_MODE=true
fi
```

### Safe Deletion Checks
```bash
# Verify files are not in use
lsof +D . 2>/dev/null | grep -E "\.(tmp|log|cache)$" && {
    echo "Files in use, skipping deletion"
    exit 1
}

# Preserve important files
find . -name "*.important" -o -name "README*" -o -name "LICENSE*" | while read file; do
    touch "$file"  # Update access time to prevent deletion
done
```

### Recovery Safeguards
```bash
# Create recovery information
echo "Cleanup started at $(date)" > .cleanup_log
echo "Files to be deleted:" >> .cleanup_log
find . -name "*.tmp" -mtime +7 >> .cleanup_log

# Create file manifest before deletion
find . -type f > .pre_cleanup_manifest
```

## Performance Optimization

### Repository Optimization
```bash
# Git repository performance tuning
git config core.preloadindex true
git config core.fscache true
git config gc.auto 256

# Optimize pack files
git repack -A -d --depth=50 --window=50

# Configure bitmap index for faster clones
git config pack.writeBitmaps true
```

### File System Optimization
```bash
# Optimize file access patterns
find . -type f -name "*.md" -exec touch {} \; # Update access times

# Defragment files on supported systems
if command -v e4defrag >/dev/null 2>&1; then
    e4defrag .
fi

# Set optimal file attributes
find . -type f -name "*.sh" -exec chmod 755 {} \;
find . -type f -name "*.md" -exec chmod 644 {} \;
```

## Monitoring and Reporting

### Cleanup Metrics
```bash
# Measure cleanup impact
BEFORE_SIZE=$(du -sb . | cut -f1)
# ... perform cleanup ...
AFTER_SIZE=$(du -sb . | cut -f1)
SAVED_SPACE=$((BEFORE_SIZE - AFTER_SIZE))

echo "Cleanup saved $(( SAVED_SPACE / 1024 / 1024 )) MB"
```

### Health Monitoring
```bash
# Repository health score
HEALTH_SCORE=100

# Check for large files
if find . -size +100M | grep -q .; then
    HEALTH_SCORE=$((HEALTH_SCORE - 10))
    echo "Warning: Large files detected"
fi

# Check for deep directory structures
if find . -type d -depth -name "*" | head -1 | wc -c | xargs test 200 -lt; then
    HEALTH_SCORE=$((HEALTH_SCORE - 5))
    echo "Warning: Deep directory structure"
fi

echo "Repository health score: $HEALTH_SCORE/100"
```

### Automated Reporting
```json
{
  "cleanup_timestamp": "2024-01-15T10:30:00Z",
  "scope": "all",
  "metrics": {
    "space_freed_mb": 256,
    "files_removed": 1247,
    "git_objects_pruned": 89,
    "execution_time_seconds": 45
  },
  "health_score": 95,
  "recommendations": [
    "Schedule weekly git cleanup",
    "Consider dependency cleanup",
    "Monitor disk usage trends"
  ]
}
```

## Scheduled Automation

### Cron Job Setup
```bash
# Daily cleanup at 2 AM
0 2 * * * cd /path/to/claude-agents-pro && ./cleanup-automation --scope cache,logs >/dev/null 2>&1

# Weekly cleanup on Sundays at 3 AM
0 3 * * 0 cd /path/to/claude-agents-pro && ./cleanup-automation --scope all --schedule weekly

# Monthly deep cleanup on the 1st at 4 AM
0 4 1 * * cd /path/to/claude-agents-pro && ./cleanup-automation --scope all --aggressive
```

### CI/CD Integration
```yaml
# GitHub Actions scheduled cleanup
name: Repository Cleanup
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM UTC
  workflow_dispatch:

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Cleanup
        run: ./cleanup-automation --scope all
      - name: Commit Changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add .
          git diff --staged --quiet || git commit -m "chore: automated cleanup"
```

## Recovery and Rollback

### Pre-cleanup Backup
```bash
# Create comprehensive backup
tar -czf "cleanup-backup-$(date +%Y%m%d-%H%M%S).tar.gz" \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.tmp' \
    .
```

### Recovery Procedures
```bash
# Restore from backup
tar -xzf cleanup-backup-*.tar.gz

# Restore specific files
git checkout HEAD~1 -- path/to/file

# Undo git cleanup
git reflog | head -20  # Find pre-cleanup state
git reset --hard HEAD@{3}  # Reset to before cleanup
```

## Follow Up
- Review cleanup report and metrics
- Monitor system performance post-cleanup
- Update retention policies based on results
- Schedule next maintenance window
- Document any issues encountered
- Update cleanup procedures as needed

## Troubleshooting

### Common Issues
- **Permission Denied**: Run with appropriate permissions
- **Files in Use**: Stop processes using target files
- **Disk Space**: Ensure sufficient space for operations
- **Git Corruption**: Verify repository integrity before cleanup

### Diagnostic Commands
```bash
# Check repository integrity
git fsck --full

# Verify file system health
fsck -n /dev/disk  # Read-only check

# Monitor disk usage
watch -n 1 'df -h && echo && du -sh .'
```
