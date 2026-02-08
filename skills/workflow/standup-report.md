---
name: standup-report
description: Load when user needs to generate standup reports, daily summaries, progress updates, or work status reports
trigger_keywords: [standup, daily report, progress report, status update, what did I do, yesterday today blockers, scrum report, daily summary]
---

# Standup Report Generation

Automated standup report generation from git history, task management, and code changes to streamline daily sync meetings.

## Core Concepts

### Standup Components

**Standard Format** (Yesterday/Today/Blockers):
- **Yesterday**: Completed work, merged PRs, resolved issues
- **Today**: Planned tasks, in-progress work, meetings
- **Blockers**: Dependencies, unclear requirements, technical obstacles

### Data Sources

```yaml
Git History:
  - Commits (last 24-48 hours)
  - PRs merged/opened/reviewed
  - Branches worked on

Task Management:
  - Completed tickets
  - In-progress items
  - Blocked/waiting status

Code Changes:
  - Files modified
  - Features implemented
  - Bugs fixed
```

## Implementation Patterns

### 1. Git-Based Report Generator

```bash
#!/bin/bash
# Generate standup from git history

AUTHOR=$(git config user.name)
SINCE="yesterday 6am"
UNTIL="today 6am"

echo "## 📊 Standup Report - $(date +%Y-%m-%d)"
echo ""
echo "### Yesterday"
echo ""

# Commits
git log --author="$AUTHOR" --since="$SINCE" --until="$UNTIL" \
  --pretty=format:"- %s" --no-merges | head -10

echo ""

# PRs merged (if gh CLI available)
if command -v gh &> /dev/null; then
  echo ""
  echo "**PRs Merged:**"
  gh pr list --author="@me" --state=merged \
    --json title,mergedAt \
    --jq '.[] | select(.mergedAt > "'$SINCE'") | "- \(.title)"'
fi

echo ""
echo "### Today"
echo ""

# In-progress work
git branch --list | grep -E "(feature|fix|task)" | head -5 | \
  sed 's/^/* Continue work on/'

echo ""
echo "### Blockers"
echo ""
echo "- [ ] None currently"
```

### 2. Structured Report Template

```markdown
## Standup Report - [DATE]

### ✅ Yesterday (Completed)
- [TASK_1]: Brief description of what was accomplished
- [TASK_2]: Feature/fix delivered
- PR #[NUM]: [Title] - merged/reviewed

### 🔄 Today (Planned)
- [ ] [TASK_3]: What you plan to work on
- [ ] [TASK_4]: Secondary priority item
- [ ] Code review for PR #[NUM]

### 🚧 Blockers
- ⚠️ [BLOCKER_1]: Waiting on [PERSON/TEAM] for [THING]
- ⚠️ [BLOCKER_2]: Technical issue with [COMPONENT]

### 📝 Notes
- [Any additional context, meetings, OOO, etc.]
```

### 3. Automated Report Generation

```python
"""Standup report generator from git and GitHub."""
from datetime import datetime, timedelta
import subprocess
import json

def get_git_commits(author: str, since: datetime) -> list[str]:
    """Get commits from git log."""
    result = subprocess.run(
        [
            "git", "log",
            f"--author={author}",
            f"--since={since.isoformat()}",
            "--pretty=format:%s",
            "--no-merges"
        ],
        capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.split("\n") if line.strip()]

def get_prs_merged(author: str, since: datetime) -> list[dict]:
    """Get merged PRs from GitHub CLI."""
    result = subprocess.run(
        [
            "gh", "pr", "list",
            "--author", author,
            "--state", "merged",
            "--json", "title,number,mergedAt"
        ],
        capture_output=True, text=True
    )
    prs = json.loads(result.stdout) if result.stdout else []
    return [pr for pr in prs if pr.get("mergedAt", "") > since.isoformat()]

def generate_standup(author: str = "@me") -> str:
    """Generate complete standup report."""
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.replace(hour=6, minute=0, second=0)

    commits = get_git_commits(author, yesterday)
    prs = get_prs_merged(author, yesterday)

    report = []
    report.append(f"## Standup Report - {datetime.now().strftime('%Y-%m-%d')}")
    report.append("")
    report.append("### ✅ Yesterday")

    for commit in commits[:5]:
        report.append(f"- {commit}")

    for pr in prs[:3]:
        report.append(f"- PR #{pr['number']}: {pr['title']} (merged)")

    if not commits and not prs:
        report.append("- No tracked commits (meetings/planning day?)")

    report.append("")
    report.append("### 🔄 Today")
    report.append("- [ ] ")  # User fills in

    report.append("")
    report.append("### 🚧 Blockers")
    report.append("- None currently")

    return "\n".join(report)
```

## Best Practices

### Report Quality

1. **Be Specific**: "Implemented user authentication" not "Worked on auth"
2. **Link Context**: Include PR numbers, ticket IDs, file paths
3. **Quantify When Possible**: "Fixed 3 flaky tests" not "Fixed tests"
4. **Highlight Blockers Early**: Surface impediments before they become critical
5. **Keep It Brief**: 2-3 minutes to read, max 5 bullets per section

### Automation Guidelines

```yaml
Do:
  - Auto-generate from git/GitHub
  - Include PR/issue links
  - Categorize by feature area
  - Add time context (hours spent)

Don't:
  - Include every single commit
  - Copy-paste without editing
  - List meetings as "work"
  - Omit blocked items to look productive
```

### Team Sync Tips

- **Async-First**: Post written standups, discuss verbally only if needed
- **Time-Box**: 15 minutes max for whole team
- **Focus on Blockers**: Help unblock teammates first
- **Skip Details**: Deep-dives happen offline

## Quality Standards

- **Completeness**: All significant work captured
- **Clarity**: Anyone on team can understand
- **Brevity**: <5 bullets per section
- **Actionable Blockers**: Clear what's needed to unblock
- **Timeliness**: Posted before standup meeting

---

**Skill Type**: Workflow - Communication
**Complexity**: Simple
**Typical Usage**: Daily before standup meetings
**Tools**: Git, GitHub CLI, task management systems
