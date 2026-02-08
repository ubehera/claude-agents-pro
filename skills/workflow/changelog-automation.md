---
name: changelog-automation
description: Use when generating structured changelogs from commit history and release metadata with consistent sections, versioning, and upgrade notes.
trigger_keywords: [changelog automation, release notes automation, conventional commits changelog, changelog generation, release summary]
---

# Changelog Automation

Use this skill to produce release notes that are consistent, diff-driven, and easy to consume.

## When to Use This Skill

- Preparing release notes for version tags
- Standardizing changelog format across repositories
- Generating upgrade notes from commit metadata
- Enforcing changelog quality in CI workflows

## Core Concepts

- **Conventional commit mapping** gives deterministic sections.
- **Version windows** should use explicit tag ranges.
- **Breaking changes** require dedicated callouts.
- **Human curation** is still needed for context and migration notes.

## Implementation Patterns

```bash
#!/usr/bin/env bash
set -euo pipefail

PREV_TAG="v1.8.0"
NEW_TAG="v1.9.0"

git log --pretty=format:'%s' "${PREV_TAG}..${NEW_TAG}" \
  | awk '
    /^feat:/ {print "### Added\n- " substr($0, 6)}
    /^fix:/ {print "### Fixed\n- " substr($0, 5)}
    /^docs:/ {print "### Docs\n- " substr($0, 6)}
  '
```

## Validation Checklist

- Changelog uses explicit version boundaries
- Breaking changes are highlighted with migration guidance
- Entries are grouped by category and de-duplicated
- Release date and artifact links are included
