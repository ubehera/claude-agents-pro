---
description: Surface the mandatory checklist before updating agents
args:
tools: Read
model: sonnet
---

## Purpose
Review the canonical checklist prior to modifying or adding agents so no quality gates are skipped.

## Required Checklist
@agents/AGENT_CHECKLIST.md

## Reminders
- Confirm verification (`/verify-agents`) passes after your changes.
- Install affected agents with `/install-agents --dry-run` when testing impact.
- Capture quality deltas with `/score-agents <path>` for significant updates.
