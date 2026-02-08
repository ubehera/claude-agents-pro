---
description: Score agents against the quality rubric
args: [target-path] [--report report.md]
tools: Bash(python3 scripts/quality-scorer.py:*)
model: sonnet
---

## Objective
Generate rubric scores for one or more agents using `python3 scripts/quality-scorer.py`.

## Steps
1. Choose the scope to analyze (single file, directory, or glob).
2. Run the scorer:

```bash
python3 scripts/quality-scorer.py $ARGUMENTS
```

3. Review the detailed breakdown to prioritize follow-up work.

## Tips
- Default target is `agents`, so `/score-agents` without arguments evaluates everything.
- Provide a relative path for narrower reviews, e.g. `/score-agents agents/03-specialists`.
- Add `--report <file>` to capture the results for sharing.

## Related Resources
- `scripts/quality-scorer.py`
