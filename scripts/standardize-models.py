#!/usr/bin/env python3
"""
Standardize Agent Models to Opus 4.6

Updates all agent frontmatter to use claude-opus-4-6 model.
Maximum capability for all agents - no compromises.

Usage:
    python3 scripts/standardize-models.py                # Preview changes
    python3 scripts/standardize-models.py --apply        # Apply changes
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


TARGET_MODEL = "claude-opus-4-6"
TARGET_RATIONALE = "Maximum capability for optimal results"


def extract_frontmatter_and_body(content: str) -> tuple[dict | None, str, str]:
    """Extract YAML frontmatter and body from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1))
            raw_frontmatter = match.group(1)
            body = match.group(2)
            return frontmatter, raw_frontmatter, body
        except yaml.YAMLError as e:
            print(f"  YAML parse error: {e}")
            return None, "", content
    return None, "", content


def update_frontmatter(raw_frontmatter: str, frontmatter: dict) -> str:
    """Update model and model_rationale in frontmatter."""
    lines = raw_frontmatter.split('\n')
    new_lines = []
    model_found = False
    rationale_found = False

    for line in lines:
        # Update model line
        if line.startswith('model:'):
            new_lines.append(f'model: {TARGET_MODEL}')
            model_found = True
        # Update model_rationale line
        elif line.startswith('model_rationale:'):
            new_lines.append(f'model_rationale: {TARGET_RATIONALE}')
            rationale_found = True
        else:
            new_lines.append(line)

    # Add model if not found (after description)
    if not model_found:
        for i, line in enumerate(new_lines):
            if line.startswith('complexity:'):
                new_lines.insert(i + 1, f'model: {TARGET_MODEL}')
                new_lines.insert(i + 2, f'model_rationale: {TARGET_RATIONALE}')
                model_found = True
                rationale_found = True
                break

    return '\n'.join(new_lines)


def process_agent(agent_path: Path, apply: bool = False) -> tuple[bool, str]:
    """Process a single agent file."""
    content = agent_path.read_text(encoding="utf-8")
    frontmatter, raw_frontmatter, body = extract_frontmatter_and_body(content)

    if frontmatter is None:
        return False, "No valid frontmatter found"

    current_model = frontmatter.get("model", "not set")

    if current_model == TARGET_MODEL:
        return True, f"Already using {TARGET_MODEL}"

    # Update frontmatter
    updated_frontmatter = update_frontmatter(raw_frontmatter, frontmatter)
    new_content = f"---\n{updated_frontmatter}\n---\n{body}"

    if apply:
        agent_path.write_text(new_content, encoding="utf-8")
        return True, f"Updated from {current_model} to {TARGET_MODEL}"
    else:
        return True, f"Would update from {current_model} to {TARGET_MODEL}"


def find_agent_files(agents_dir: Path) -> list[Path]:
    """Find all agent markdown files."""
    excluded = {"README.md", "TESTING.md", "AGENT_CHECKLIST.md", "finance-glossary.md"}
    agents = []

    for tier_dir in sorted(agents_dir.iterdir()):
        if tier_dir.is_dir() and tier_dir.name[0].isdigit():
            for md_file in tier_dir.glob("*.md"):
                if md_file.name not in excluded:
                    agents.append(md_file)

    return agents


def main():
    parser = argparse.ArgumentParser(description=f"Standardize all agents to {TARGET_MODEL}")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: preview only)")
    parser.add_argument("--agents-dir", type=Path, default=Path("agents"), help="Agents directory")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    agents_dir = project_root / args.agents_dir

    if not agents_dir.exists():
        print(f"Error: Agents directory not found at {agents_dir}")
        sys.exit(1)

    agent_files = find_agent_files(agents_dir)

    if not agent_files:
        print("No agent files found")
        sys.exit(1)

    mode = "APPLYING" if args.apply else "PREVIEW"
    print(f"\n{mode}: Standardizing {len(agent_files)} agents to {TARGET_MODEL}\n")
    print("=" * 70)

    updated = 0
    already_correct = 0
    errors = 0

    for agent_path in sorted(agent_files):
        relative_path = agent_path.relative_to(project_root)
        success, message = process_agent(agent_path, args.apply)

        if success:
            if "Already" in message:
                already_correct += 1
                print(f"✓ {relative_path}: {message}")
            else:
                updated += 1
                print(f"→ {relative_path}: {message}")
        else:
            errors += 1
            print(f"✗ {relative_path}: {message}")

    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Already correct: {already_correct}")
    print(f"  {'Updated' if args.apply else 'Would update'}: {updated}")
    print(f"  Errors: {errors}")

    if not args.apply and updated > 0:
        print(f"\nRun with --apply to make changes")


if __name__ == "__main__":
    main()
