#!/usr/bin/env python3
"""
Agent Frontmatter Schema Validator

Validates agent markdown files against the JSON Schema definition.
Supports CI integration with exit codes and JSON output.

Usage:
    python3 scripts/schema-validator.py                    # Validate all agents
    python3 scripts/schema-validator.py --agent agents/01-foundation/api-platform-engineer.md
    python3 scripts/schema-validator.py --ci               # CI mode with strict exit codes
    python3 scripts/schema-validator.py --output report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    print("Error: jsonschema required. Install with: pip install jsonschema")
    sys.exit(1)


def load_schema(schema_path: Path) -> dict:
    """Load the JSON Schema from file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            print(f"  YAML parse error: {e}")
            return None
    return None


def validate_agent(
    agent_path: Path, schema: dict, validator: Draft7Validator
) -> tuple[bool, list[str]]:
    """Validate a single agent file against the schema."""
    errors = []

    # Check file exists
    if not agent_path.exists():
        return False, [f"File not found: {agent_path}"]

    # Read content
    try:
        content = agent_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

    # Extract frontmatter
    frontmatter = extract_frontmatter(content)
    if frontmatter is None:
        return False, ["No valid YAML frontmatter found"]

    # Validate name matches filename
    expected_name = agent_path.stem
    actual_name = frontmatter.get("name", "")
    if actual_name != expected_name:
        errors.append(f"Name mismatch: frontmatter has '{actual_name}', filename is '{expected_name}'")

    # Validate against schema
    for error in validator.iter_errors(frontmatter):
        path = " -> ".join(str(p) for p in error.absolute_path) or "root"
        errors.append(f"[{path}] {error.message}")

    return len(errors) == 0, errors


def find_agent_files(agents_dir: Path) -> list[Path]:
    """Find all agent markdown files (excluding README, TESTING, etc.)."""
    excluded = {"README.md", "TESTING.md", "AGENT_CHECKLIST.md"}
    agents = []

    for tier_dir in sorted(agents_dir.iterdir()):
        if tier_dir.is_dir() and tier_dir.name.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            for md_file in tier_dir.glob("*.md"):
                if md_file.name not in excluded:
                    agents.append(md_file)

    return agents


def main():
    parser = argparse.ArgumentParser(description="Validate agent frontmatter against JSON Schema")
    parser.add_argument("--agent", type=Path, help="Validate a single agent file")
    parser.add_argument("--agents-dir", type=Path, default=Path("agents"), help="Agents directory")
    parser.add_argument("--schema", type=Path, default=Path("schemas/agent-frontmatter.schema.json"), help="Schema file")
    parser.add_argument("--ci", action="store_true", help="CI mode: strict exit codes")
    parser.add_argument("--output", type=Path, help="Output JSON report to file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    schema_path = project_root / args.schema
    if not schema_path.exists():
        print(f"Error: Schema not found at {schema_path}")
        sys.exit(1)

    # Load schema and create validator
    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    # Determine which agents to validate
    if args.agent:
        agent_files = [args.agent if args.agent.is_absolute() else project_root / args.agent]
    else:
        agents_dir = project_root / args.agents_dir
        if not agents_dir.exists():
            print(f"Error: Agents directory not found at {agents_dir}")
            sys.exit(1)
        agent_files = find_agent_files(agents_dir)

    if not agent_files:
        print("No agent files found to validate")
        sys.exit(1)

    # Validate agents
    results = {
        "total": len(agent_files),
        "passed": 0,
        "failed": 0,
        "agents": []
    }

    if not args.quiet:
        print(f"\nValidating {len(agent_files)} agents against schema...\n")
        print("=" * 60)

    for agent_path in sorted(agent_files):
        relative_path = agent_path.relative_to(project_root) if project_root in agent_path.parents else agent_path
        valid, errors = validate_agent(agent_path, schema, validator)

        agent_result = {
            "path": str(relative_path),
            "valid": valid,
            "errors": errors
        }
        results["agents"].append(agent_result)

        if valid:
            results["passed"] += 1
            if not args.quiet:
                print(f"✅ {relative_path}")
        else:
            results["failed"] += 1
            if not args.quiet:
                print(f"❌ {relative_path}")
                for error in errors:
                    print(f"   └─ {error}")

    if not args.quiet:
        print("=" * 60)
        print(f"\nResults: {results['passed']}/{results['total']} passed")
        if results["failed"] > 0:
            print(f"         {results['failed']} failed")

    # Output JSON report if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"\nReport written to: {args.output}")

    # Exit code
    if args.ci and results["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
