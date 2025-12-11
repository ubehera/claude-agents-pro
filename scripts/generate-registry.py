#!/usr/bin/env python3
"""
Agent Registry Generator

Generates comprehensive agent registry files for the Claude Agents Pro marketplace.

Outputs:
  - agent-metadata.json: Detailed metadata with quality scores and relationships
  - marketplace.json: Public-facing marketplace catalog
  - registry.json: Simplified agent listing

Usage:
    python3 scripts/generate-registry.py
    python3 scripts/generate-registry.py --agents-dir agents --output-metadata configs/agent-metadata.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


# ============================================================================
# Tier and Category Mappings
# ============================================================================

TIER_CATEGORIES = {
    "00-meta": "Meta-Orchestration",
    "01-foundation": "Foundation",
    "02-development": "Development",
    "03-specialists": "Specialists",
    "04-experts": "Experts",
    "05-domains": "Domains",
    "06-integration": "Integration",
    "07-quality": "Quality",
    "08-automation": "Automation",
    "08-finance": "Finance",
    "09-enterprise": "Enterprise",
}

TIER_DESCRIPTIONS = {
    "00-meta": "Multi-agent workflow orchestration and coordination",
    "01-foundation": "Core engineering agents for essential development tasks",
    "02-development": "Language and platform specialists",
    "03-specialists": "Domain experts for infrastructure and architecture",
    "04-experts": "Advanced specialists for ML, AI, and specialized domains",
    "05-domains": "Domain-specific experts for specialized business contexts",
    "06-integration": "Research, documentation, and knowledge integration",
    "07-quality": "Quality assurance, testing, and validation specialists",
    "08-automation": "Automation and infrastructure tooling experts",
    "08-finance": "Financial analysis and quantitative modeling specialists",
    "09-enterprise": "Enterprise architecture and governance specialists",
}

TIER_COMPLEXITY = {
    "00-meta": "expert",
    "01-foundation": "complex",
    "02-development": "complex",
    "03-specialists": "complex",
    "04-experts": "expert",
    "05-domains": "moderate",
    "06-integration": "moderate",
    "07-quality": "complex",
    "08-automation": "expert",
    "08-finance": "expert",
    "09-enterprise": "expert",
}

# Keywords for capability extraction
CAPABILITY_KEYWORDS = [
    'REST', 'GraphQL', 'gRPC', 'API', 'OpenAPI', 'OAuth', 'JWT',
    'React', 'Vue', 'Angular', 'TypeScript', 'JavaScript', 'Python', 'Go', 'Java',
    'AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'Terraform',
    'PostgreSQL', 'MongoDB', 'Redis', 'DynamoDB',
    'CI/CD', 'DevOps', 'MLOps', 'SRE', 'Security',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
    'Testing', 'Performance', 'Observability', 'Monitoring',
    'Frontend', 'Backend', 'Full Stack', 'Mobile',
    'Microservices', 'Event-Driven', 'CQRS', 'DDD',
]


# ============================================================================
# Utility Functions
# ============================================================================

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


def extract_tier(file_path: Path, agents_dir: Path) -> str:
    """Extract tier from file path (e.g., '01-foundation')."""
    try:
        relative_path = file_path.relative_to(agents_dir)
        tier_part = relative_path.parts[0] if relative_path.parts else None
        if tier_part and re.match(r"^\d{2}-", tier_part):
            return tier_part
    except ValueError:
        pass
    return "uncategorized"


def normalize_model(model: str | None) -> str:
    """Normalize model name to standard format."""
    if not model:
        return "sonnet"

    model_lower = model.lower()
    if "opus" in model_lower:
        return "opus"
    if "sonnet" in model_lower:
        return "sonnet"
    if "haiku" in model_lower:
        return "haiku"

    return "sonnet"


def extract_capabilities(description: str) -> list[str]:
    """Extract capabilities from description."""
    capabilities = []
    for keyword in CAPABILITY_KEYWORDS:
        if re.search(rf'\b{re.escape(keyword)}\b', description, re.IGNORECASE):
            capabilities.append(keyword)

    return list(set(capabilities))[:10]  # Limit to 10 unique


def extract_tags(description: str, name: str) -> list[str]:
    """Extract tags from description and name."""
    tags = set()

    # Add name-based tags
    name_parts = name.split('-')
    for part in name_parts:
        if len(part) > 3:
            tags.add(part.lower())

    # Extract from description
    tag_keywords = [
        'api', 'rest', 'graphql', 'design', 'architecture', 'security',
        'testing', 'performance', 'optimization', 'monitoring', 'observability',
        'cloud', 'aws', 'kubernetes', 'docker', 'devops', 'ci/cd',
        'frontend', 'backend', 'mobile', 'react', 'typescript', 'python',
        'database', 'sql', 'nosql', 'machine learning', 'ml', 'ai',
        'documentation', 'research', 'review', 'debugging',
    ]

    for keyword in tag_keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', description, re.IGNORECASE):
            tags.add(keyword.lower())

    return sorted(list(tags))[:15]  # Limit to 15 tags


def determine_complexity(description: str, tools: list[str]) -> str:
    """Determine complexity based on description and tools."""
    desc_length = len(description)
    tool_count = len(tools)

    if desc_length > 300 or tool_count >= 6:
        return "complex"
    if desc_length > 150 or tool_count >= 4:
        return "moderate"
    return "simple"


# ============================================================================
# Agent Parsing
# ============================================================================

def parse_agent_file(file_path: Path, agents_dir: Path, project_root: Path) -> dict | None:
    """Parse a single agent markdown file and extract metadata."""
    try:
        content = file_path.read_text(encoding='utf-8')
        frontmatter = extract_frontmatter(content)

        if not frontmatter:
            print(f"⚠ Skipping {file_path.name}: no valid frontmatter")
            return None

        # Validate required fields
        if 'name' not in frontmatter or 'description' not in frontmatter:
            print(f"⚠ Skipping {file_path.name}: missing required fields")
            return None

        name = frontmatter['name']
        description = frontmatter['description']
        tier = extract_tier(file_path, agents_dir)

        # Parse tools
        tools = frontmatter.get('tools', [])
        if isinstance(tools, str):
            tools = [t.strip() for t in re.split(r'[,\s]+', tools) if t.strip()]
        elif not isinstance(tools, list):
            tools = []

        # Extract metadata
        relative_path = file_path.relative_to(project_root)
        model = normalize_model(frontmatter.get('model'))
        complexity = determine_complexity(description, tools)
        capabilities = extract_capabilities(description)
        tags = extract_tags(description, name)

        return {
            "name": name,
            "description": description.strip(),
            "tier": tier,
            "category": TIER_CATEGORIES.get(tier, "Uncategorized"),
            "complexity": complexity,
            "model": model,
            "model_rationale": frontmatter.get('model_rationale', ''),
            "tools": tools,
            "capabilities": capabilities,
            "tags": tags,
            "file": str(relative_path),
            "subagent_type": frontmatter.get('subagent_type', name),
            "enhanced_capabilities": frontmatter.get('enhanced_capabilities', []),
        }

    except Exception as e:
        print(f"✗ Error parsing {file_path.name}: {e}")
        return None


def find_agent_files(agents_dir: Path) -> list[Path]:
    """Find all agent markdown files (excluding README, TESTING, etc.)."""
    excluded = {"README.md", "TESTING.md", "AGENT_CHECKLIST.md"}
    agents = []

    for tier_dir in sorted(agents_dir.iterdir()):
        if tier_dir.is_dir() and re.match(r'^\d{2}-', tier_dir.name):
            for md_file in tier_dir.glob("*.md"):
                # Exclude documentation files and glossaries
                if md_file.name not in excluded and not md_file.name.endswith("-glossary.md"):
                    agents.append(md_file)

    return sorted(agents)


# ============================================================================
# Registry Generation
# ============================================================================

def generate_agent_metadata(agents: list[dict]) -> dict:
    """Generate agent-metadata.json structure."""
    metadata = {
        "schema_version": "1.0",
        "metadata": {}
    }

    # Group by tier
    by_tier = defaultdict(list)
    for agent in agents:
        by_tier[agent['tier']].append(agent)

    # Build metadata structure
    for tier, tier_agents in sorted(by_tier.items()):
        tier_num = int(tier.split('-')[0])
        metadata["metadata"][tier] = {
            "tier": tier_num,
            "category": TIER_CATEGORIES.get(tier, "Uncategorized"),
            "agents": {}
        }

        for agent in tier_agents:
            metadata["metadata"][tier]["agents"][agent['name']] = {
                "quality_score": 8.0,  # Default, can be enhanced with actual scoring
                "dependencies": [],
                "delegates_to": [],
                "tools": agent['tools'],
                "activation_patterns": agent['tags'][:5],
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "maintainer": "ubehera"
            }

    return metadata


def generate_marketplace(agents: list[dict]) -> dict:
    """Generate marketplace.json structure."""
    # Group by tier
    by_tier = defaultdict(list)
    for agent in agents:
        by_tier[agent['tier']].append(agent)

    # Build tier information
    tiers = []
    for tier in sorted(by_tier.keys()):
        tier_num = int(tier.split('-')[0])
        tiers.append({
            "id": tier,
            "name": TIER_CATEGORIES.get(tier, "Uncategorized"),
            "description": TIER_DESCRIPTIONS.get(tier, ""),
            "level": tier_num,
            "complexity": TIER_COMPLEXITY.get(tier, "moderate")
        })

    # Build agent entries
    agent_entries = []
    for agent in agents:
        agent_entries.append({
            "id": agent['name'],
            "name": agent['name'],
            "tier": agent['tier'],
            "category": agent['category'],
            "description": agent['description'],
            "model": agent['model'],
            "model_rationale": agent['model_rationale'],
            "complexity": agent['complexity'],
            "capabilities": agent['capabilities'],
            "tags": agent['tags'],
            "tools": agent['tools'],
            "path": agent['file']
        })

    return {
        "version": "1.0.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "registry": {
            "name": "Claude Agents Pro Marketplace",
            "description": "Production-ready Claude Code agents with tiered orchestration system",
            "repository": "https://github.com/ubehera/claude-agents-pro",
            "license": "Apache-2.0"
        },
        "tiers": tiers,
        "agents": agent_entries,
        "statistics": {
            "total_agents": len(agents),
            "by_tier": {tier: len(tier_agents) for tier, tier_agents in by_tier.items()},
            "by_model": dict(defaultdict(int, {agent['model']: sum(1 for a in agents if a['model'] == agent['model']) for agent in agents})),
            "by_complexity": dict(defaultdict(int, {agent['complexity']: sum(1 for a in agents if a['complexity'] == agent['complexity']) for agent in agents}))
        }
    }


def generate_registry(agents: list[dict]) -> dict:
    """Generate simplified registry.json structure."""
    # Group by tier
    by_tier = defaultdict(list)
    for agent in agents:
        by_tier[agent['tier']].append(agent)

    agent_entries = []
    for agent in agents:
        agent_entries.append({
            "name": agent['name'],
            "description": agent['description'],
            "tier": agent['tier'],
            "path": agent['file'],
            "tools": agent['tools'],
            "enhanced_capabilities": agent['enhanced_capabilities'],
            "subagent_type": agent['subagent_type']
        })

    return {
        "version": "1.0.0",
        "generated": datetime.now().isoformat(),
        "total_agents": len(agents),
        "agents": agent_entries,
        "tiers": sorted(by_tier.keys())
    }


def calculate_statistics(agents: list[dict]) -> dict:
    """Calculate registry statistics."""
    by_tier = defaultdict(int)
    by_category = defaultdict(int)
    by_complexity = defaultdict(int)
    by_model = defaultdict(int)

    for agent in agents:
        by_tier[agent['tier']] += 1
        by_category[agent['category']] += 1
        by_complexity[agent['complexity']] += 1
        by_model[agent['model']] += 1

    return {
        "total": len(agents),
        "by_tier": dict(by_tier),
        "by_category": dict(by_category),
        "by_complexity": dict(by_complexity),
        "by_model": dict(by_model)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate agent registry files")
    parser.add_argument("--agents-dir", type=Path, default=Path("agents"), help="Agents directory")
    parser.add_argument("--output-metadata", type=Path, default=Path("configs/agent-metadata.json"), help="Agent metadata output")
    parser.add_argument("--output-marketplace", type=Path, default=Path("configs/marketplace.json"), help="Marketplace output")
    parser.add_argument("--output-registry", type=Path, default=Path("registry.json"), help="Registry output")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    agents_dir = project_root / args.agents_dir
    if not agents_dir.exists():
        print(f"Error: Agents directory not found at {agents_dir}")
        sys.exit(1)

    # Find and parse agents
    if not args.quiet:
        print("🚀 Generating agent registry...\n")

    agent_files = find_agent_files(agents_dir)
    if not agent_files:
        print("Error: No agent files found")
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(agent_files)} agent files to process...")

    agents = []
    for file_path in agent_files:
        agent = parse_agent_file(file_path, agents_dir, project_root)
        if agent:
            agents.append(agent)

    if not agents:
        print("Error: No valid agents parsed")
        sys.exit(1)

    # Sort by tier and name
    agents.sort(key=lambda a: (a['tier'], a['name']))

    # Generate outputs
    metadata = generate_agent_metadata(agents)
    marketplace = generate_marketplace(agents)
    registry = generate_registry(agents)
    stats = calculate_statistics(agents)

    # Write outputs
    output_files = [
        (args.output_metadata, metadata),
        (args.output_marketplace, marketplace),
        (args.output_registry, registry),
    ]

    for output_path, data in output_files:
        full_path = project_root / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')

        if not args.quiet:
            print(f"✓ Generated: {full_path.relative_to(project_root)}")

    # Print statistics
    if not args.quiet:
        print("\n📊 Statistics:")
        print(f"  • Total agents: {stats['total']}")
        print("  • By tier:")
        for tier, count in sorted(stats['by_tier'].items()):
            print(f"    - {tier}: {count}")
        print("  • By category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"    - {category}: {count}")
        print("  • By complexity:")
        for complexity, count in sorted(stats['by_complexity'].items(), key=lambda x: -x[1]):
            print(f"    - {complexity}: {count}")
        print("  • By model:")
        for model, count in sorted(stats['by_model'].items(), key=lambda x: -x[1]):
            print(f"    - {model}: {count}")

    print("\n✅ Registry generation complete!")


if __name__ == "__main__":
    main()
