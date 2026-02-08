"""Shared utility for locating agent files across tier directories."""

from pathlib import Path
from typing import Optional


def find_agent_file(agents_dir: Path, agent_name: str) -> Optional[Path]:
    """Find an agent file by name across all tier directories.

    Tries exact filename match first, then falls back to fuzzy
    (substring) matching against agent file stems.

    Args:
        agents_dir: Root agents directory containing tier subdirectories.
        agent_name: Agent name (without .md extension).

    Returns:
        Path to the matching agent file, or None if not found.
    """
    # Exact match
    for tier_dir in agents_dir.glob("[0-9][0-9]-*"):
        if tier_dir.is_dir():
            agent_file = tier_dir / f"{agent_name}.md"
            if agent_file.exists():
                return agent_file

    # Fuzzy match (substring)
    for tier_dir in agents_dir.glob("[0-9][0-9]-*"):
        if tier_dir.is_dir():
            for agent_file in tier_dir.glob("*.md"):
                if agent_file.name == "README.md":
                    continue
                if agent_name.lower() in agent_file.stem.lower():
                    return agent_file

    return None
