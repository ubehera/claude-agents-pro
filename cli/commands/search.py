"""Search agents command."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List, Tuple
from difflib import SequenceMatcher

from ..utils.agent_parser import AgentParser

console = Console()


@click.command()
@click.argument('query', type=str)
@click.option(
    '--scope',
    type=click.Choice(['user', 'project', 'repo']),
    default='repo',
    help='Search scope (user, project, or repo)'
)
@click.option(
    '--agents-dir',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Custom agents directory to search'
)
@click.option(
    '--limit',
    type=int,
    default=10,
    help='Maximum number of results to show'
)
@click.option(
    '--min-score',
    type=float,
    default=0.3,
    help='Minimum similarity score (0.0-1.0)'
)
def search(query: str, scope: str, agents_dir: Path, limit: int, min_score: float):
    """Search agents by capability using fuzzy matching."""

    # Determine agents directory
    if agents_dir is None:
        if scope == 'user':
            agents_dir = Path.home() / ".claude" / "agents"
        elif scope == 'project':
            agents_dir = Path.cwd() / ".claude" / "agents"
        else:  # repo
            cli_root = Path(__file__).parent.parent.parent
            agents_dir = cli_root / "agents"

    if not agents_dir.exists():
        console.print(f"[red]Error:[/red] Agents directory not found: {agents_dir}")
        raise click.Abort()

    # Parse all agents
    parser = AgentParser()
    agents = []

    for tier_dir in sorted(agents_dir.glob("[0-9][0-9]-*")):
        if not tier_dir.is_dir():
            continue

        for agent_file in tier_dir.glob("*.md"):
            if agent_file.name in ['README.md', 'AGENT_CHECKLIST.md', 'TESTING.md']:
                continue

            try:
                agent_data = parser.parse_agent_file(agent_file)
                agent_data['tier'] = tier_dir.name
                agent_data['file'] = agent_file
                agents.append(agent_data)
            except Exception as e:
                pass  # Skip problematic files

    # Calculate similarity scores
    scored_agents = []
    query_lower = query.lower()

    for agent in agents:
        score = _calculate_match_score(query_lower, agent)
        if score >= min_score:
            scored_agents.append((score, agent))

    # Sort by score descending
    scored_agents.sort(key=lambda x: x[0], reverse=True)

    # Limit results
    results = scored_agents[:limit]

    if not results:
        console.print(f"[yellow]No agents found matching '{query}'[/yellow]")
        console.print(f"Try a broader search term or lower --min-score (current: {min_score})")
        return

    # Display results
    console.print(f"\n[bold]Found {len(results)} matching agents for:[/bold] '{query}'\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Score", style="green", width=6)
    table.add_column("Agent", style="yellow")
    table.add_column("Tier", style="blue", width=15)
    table.add_column("Description", style="white")

    for idx, (score, agent) in enumerate(results, 1):
        table.add_row(
            f"{idx}.",
            f"{score:.2f}",
            agent.get('name', 'unknown'),
            agent.get('tier', 'unknown'),
            agent.get('description', '')[:70] + '...' if len(agent.get('description', '')) > 70 else agent.get('description', '')
        )

    console.print(table)

    # Show top result details
    if results:
        top_score, top_agent = results[0]
        console.print(f"\n[bold]Top match:[/bold] {top_agent.get('name')}")
        console.print(f"[dim]Use 'claude-agents info {top_agent.get('name')}' for details[/dim]")


def _calculate_match_score(query: str, agent: dict) -> float:
    """Calculate fuzzy match score for an agent."""
    scores = []

    # Name match (high weight)
    name = agent.get('name', '').lower()
    name_score = SequenceMatcher(None, query, name).ratio()
    scores.append(name_score * 1.5)

    # Description match (medium weight)
    description = agent.get('description', '').lower()
    desc_score = SequenceMatcher(None, query, description).ratio()
    scores.append(desc_score * 1.2)

    # Keyword match (high weight if exact)
    keywords = agent.get('auto_activate', {}).get('keywords', [])
    if keywords:
        keyword_matches = sum(1 for kw in keywords if query in kw.lower())
        keyword_score = keyword_matches / len(keywords)
        scores.append(keyword_score * 1.8)

    # Capabilities match (medium weight)
    capabilities = agent.get('capabilities', [])
    if capabilities:
        cap_text = ' '.join(capabilities).lower()
        cap_score = SequenceMatcher(None, query, cap_text).ratio()
        scores.append(cap_score * 1.0)

    # Category match (low weight)
    category = agent.get('category', '').lower()
    if query in category:
        scores.append(0.8)

    # Contains query as substring (bonus)
    searchable_text = f"{name} {description} {' '.join(capabilities)}".lower()
    if query in searchable_text:
        scores.append(0.5)

    # Return weighted average
    return sum(scores) / len(scores) if scores else 0.0
