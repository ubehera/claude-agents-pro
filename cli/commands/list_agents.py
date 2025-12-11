"""List agents command."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional

from ..utils.agent_parser import AgentParser

console = Console()


@click.command(name='list')
@click.option(
    '--tier',
    type=str,
    help='Filter by tier (e.g., 00-meta, 01-foundation)'
)
@click.option(
    '--scope',
    type=click.Choice(['user', 'project']),
    default='user',
    help='List agents from user or project scope'
)
@click.option(
    '--agents-dir',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Custom agents directory to list'
)
@click.option(
    '--format',
    type=click.Choice(['table', 'json', 'simple']),
    default='table',
    help='Output format'
)
def list_cmd(
    tier: Optional[str],
    scope: str,
    agents_dir: Optional[Path],
    format: str
):
    """List installed Claude Code agents."""

    # Determine agents directory
    if agents_dir is None:
        if scope == 'user':
            agents_dir = Path.home() / ".claude" / "agents"
        else:
            agents_dir = Path.cwd() / ".claude" / "agents"

        # Fallback to repository agents if not installed
        if not agents_dir.exists():
            cli_root = Path(__file__).parent.parent.parent
            agents_dir = cli_root / "agents"

    if not agents_dir.exists():
        console.print(f"[red]Error:[/red] Agents directory not found: {agents_dir}")
        raise click.Abort()

    # Parse agents
    parser = AgentParser()
    agents = []

    tier_dirs = sorted(agents_dir.glob("[0-9][0-9]-*")) if tier is None else [agents_dir / tier]

    for tier_dir in tier_dirs:
        if not tier_dir.is_dir():
            continue

        for agent_file in sorted(tier_dir.glob("*.md")):
            if agent_file.name in ['README.md', 'AGENT_CHECKLIST.md', 'TESTING.md']:
                continue

            try:
                agent_data = parser.parse_agent_file(agent_file)
                agent_data['tier'] = tier_dir.name
                agent_data['file'] = agent_file
                agents.append(agent_data)
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to parse {agent_file.name}: {e}")

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        return

    # Output based on format
    if format == 'table':
        _display_table(agents)
    elif format == 'json':
        _display_json(agents)
    else:  # simple
        _display_simple(agents)


def _display_table(agents):
    """Display agents in a formatted table."""
    table = Table(title="Claude Code Agents", show_header=True, header_style="bold magenta")

    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Tier", style="blue")
    table.add_column("Category", style="green")
    table.add_column("Complexity", style="yellow")
    table.add_column("Description", style="white")

    for agent in agents:
        table.add_row(
            agent.get('name', 'unknown'),
            agent.get('tier', 'unknown'),
            agent.get('category', 'unknown'),
            agent.get('complexity', 'unknown'),
            agent.get('description', '')[:60] + '...' if len(agent.get('description', '')) > 60 else agent.get('description', '')
        )

    console.print(table)
    console.print(f"\n[bold]Total agents:[/bold] {len(agents)}")


def _display_json(agents):
    """Display agents in JSON format."""
    import json

    output = [
        {
            'name': agent.get('name'),
            'tier': agent.get('tier'),
            'category': agent.get('category'),
            'complexity': agent.get('complexity'),
            'description': agent.get('description'),
            'capabilities': agent.get('capabilities', [])
        }
        for agent in agents
    ]

    console.print(json.dumps(output, indent=2))


def _display_simple(agents):
    """Display agents in simple list format."""
    current_tier = None

    for agent in agents:
        tier = agent.get('tier', 'unknown')
        if tier != current_tier:
            console.print(f"\n[bold blue]{tier}[/bold blue]")
            current_tier = tier

        console.print(f"  • {agent.get('name', 'unknown')} - {agent.get('description', '')}")
