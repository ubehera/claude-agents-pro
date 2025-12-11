"""Agent info command."""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from ..utils.agent_parser import AgentParser

console = Console()


@click.command()
@click.argument('agent_name', type=str)
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
    help='Custom agents directory'
)
@click.option(
    '--full',
    is_flag=True,
    help='Show full agent content'
)
def info(agent_name: str, scope: str, agents_dir: Path, full: bool):
    """Show detailed information about a specific agent."""

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

    # Find agent file
    agent_file = _find_agent_file(agents_dir, agent_name)

    if not agent_file:
        console.print(f"[red]Error:[/red] Agent '{agent_name}' not found")
        raise click.Abort()

    # Parse agent
    parser = AgentParser()
    agent_data = parser.parse_agent_file(agent_file)

    # Display agent info
    _display_agent_info(agent_data, agent_file, full)


def _find_agent_file(agents_dir: Path, agent_name: str) -> Path | None:
    """Find an agent file by name."""
    # Try exact match first
    for tier_dir in agents_dir.glob("[0-9][0-9]-*"):
        if tier_dir.is_dir():
            agent_file = tier_dir / f"{agent_name}.md"
            if agent_file.exists():
                return agent_file

    # Try fuzzy match
    for tier_dir in agents_dir.glob("[0-9][0-9]-*"):
        if tier_dir.is_dir():
            for agent_file in tier_dir.glob("*.md"):
                if agent_name.lower() in agent_file.stem.lower():
                    return agent_file

    return None


def _display_agent_info(agent_data: dict, agent_file: Path, show_full: bool):
    """Display formatted agent information."""

    # Header panel
    tier = agent_file.parent.name
    header = (
        f"[bold cyan]{agent_data.get('name', 'Unknown')}[/bold cyan]\n"
        f"Tier: [blue]{tier}[/blue] | "
        f"Category: [green]{agent_data.get('category', 'unknown')}[/green] | "
        f"Complexity: [yellow]{agent_data.get('complexity', 'unknown')}[/yellow]\n\n"
        f"{agent_data.get('description', 'No description available')}"
    )

    console.print(Panel(header, title="Agent Information", border_style="blue"))

    # Metadata
    console.print("\n[bold]Metadata:[/bold]")
    console.print(f"  Model: [cyan]{agent_data.get('model', 'N/A')}[/cyan]")
    if agent_data.get('model_rationale'):
        console.print(f"  Rationale: {agent_data.get('model_rationale')}")
    console.print(f"  File: [dim]{agent_file}[/dim]")

    # Capabilities
    if agent_data.get('capabilities'):
        console.print("\n[bold]Capabilities:[/bold]")
        for cap in agent_data['capabilities']:
            console.print(f"  • {cap}")

    # Auto-activation
    auto_activate = agent_data.get('auto_activate', {})
    if auto_activate:
        console.print("\n[bold]Auto-activation:[/bold]")

        if auto_activate.get('keywords'):
            console.print("  Keywords:", ", ".join(f"[cyan]{kw}[/cyan]" for kw in auto_activate['keywords']))

        if auto_activate.get('conditions'):
            console.print("  Conditions:")
            for condition in auto_activate['conditions']:
                console.print(f"    • {condition}")

    # Tools (if specified)
    tools = agent_data.get('tools')
    if tools:
        console.print("\n[bold]Tools:[/bold]")
        if isinstance(tools, list):
            for tool in tools:
                console.print(f"  • {tool}")
        else:
            console.print(f"  {tools}")

    # Full content
    if show_full:
        console.print("\n[bold]Full Content:[/bold]")
        with open(agent_file, 'r') as f:
            content = f.read()
            # Remove frontmatter for display
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()

            md = Markdown(content)
            console.print(md)
