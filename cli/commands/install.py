"""Agent installation command."""

import click
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from typing import Optional

console = Console()


@click.command()
@click.option(
    '--scope',
    type=click.Choice(['user', 'project']),
    default='user',
    help='Installation scope (user or project)'
)
@click.option(
    '--agents-dir',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Source agents directory (default: ./agents)'
)
@click.option(
    '--tier',
    type=str,
    help='Install only specific tier (e.g., 00-meta, 01-foundation)'
)
@click.option(
    '--agent',
    type=str,
    help='Install specific agent by name'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be installed without actually installing'
)
def install(
    scope: str,
    agents_dir: Optional[Path],
    tier: Optional[str],
    agent: Optional[str],
    dry_run: bool
):
    """Install Claude Code agents to user or project scope."""

    # Determine source directory
    if agents_dir is None:
        # Try to find agents directory relative to CLI location
        cli_root = Path(__file__).parent.parent.parent
        agents_dir = cli_root / "agents"

        if not agents_dir.exists():
            console.print("[red]Error:[/red] Agents directory not found. Use --agents-dir to specify.")
            raise click.Abort()

    # Determine target directory
    if scope == 'user':
        target_dir = Path.home() / ".claude" / "agents"
    else:  # project
        target_dir = Path.cwd() / ".claude" / "agents"

    # Collect agents to install
    agents_to_install = []

    if agent:
        # Install specific agent
        agent_file = _find_agent_file(agents_dir, agent)
        if not agent_file:
            console.print(f"[red]Error:[/red] Agent '{agent}' not found")
            raise click.Abort()
        agents_to_install.append(agent_file)

    elif tier:
        # Install specific tier
        tier_dir = agents_dir / tier
        if not tier_dir.exists():
            console.print(f"[red]Error:[/red] Tier '{tier}' not found")
            raise click.Abort()
        agents_to_install = list(tier_dir.glob("*.md"))

    else:
        # Install all agents
        for tier_dir in sorted(agents_dir.glob("[0-9][0-9]-*")):
            if tier_dir.is_dir():
                agents_to_install.extend(tier_dir.glob("*.md"))

    # Filter out non-agent files
    agents_to_install = [
        f for f in agents_to_install
        if f.is_file() and not f.name.startswith('.') and f.suffix == '.md'
        and f.name not in ['README.md', 'AGENT_CHECKLIST.md', 'TESTING.md']
    ]

    if not agents_to_install:
        console.print("[yellow]No agents found to install[/yellow]")
        return

    # Display installation plan
    console.print(Panel(
        f"[bold]Installation Plan[/bold]\n\n"
        f"Scope: [cyan]{scope}[/cyan]\n"
        f"Target: [cyan]{target_dir}[/cyan]\n"
        f"Agents: [cyan]{len(agents_to_install)}[/cyan]\n"
        f"Dry run: [cyan]{dry_run}[/cyan]",
        title="Claude Agents Installer",
        border_style="blue"
    ))

    if dry_run:
        console.print("\n[bold]Agents to be installed:[/bold]")
        for agent_file in agents_to_install:
            tier_name = agent_file.parent.name
            console.print(f"  • {tier_name}/{agent_file.name}")
        console.print(f"\n[yellow]Dry run complete. No files were modified.[/yellow]")
        return

    # Perform installation
    target_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Installing agents...", total=len(agents_to_install))

        installed_count = 0
        for agent_file in agents_to_install:
            try:
                # Preserve tier structure
                tier_name = agent_file.parent.name
                target_tier_dir = target_dir / tier_name
                target_tier_dir.mkdir(exist_ok=True)

                target_file = target_tier_dir / agent_file.name
                shutil.copy2(agent_file, target_file)

                installed_count += 1
                progress.update(task, advance=1)

            except Exception as e:
                console.print(f"[red]Failed to install {agent_file.name}:[/red] {e}")

        progress.update(task, completed=len(agents_to_install))

    console.print(f"\n[green]✓[/green] Successfully installed {installed_count} agents to {target_dir}")

    # Install meta-CLAUDE.md if available and user scope
    if scope == 'user' and not agent and not tier:
        meta_claude = agents_dir.parent / "meta-CLAUDE.md"
        if meta_claude.exists():
            target_claude = Path.home() / ".claude" / "CLAUDE.md"
            try:
                shutil.copy2(meta_claude, target_claude)
                console.print(f"[green]✓[/green] Installed CLAUDE.md to {target_claude}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not install CLAUDE.md: {e}")


def _find_agent_file(agents_dir: Path, agent_name: str) -> Optional[Path]:
    """Find an agent file by name across all tiers."""
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
