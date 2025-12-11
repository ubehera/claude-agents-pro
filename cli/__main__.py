#!/usr/bin/env python3
"""
Claude Agents CLI - Main entry point

Usage:
    claude-agents install [--scope user|project]
    claude-agents list [--tier X]
    claude-agents search <query>
    claude-agents info <agent-name>
    claude-agents validate
    claude-agents score [agent-name]
"""

import click
from rich.console import Console
from pathlib import Path
import sys

from .commands import install, list_agents, search, info, validate, score

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="claude-agents")
def cli():
    """Claude Agents Pro - CLI for managing Claude Code agents."""
    pass


# Register commands
cli.add_command(install.install)
cli.add_command(list_agents.list_cmd)
cli.add_command(search.search)
cli.add_command(info.info)
cli.add_command(validate.validate)
cli.add_command(score.score)


def main():
    """Main execution entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
