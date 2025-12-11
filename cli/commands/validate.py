"""Validate agents command."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

from ..utils.agent_parser import AgentParser

console = Console()


@click.command()
@click.option(
    '--agents-dir',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Agents directory to validate'
)
@click.option(
    '--strict',
    is_flag=True,
    help='Enable strict validation with additional checks'
)
@click.option(
    '--fix',
    is_flag=True,
    help='Attempt to auto-fix common issues'
)
def validate(agents_dir: Path, strict: bool, fix: bool):
    """Validate all agents against quality standards."""

    # Determine agents directory
    if agents_dir is None:
        cli_root = Path(__file__).parent.parent.parent
        agents_dir = cli_root / "agents"

    if not agents_dir.exists():
        console.print(f"[red]Error:[/red] Agents directory not found: {agents_dir}")
        raise click.Abort()

    console.print(f"[bold]Validating agents in:[/bold] {agents_dir}\n")

    # Validation rules
    required_frontmatter = ['name', 'description', 'category', 'complexity', 'model']
    required_sections = ['Core Expertise', 'Approach & Philosophy', 'Technical Implementation']

    # Collect all agent files
    agent_files = []
    for tier_dir in sorted(agents_dir.glob("[0-9][0-9]-*")):
        if tier_dir.is_dir():
            for agent_file in tier_dir.glob("*.md"):
                if agent_file.name not in ['README.md', 'AGENT_CHECKLIST.md', 'TESTING.md']:
                    agent_files.append(agent_file)

    if not agent_files:
        console.print("[yellow]No agent files found[/yellow]")
        return

    # Validation results
    results = []
    parser = AgentParser()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Validating agents...", total=len(agent_files))

        for agent_file in agent_files:
            issues = []

            try:
                # Parse agent
                agent_data = parser.parse_agent_file(agent_file)

                # Check required frontmatter fields
                for field in required_frontmatter:
                    if field not in agent_data or not agent_data[field]:
                        issues.append(f"Missing required field: {field}")

                # Validate name matches filename
                expected_name = agent_file.stem
                actual_name = agent_data.get('name', '')
                if actual_name != expected_name:
                    issues.append(f"Name mismatch: '{actual_name}' != '{expected_name}'")

                # Check model is valid
                valid_models = [
                    'claude-sonnet-4-5-20250929',
                    'claude-opus-4-5-20251101',
                    'claude-haiku-4-5-20250929'
                ]
                model = agent_data.get('model', '')
                if model and model not in valid_models:
                    issues.append(f"Invalid model: {model}")

                # Read full content for section checks
                with open(agent_file, 'r') as f:
                    content = f.read()

                # Check required sections
                if strict:
                    for section in required_sections:
                        if f"## {section}" not in content and f"# {section}" not in content:
                            issues.append(f"Missing section: {section}")

                # Check for code examples
                if strict and '```' not in content:
                    issues.append("No code examples found")

                # Check description length
                description = agent_data.get('description', '')
                if len(description) < 50:
                    issues.append("Description too short (min 50 chars)")

                # Tool count check
                tools = agent_data.get('tools', [])
                if isinstance(tools, list) and len(tools) > 7:
                    issues.append(f"Too many tools specified: {len(tools)} (max recommended: 7)")

            except yaml.YAMLError as e:
                issues.append(f"YAML parsing error: {str(e)}")
            except Exception as e:
                issues.append(f"Parse error: {str(e)}")

            results.append({
                'file': agent_file,
                'tier': agent_file.parent.name,
                'name': agent_file.stem,
                'issues': issues,
                'valid': len(issues) == 0
            })

            progress.update(task, advance=1)

    # Display results
    console.print("\n[bold]Validation Results:[/bold]\n")

    # Summary
    total = len(results)
    valid = sum(1 for r in results if r['valid'])
    invalid = total - valid

    console.print(f"Total agents: [cyan]{total}[/cyan]")
    console.print(f"Valid: [green]{valid}[/green]")
    console.print(f"Invalid: [red]{invalid}[/red]")
    console.print(f"Pass rate: [cyan]{(valid/total)*100:.1f}%[/cyan]\n")

    # Show invalid agents
    if invalid > 0:
        table = Table(title="Validation Issues", show_header=True, header_style="bold red")
        table.add_column("Agent", style="yellow")
        table.add_column("Tier", style="blue")
        table.add_column("Issues", style="white")

        for result in results:
            if not result['valid']:
                issues_text = "\n".join(f"• {issue}" for issue in result['issues'])
                table.add_row(
                    result['name'],
                    result['tier'],
                    issues_text
                )

        console.print(table)

        # Exit with error code if validation failed
        raise click.Abort()
    else:
        console.print("[green]✓ All agents passed validation![/green]")
