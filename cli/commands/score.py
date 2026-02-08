"""Agent quality scoring command."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys

from ..utils.agent_finder import find_agent_file

console = Console()


@click.command()
@click.argument('agent_name', required=False, type=str)
@click.option(
    '--agents-dir',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Agents directory to score'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output JSON report file'
)
@click.option(
    '--min-score',
    type=float,
    default=0.7,
    help='Minimum acceptable quality score (0.0-1.0)'
)
def score(agent_name: str, agents_dir: Path, output: Path, min_score: float):
    """Run quality scoring on agents."""

    # Import quality scorer (from parent scripts directory)
    cli_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(cli_root / "scripts"))

    try:
        from quality_scorer import AgentQualityScorer
    except ImportError:
        console.print("[red]Error:[/red] quality-scorer.py not found in scripts/")
        raise click.Abort()

    # Determine agents directory
    if agents_dir is None:
        agents_dir = cli_root / "agents"

    if not agents_dir.exists():
        console.print(f"[red]Error:[/red] Agents directory not found: {agents_dir}")
        raise click.Abort()

    # Initialize scorer
    scorer = AgentQualityScorer(str(agents_dir))

    if agent_name:
        # Score specific agent
        agent_file = _find_agent_file(agents_dir, agent_name)
        if not agent_file:
            console.print(f"[red]Error:[/red] Agent '{agent_name}' not found")
            raise click.Abort()

        metrics, analysis = scorer.evaluate_agent(agent_file)
        _display_single_score(agent_name, metrics, analysis, min_score)

    else:
        # Score all agents
        console.print("[bold]Running quality analysis on all agents...[/bold]\n")

        results = scorer.evaluate_all_agents()
        _display_all_scores(results, min_score)

        # Save report if requested
        if output:
            import json

            report_data = {
                'results': {
                    k: {
                        'metrics': {
                            'completeness': v['metrics'].completeness,
                            'accuracy': v['metrics'].accuracy,
                            'usability': v['metrics'].usability,
                            'performance': v['metrics'].performance,
                            'maintainability': v['metrics'].maintainability,
                            'overall_score': v['metrics'].overall_score
                        },
                        'analysis': v['analysis']
                    }
                    for k, v in results.items()
                }
            }

            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(report_data, f, indent=2)

            console.print(f"\n[green]✓[/green] Report saved to {output}")


_find_agent_file = find_agent_file  # backward-compat alias


def _display_single_score(name: str, metrics, analysis: dict, min_score: float):
    """Display quality score for a single agent."""
    overall = metrics.overall_score

    # Header
    status = "[green]PASS[/green]" if overall >= min_score else "[red]FAIL[/red]"
    header = (
        f"[bold]Agent:[/bold] {name}\n"
        f"[bold]Overall Score:[/bold] {overall:.2f}/1.00 {status}\n"
        f"[bold]Tier:[/bold] {analysis.get('tier_classification', 'Unknown')}"
    )

    console.print(Panel(header, title="Quality Score", border_style="blue"))

    # Metrics breakdown
    console.print("\n[bold]Metrics Breakdown:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Weight", style="blue")
    table.add_column("Status", style="white")

    metric_weights = {
        'Completeness': (metrics.completeness, '20%'),
        'Accuracy': (metrics.accuracy, '20%'),
        'Usability': (metrics.usability, '20%'),
        'Performance': (metrics.performance, '15%'),
        'Maintainability': (metrics.maintainability, '10%'),
        'Security': (metrics.security, '15%')
    }

    for metric_name, (score, weight) in metric_weights.items():
        status = "✓" if score >= 0.8 else "⚠" if score >= 0.6 else "✗"
        color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"

        table.add_row(
            metric_name,
            f"[{color}]{score:.2f}[/{color}]",
            weight,
            f"[{color}]{status}[/{color}]"
        )

    console.print(table)

    # Recommendations
    if analysis.get('recommendations'):
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in analysis['recommendations']:
            console.print(f"  • {rec}")


def _display_all_scores(results: dict, min_score: float):
    """Display quality scores for all agents."""

    # Prepare data
    agent_scores = []
    for agent_path, data in results.items():
        metrics = data['metrics']
        analysis = data['analysis']

        agent_scores.append({
            'name': Path(agent_path).stem,
            'tier': Path(agent_path).parent.name,
            'score': metrics.overall_score,
            'completeness': metrics.completeness,
            'accuracy': metrics.accuracy,
            'usability': metrics.usability,
            'performance': metrics.performance,
            'maintainability': metrics.maintainability,
            'tier_class': analysis.get('tier_classification', 'Unknown')
        })

    # Sort by score descending
    agent_scores.sort(key=lambda x: x['score'], reverse=True)

    # Summary statistics
    total = len(agent_scores)
    passing = sum(1 for a in agent_scores if a['score'] >= min_score)
    avg_score = sum(a['score'] for a in agent_scores) / total if total > 0 else 0

    summary = (
        f"[bold]Total agents:[/bold] {total}\n"
        f"[bold]Passing (>={min_score}):[/bold] [green]{passing}[/green]\n"
        f"[bold]Failing (<{min_score}):[/bold] [red]{total - passing}[/red]\n"
        f"[bold]Average score:[/bold] {avg_score:.2f}\n"
        f"[bold]Pass rate:[/bold] {(passing/total)*100:.1f}%"
    )

    console.print(Panel(summary, title="Quality Summary", border_style="blue"))

    # Top and bottom performers
    console.print("\n[bold green]Top 5 Performers:[/bold green]")
    for agent in agent_scores[:5]:
        console.print(f"  {agent['score']:.2f} - {agent['name']} ({agent['tier']})")

    if len(agent_scores) > 5:
        console.print("\n[bold red]Bottom 5 Performers:[/bold red]")
        for agent in agent_scores[-5:]:
            console.print(f"  {agent['score']:.2f} - {agent['name']} ({agent['tier']})")

    # Detailed table
    console.print("\n[bold]Detailed Scores:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    table.add_column("Tier", style="blue")
    table.add_column("Overall", style="yellow")
    table.add_column("C", style="white", width=4)  # Completeness
    table.add_column("A", style="white", width=4)  # Accuracy
    table.add_column("U", style="white", width=4)  # Usability
    table.add_column("P", style="white", width=4)  # Performance
    table.add_column("M", style="white", width=4)  # Maintainability
    table.add_column("Status", style="white")

    for agent in agent_scores:
        status = "[green]✓[/green]" if agent['score'] >= min_score else "[red]✗[/red]"
        score_color = "green" if agent['score'] >= min_score else "red"

        table.add_row(
            agent['name'],
            agent['tier'],
            f"[{score_color}]{agent['score']:.2f}[/{score_color}]",
            f"{agent['completeness']:.2f}",
            f"{agent['accuracy']:.2f}",
            f"{agent['usability']:.2f}",
            f"{agent['performance']:.2f}",
            f"{agent['maintainability']:.2f}",
            status
        )

    console.print(table)
    console.print("\n[dim]C=Completeness, A=Accuracy, U=Usability, P=Performance, M=Maintainability[/dim]")
