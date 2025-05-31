# cli/main.py
"""Main CLI entry point for Workflow Orchestrator."""

import click
import asyncio
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Workflow Orchestrator CLI - Manage and create workflows with AI assistance."""
    pass


# Import and register command groups
def register_commands():
    """Register all CLI command groups."""
    # Workflow commands
    from cli.commands.workflow import workflow
    cli.add_command(workflow)
    
    # State commands
    from cli.commands.state import state
    cli.add_command(state)
    
    # Plugin commands
    from cli.commands.plugin import plugin
    cli.add_command(plugin)

    # Generate code commands
    from core.generator.cli import generate
    cli.add_command(generate)
    
    # Copilot commands (AI features)
    try:
        from cli.commands.copilot import copilot
        cli.add_command(copilot)
    except ImportError:
        click.echo("AI Copilot features not available. Install enterprise dependencies.", err=True)


# Register all commands
register_commands()


if __name__ == '__main__':
    cli()