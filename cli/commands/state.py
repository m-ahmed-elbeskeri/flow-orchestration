# cli/commands/state.py
"""State management commands."""

import click


@click.group()
def state():
    """Manage workflow states."""
    pass


@state.command()
def list():
    """List available states from all plugins."""
    click.echo("Available states:")
    # TODO: Implement state listing
    click.echo("State listing not yet implemented.")