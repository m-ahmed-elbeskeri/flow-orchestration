# cli/commands/plugin.py
"""Plugin management commands."""

import click
from pathlib import Path
import yaml
import json
from typing import Optional

try:
    from enterprise.ai.plugin_discovery import PluginDiscovery
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


@click.group()
def plugin():
    """Manage plugins."""
    pass


@plugin.command()
@click.option('--format', type=click.Choice(['table', 'yaml', 'json']), default='table', help='Output format')
@click.option('--plugin-dir', type=click.Path(exists=True), help='Custom plugin directory')
def list(format, plugin_dir):
    """List all available plugins."""
    
    if DISCOVERY_AVAILABLE:
        # Use the sophisticated discovery system
        plugin_dir_path = Path(plugin_dir) if plugin_dir else None
        discovery = PluginDiscovery(plugin_dir_path)
        plugins = discovery.discover_all_plugins()
        
        if format == 'table':
            if not plugins:
                click.echo("No plugins found.")
                return
                
            click.echo("Available plugins:")
            click.echo("-" * 80)
            
            for plugin_name, plugin_info in plugins.items():
                click.echo(f"📦 {plugin_name} (v{plugin_info['version']})")
                click.echo(f"   Description: {plugin_info['description']}")
                if plugin_info.get('author'):
                    click.echo(f"   Author: {plugin_info['author']}")
                
                # Show states
                states = plugin_info.get('states', {})
                if states:
                    click.echo(f"   States ({len(states)}):")
                    for state_name, state_info in states.items():
                        state_desc = state_info.get('description', 'No description')
                        click.echo(f"     • {state_name}: {state_desc}")
                        
                        # Show inputs if any
                        inputs = state_info.get('inputs', {})
                        if inputs:
                            required_inputs = [k for k, v in inputs.items() if v.get('required', False)]
                            optional_inputs = [k for k, v in inputs.items() if not v.get('required', False)]
                            
                            input_summary = []
                            if required_inputs:
                                input_summary.append(f"{len(required_inputs)} required")
                            if optional_inputs:
                                input_summary.append(f"{len(optional_inputs)} optional")
                            
                            if input_summary:
                                click.echo(f"       Inputs: {', '.join(input_summary)}")
                
                # Show dependencies if any
                dependencies = plugin_info.get('dependencies', [])
                if dependencies:
                    click.echo(f"   Dependencies: {', '.join(dependencies)}")
                
                click.echo()
        
        elif format == 'yaml':
            click.echo(yaml.dump(plugins, default_flow_style=False))
        
        elif format == 'json':
            click.echo(json.dumps(plugins, indent=2))
    
    else:
        # Fallback to simple directory scanning
        _list_plugins_fallback(plugin_dir)


def _list_plugins_fallback(plugin_dir: Optional[str] = None):
    """Fallback method for listing plugins when discovery system is not available."""
    plugins_dir = Path(plugin_dir) if plugin_dir else Path('plugins/contrib')
    
    if not plugins_dir.exists():
        click.echo("No plugins directory found.")
        return
    
    click.echo("Available plugins:")
    click.echo("-" * 40)
    
    found_plugins = False
    for plugin_path in plugins_dir.iterdir():
        if plugin_path.is_dir() and (plugin_path / 'manifest.yaml').exists():
            found_plugins = True
            try:
                with open(plugin_path / 'manifest.yaml', 'r') as f:
                    manifest = yaml.safe_load(f)
                
                name = manifest.get('name', plugin_path.name)
                version = manifest.get('version', '1.0.0')
                description = manifest.get('description', 'No description')
                author = manifest.get('author', 'Unknown')
                
                click.echo(f"📦 {name} (v{version})")
                click.echo(f"   Description: {description}")
                click.echo(f"   Author: {author}")
                
                # Show states if defined
                states = manifest.get('states', {})
                if states:
                    click.echo(f"   States: {', '.join(states.keys())}")
                
                click.echo()
                
            except Exception as e:
                click.echo(f"❌ Error reading {plugin_path.name}: {e}")
    
    if not found_plugins:
        click.echo("No plugins found.")


@plugin.command()
@click.argument('plugin_name')
@click.option('--plugin-dir', type=click.Path(exists=True), help='Custom plugin directory')
def info(plugin_name, plugin_dir):
    """Show detailed information about a specific plugin."""
    
    if DISCOVERY_AVAILABLE:
        plugin_dir_path = Path(plugin_dir) if plugin_dir else None
        discovery = PluginDiscovery(plugin_dir_path)
        plugins = discovery.discover_all_plugins()
        
        if plugin_name not in plugins:
            click.echo(f"Plugin '{plugin_name}' not found.")
            available = ', '.join(plugins.keys())
            if available:
                click.echo(f"Available plugins: {available}")
            return
        
        plugin_info = plugins[plugin_name]
        
        # Display detailed information
        click.echo(f"📦 {plugin_info['name']} (v{plugin_info['version']})")
        click.echo(f"Description: {plugin_info['description']}")
        if plugin_info.get('author'):
            click.echo(f"Author: {plugin_info['author']}")
        
        # Dependencies
        dependencies = plugin_info.get('dependencies', [])
        if dependencies:
            click.echo(f"Dependencies: {', '.join(dependencies)}")
        
        # Global inputs/outputs
        if plugin_info.get('inputs'):
            click.echo("\nGlobal Inputs:")
            for input_name, input_config in plugin_info['inputs'].items():
                input_type = input_config.get('type', 'any')
                required = input_config.get('required', False)
                desc = input_config.get('description', '')
                req_text = '(required)' if required else '(optional)'
                click.echo(f"  • {input_name} ({input_type}) {req_text}: {desc}")
        
        if plugin_info.get('outputs'):
            click.echo("\nGlobal Outputs:")
            for output_name, output_config in plugin_info['outputs'].items():
                output_type = output_config.get('type', 'any')
                desc = output_config.get('description', '')
                click.echo(f"  • {output_name} ({output_type}): {desc}")
        
        # States
        states = plugin_info.get('states', {})
        if states:
            click.echo(f"\nStates ({len(states)}):")
            for state_name, state_info in states.items():
                click.echo(f"\n  🔧 {state_name}")
                if state_info.get('description'):
                    click.echo(f"     Description: {state_info['description']}")
                
                # State inputs
                inputs = state_info.get('inputs', {})
                if inputs:
                    click.echo("     Inputs:")
                    for input_name, input_config in inputs.items():
                        input_type = input_config.get('type', 'any')
                        required = input_config.get('required', False)
                        desc = input_config.get('description', '')
                        default = input_config.get('default')
                        
                        req_text = 'required' if required else 'optional'
                        default_text = f', default: {default}' if default is not None else ''
                        
                        click.echo(f"       • {input_name} ({input_type}, {req_text}{default_text}): {desc}")
                
                # State outputs
                outputs = state_info.get('outputs', {})
                if outputs:
                    click.echo("     Outputs:")
                    for output_name, output_config in outputs.items():
                        output_type = output_config.get('type', 'any')
                        desc = output_config.get('description', '')
                        click.echo(f"       • {output_name} ({output_type}): {desc}")
                
                # Examples
                examples = state_info.get('examples', [])
                if examples:
                    click.echo("     Examples:")
                    for example in examples[:3]:  # Show up to 3 examples
                        click.echo(f"       • {example}")
                
                # Docstring if available
                if state_info.get('docstring'):
                    click.echo(f"     Documentation: {state_info['docstring']}")
        
        else:
            click.echo("\nNo states defined.")
    
    else:
        click.echo("Plugin discovery system not available. Install enterprise dependencies for detailed plugin information.")


@plugin.command()
@click.option('--plugin-dir', type=click.Path(exists=True), help='Custom plugin directory')
def capabilities(plugin_dir):
    """Show all plugin capabilities in a format suitable for AI prompts."""
    
    if not DISCOVERY_AVAILABLE:
        click.echo("Plugin discovery system not available. Install enterprise dependencies.")
        return
    
    plugin_dir_path = Path(plugin_dir) if plugin_dir else None
    discovery = PluginDiscovery(plugin_dir_path)
    
    capabilities_prompt = discovery.get_plugin_capabilities_prompt()
    
    if capabilities_prompt:
        click.echo("=== Plugin Capabilities ===")
        click.echo(capabilities_prompt)
    else:
        click.echo("No plugins found or no capabilities defined.")


@plugin.command()
@click.argument('query')
@click.option('--plugin-dir', type=click.Path(exists=True), help='Custom plugin directory')
def search(query, plugin_dir):
    """Search for plugins and states matching the query."""
    
    if not DISCOVERY_AVAILABLE:
        click.echo("Plugin discovery system not available. Install enterprise dependencies.")
        return
    
    plugin_dir_path = Path(plugin_dir) if plugin_dir else None
    discovery = PluginDiscovery(plugin_dir_path)
    plugins = discovery.discover_all_plugins()
    
    query_lower = query.lower()
    found_results = False
    
    click.echo(f"Searching for: '{query}'")
    click.echo("-" * 40)
    
    for plugin_name, plugin_info in plugins.items():
        plugin_matches = []
        
        # Check plugin name and description
        if (query_lower in plugin_name.lower() or 
            query_lower in plugin_info.get('description', '').lower()):
            plugin_matches.append("plugin name/description")
        
        # Check states
        state_matches = []
        for state_name, state_info in plugin_info.get('states', {}).items():
            if (query_lower in state_name.lower() or 
                query_lower in state_info.get('description', '').lower()):
                state_matches.append(state_name)
        
        if plugin_matches or state_matches:
            found_results = True
            click.echo(f"📦 {plugin_name}")
            
            if plugin_matches:
                click.echo(f"   Matched: {', '.join(plugin_matches)}")
            
            if state_matches:
                click.echo(f"   States: {', '.join(state_matches)}")
                for state_name in state_matches:
                    state_info = plugin_info['states'][state_name]
                    desc = state_info.get('description', 'No description')
                    click.echo(f"     • {state_name}: {desc}")
            
            click.echo()
    
    if not found_results:
        click.echo("No plugins or states found matching the query.")
        click.echo("\nTip: Try broader search terms or use 'plugin list' to see all available plugins.")