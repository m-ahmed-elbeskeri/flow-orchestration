# core/generator/cli.py
"""CLI commands for YAML-to-code generation with auto-install functionality."""

import click
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
import yaml
import shutil

from .engine import CodeGenerator
from .parser import parse_workflow_file, ValidationError, WorkflowParser

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--auto-install', is_flag=True, help='Automatically install missing plugin dependencies')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def generate(ctx, auto_install: bool, verbose: bool):
    """Generate workflow code from YAML definitions."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store global options in context
    ctx.obj['auto_install'] = auto_install
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('plugins.registry').setLevel(logging.INFO)
    
    if auto_install:
        click.echo("🔧 Auto-install mode enabled - will install missing dependencies")


@generate.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output directory (default: ./generated-<workflow-name>)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing output directory')
@click.option('--validate-only', is_flag=True, help='Only validate, do not generate')
@click.pass_context
def yaml_to_code(ctx, yaml_file: Path, output: Optional[Path], force: bool, validate_only: bool):
    """Generate standalone code from YAML workflow definition."""
    auto_install = ctx.obj.get('auto_install', False)
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Setup plugin registry with auto-install
        from plugins.registry import plugin_registry
        if auto_install:
            plugin_registry.set_auto_install(True)
            click.echo("📦 Loading plugins with auto-install enabled...")
        
        # Parse and validate
        click.echo(f"📖 Parsing workflow definition: {yaml_file}")
        
        # Use parser with better error handling
        parser = WorkflowParser()
        spec = parser.parse_file(yaml_file)
        
        # Show installation stats if auto-install was used
        if auto_install:
            stats = plugin_registry.get_installation_stats()
            if stats['installed_dependencies']:
                click.echo(f"📦 Auto-installed dependencies: {', '.join(stats['installed_dependencies'])}")
            if stats['failed_installations']:
                click.echo(f"⚠️  Failed to install: {', '.join(stats['failed_installations'])}")
        
        # Show validation results
        if parser.warnings:
            click.echo(f"⚠️  Found {len(parser.warnings)} warnings:")
            for warning in parser.warnings:
                click.echo(f"   • {warning.path}: {warning.message}")
        
        if parser.errors:
            click.echo(f"❌ Found {len(parser.errors)} errors:")
            for error in parser.errors:
                click.echo(f"   • {error.path}: {error.message}")
            sys.exit(1)
        
        click.echo(f"✅ Successfully parsed workflow: {spec.name} v{spec.version}")
        click.echo(f"   📄 Description: {spec.description}")
        click.echo(f"   👤 Author: {spec.author}")
        click.echo(f"   🔧 States: {len(spec.states)}")
        click.echo(f"   🔌 Integrations: {len(spec.integrations)}")
        
        if validate_only:
            click.echo("✅ Validation completed successfully")
            return
        
        # Set output directory
        if not output:
            output = Path(f"./generated-{spec.snake_case_name}")
        
        # Check for existing directory
        if output.exists() and not force:
            click.echo(f"❌ Output directory {output} already exists. Use --force to overwrite.")
            sys.exit(1)
        
        # Generate code
        click.echo(f"🏗️  Generating code in: {output}")
        
        try:
            generator = CodeGenerator()
            generator.generate_from_spec(spec, output)
        except Exception as e:
            click.echo(f"❌ Code generation failed: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        # Success message with next steps
        click.echo(f"")
        click.echo(f"🎉 Successfully generated workflow project!")
        click.echo(f"📁 Location: {output.absolute()}")
        click.echo(f"")
        click.echo(f"📝 Next steps:")
        click.echo(f"   cd {output}")
        click.echo(f"   pip install -r requirements.txt")
        
        if spec.all_secrets:
            click.echo(f"   cp .env.template .env")
            click.echo(f"   # Edit .env with your credentials:")
            for secret in spec.all_secrets:
                click.echo(f"   #   {secret}=your_value_here")
        
        click.echo(f"   python run.py")
        
        if spec.schedule:
            click.echo(f"")
            click.echo(f"⏰ Workflow has schedule: {spec.schedule.cron}")
            click.echo(f"   Consider setting up cron or using a scheduler")
        
        # Show file structure if verbose
        if verbose:
            click.echo(f"")
            click.echo(f"📁 Generated files:")
            for file_path in sorted(output.rglob("*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output)
                    click.echo(f"   {rel_path}")
        
    except ValueError as e:
        click.echo(f"Validation failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"File not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        click.echo(f"Invalid YAML syntax: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@generate.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate(ctx, yaml_file: Path):
    """Validate YAML workflow definition."""
    auto_install = ctx.obj.get('auto_install', False)
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Setup plugin registry
        from plugins.registry import plugin_registry
        if auto_install:
            plugin_registry.set_auto_install(True)
            click.echo("📦 Loading plugins with auto-install enabled...")
        
        click.echo(f"🔍 Validating workflow: {yaml_file}")
        
        parser = WorkflowParser()
        spec = parser.parse_file(yaml_file)
        
        # Show installation stats if auto-install was used
        if auto_install:
            stats = plugin_registry.get_installation_stats()
            if stats['installed_dependencies']:
                click.echo(f"📦 Auto-installed: {', '.join(stats['installed_dependencies'])}")
        
        # Show validation results
        if parser.warnings:
            click.echo(f"⚠️  Found {len(parser.warnings)} warnings:")
            for warning in parser.warnings:
                severity_icon = "⚠️" if warning.severity == "warning" else "ℹ️"
                click.echo(f"   {severity_icon} {warning.path}: {warning.message}")
        
        if parser.errors:
            click.echo(f"❌ Found {len(parser.errors)} errors:")
            for error in parser.errors:
                click.echo(f"   • {error.path}: {error.message}")
            sys.exit(1)
        else:
            click.echo("✅ Workflow definition is valid!")
            
        # Show workflow summary
        click.echo(f"")
        click.echo(f"📊 Workflow Summary:")
        click.echo(f"   Name: {spec.name}")
        click.echo(f"   Version: {spec.version}")
        click.echo(f"   Author: {spec.author}")
        click.echo(f"   Description: {spec.description}")
        click.echo(f"   States: {len(spec.states)}")
        
        if spec.required_integrations:
            click.echo(f"   Required integrations: {', '.join(spec.required_integrations)}")
        else:
            click.echo(f"   Required integrations: None (builtin states only)")
        
        if spec.all_secrets:
            click.echo(f"   Secrets required: {', '.join(spec.all_secrets)}")
        else:
            click.echo(f"   Secrets required: None")
        
        if spec.schedule:
            click.echo(f"   Schedule: {spec.schedule.cron} ({spec.schedule.timezone})")
        
        # Show state details if verbose
        if verbose:
            click.echo(f"")
            click.echo(f"🔧 States Detail:")
            for state in spec.states:
                click.echo(f"   • {state.name} ({state.type})")
                if state.description:
                    click.echo(f"     Description: {state.description}")
                if state.dependencies:
                    deps = [dep.name for dep in state.dependencies]
                    click.echo(f"     Dependencies: {', '.join(deps)}")
                if state.transitions:
                    trans = [f"{t.condition}→{t.target}" for t in state.transitions]
                    click.echo(f"     Transitions: {', '.join(trans)}")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@generate.command()
@click.option('--search', '-s', type=str, help='Search plugins by name or description')
@click.option('--show-deps', is_flag=True, help='Show dependency information')
@click.option('--show-failed', is_flag=True, help='Show plugins that failed to load')
@click.pass_context
def list_plugins(ctx, search: Optional[str], show_deps: bool, show_failed: bool):
    """List available plugins and integrations."""
    auto_install = ctx.obj.get('auto_install', False)
    verbose = ctx.obj.get('verbose', False)
    
    click.echo("🔌 Available Plugins:")
    
    try:
        # Setup plugin registry
        from plugins.registry import plugin_registry
        if auto_install:
            plugin_registry.set_auto_install(True)
            click.echo("📦 Loading plugins with auto-install enabled...")
        
        # Try to import and use enhanced discovery
        try:
            from core.generator.discovery import plugin_discovery
            plugins = plugin_discovery.discover_all_plugins()
            use_discovery = True
        except ImportError:
            # Fallback to basic registry
            plugin_registry.load_plugins()
            plugins = {p['name']: p for p in plugin_registry.list_plugins()}
            use_discovery = False
        
        # Show installation stats if auto-install was used
        if auto_install:
            stats = plugin_registry.get_installation_stats()
            if stats['installed_dependencies']:
                click.echo(f"📦 Auto-installed: {', '.join(stats['installed_dependencies'])}")
            if stats['failed_installations']:
                click.echo(f"⚠️  Installation failed: {', '.join(stats['failed_installations'])}")
        
        if not plugins:
            click.echo("   ❌ No plugins found!")
            click.echo("")
            _show_plugin_troubleshooting(verbose)
            return
        
        # Filter plugins
        if search:
            search_lower = search.lower()
            filtered_plugins = {
                name: info for name, info in plugins.items()
                if (search_lower in name.lower() or 
                    search_lower in info.get('description', '').lower())
            }
            plugins = filtered_plugins
            
            if not plugins:
                click.echo(f"   ❌ No plugins found matching '{search}'")
                return
        
        # Filter by status
        if not show_failed:
            plugins = {
                name: info for name, info in plugins.items()
                if info.get('status', 'unknown') == 'loaded'
            }
        
        # Group by integration type if using enhanced discovery
        if use_discovery:
            _display_plugins_by_type(plugins, verbose, show_deps)
        else:
            _display_plugins_simple(plugins, verbose, show_deps)
        
        # Summary
        loaded_count = sum(1 for p in plugins.values() if p.get('status') == 'loaded')
        failed_count = sum(1 for p in plugins.values() if p.get('status') in ['failed', 'error'])
        
        click.echo(f"\n📈 Summary: {loaded_count} loaded, {failed_count} failed")
        
        if failed_count > 0 and not show_failed:
            click.echo("💡 Use --show-failed to see failed plugins")
        
        if not verbose:
            click.echo("💡 Use --verbose for detailed information")
        
        if not show_deps:
            click.echo("💡 Use --show-deps to see dependency information")
        
    except Exception as e:
        click.echo(f"❌ Error discovering plugins: {str(e)}")
        _show_plugin_troubleshooting(verbose)


def _display_plugins_by_type(plugins: dict, verbose: bool, show_deps: bool):
    """Display plugins grouped by type."""
    by_type = {}
    for plugin_name, plugin_info in plugins.items():
        integration_type = plugin_info.get('integration_type', 'utility')
        if integration_type not in by_type:
            by_type[integration_type] = []
        by_type[integration_type].append(plugin_info)
    
    for integration_type, type_plugins in sorted(by_type.items()):
        click.echo(f"\n📂 {integration_type.upper()}:")
        
        for plugin in sorted(type_plugins, key=lambda x: x['name']):
            _display_plugin_info(plugin, verbose, show_deps)


def _display_plugins_simple(plugins: dict, verbose: bool, show_deps: bool):
    """Display plugins in simple list."""
    for plugin_name, plugin_info in sorted(plugins.items()):
        _display_plugin_info(plugin_info, verbose, show_deps)


def _display_plugin_info(plugin_info: dict, verbose: bool, show_deps: bool):
    """Display information about a single plugin."""
    status = plugin_info.get('status', 'unknown')
    status_icon = {
        'loaded': '✅',
        'failed': '❌',
        'error': '⚠️'
    }.get(status, '❓')
    
    click.echo(f"   {status_icon} {plugin_info['name']} v{plugin_info['version']}")
    click.echo(f"      {plugin_info['description']}")
    
    if plugin_info.get('author'):
        click.echo(f"      👤 Author: {plugin_info['author']}")
    
    # Show dependencies if requested
    if show_deps and plugin_info.get('dependencies'):
        deps = plugin_info['dependencies']
        if len(deps) <= 5:
            click.echo(f"      📦 Dependencies: {', '.join(deps)}")
        else:
            click.echo(f"      📦 Dependencies: {', '.join(deps[:3])} and {len(deps)-3} more")
    
    # Show status details for non-loaded plugins
    if status != 'loaded':
        if status == 'failed':
            click.echo(f"      ❌ Status: Failed to load")
            if verbose:
                click.echo(f"         Reason: {plugin_info['description']}")
                if show_deps:
                    click.echo("         💡 Try: workflow-gen generate install-deps <plugin_name>")
        return
    
    # Show states for loaded plugins
    states = plugin_info.get('states', [])
    if states:
        if verbose:
            click.echo("      🔧 States:")
            for state in sorted(states):
                full_name = f"{plugin_info['name']}.{state}"
                click.echo(f"         • {full_name}")
        else:
            states_text = ", ".join(f"{plugin_info['name']}.{s}" for s in sorted(states))
            if len(states_text) > 80:
                states_text = states_text[:77] + "..."
            click.echo(f"      🔧 States: {states_text}")


def _show_plugin_troubleshooting(verbose: bool):
    """Show plugin troubleshooting information."""
    click.echo("   📁 Plugin search locations:")
    
    try:
        from plugins.registry import plugin_registry
        for plugin_dir in plugin_registry._plugin_dirs:
            exists = "✅" if plugin_dir.exists() else "❌"
            click.echo(f"      {exists} {plugin_dir}")
            if verbose and plugin_dir.exists():
                subdirs = [d for d in plugin_dir.iterdir() if d.is_dir()]
                for subdir in subdirs:
                    has_manifest = (subdir / "manifest.yaml").exists()
                    status = "📦" if has_manifest else "📁"
                    click.echo(f"         {status} {subdir.name}")
    except Exception as e:
        click.echo(f"      Error accessing plugin directories: {e}")
    
    click.echo("")
    click.echo("   💡 To fix plugin issues:")
    click.echo("      1. Run: workflow-gen generate setup-plugins --create-example")
    click.echo("      2. Install missing dependencies with --auto-install")
    click.echo("      3. Or manually: workflow-gen generate install-deps <plugin_name>")
    click.echo("      4. Check plugin manifest.yaml files are valid")


@generate.command()
@click.argument('plugin_name', type=str)
@click.option('--force', '-f', is_flag=True, help='Force reinstall even if already installed')
@click.option('--dry-run', is_flag=True, help='Show what would be installed without installing')
def install_deps(plugin_name: str, force: bool, dry_run: bool):
    """Install dependencies for a specific plugin."""
    try:
        from plugins.registry import plugin_registry
        
        if dry_run:
            click.echo(f"🔍 Checking dependencies for plugin: {plugin_name}")
        else:
            click.echo(f"📦 Installing dependencies for plugin: {plugin_name}")
        
        # Load plugins to get info
        plugin_registry.load_plugins()
        
        # Check if plugin exists
        all_plugins = plugin_registry.list_plugins()
        plugin_info = None
        for p in all_plugins:
            if p['name'] == plugin_name:
                plugin_info = p
                break
        
        if not plugin_info:
            click.echo(f"❌ Plugin '{plugin_name}' not found")
            available = [p['name'] for p in all_plugins]
            if available:
                click.echo(f"Available plugins: {', '.join(available)}")
            return
        
        # Get dependencies
        dependencies = plugin_info.get('dependencies', [])
        if not dependencies:
            click.echo("✅ No dependencies required")
            return
        
        click.echo(f"📋 Required dependencies:")
        for dep in dependencies:
            click.echo(f"   • {dep}")
        
        if dry_run:
            click.echo(f"\n💡 To install, run: workflow-gen generate install-deps {plugin_name}")
            return
        
        # Install dependencies
        success = plugin_registry.install_plugin_dependencies(plugin_name, force=force)
        
        if success:
            click.echo("\n🔄 Refreshing plugin registry...")
            plugin_registry.refresh()
            
            # Check if plugin loads now
            updated_plugin = plugin_registry.get_plugin(plugin_name)
            if updated_plugin:
                click.echo(f"✅ Plugin {plugin_name} now loads successfully!")
            else:
                click.echo(f"⚠️  Plugin {plugin_name} still has issues")
                
                # Show any remaining errors
                errors = plugin_registry.get_plugin_errors()
                if plugin_name in errors:
                    click.echo(f"   Error: {errors[plugin_name]}")
        else:
            click.echo(f"❌ Failed to install dependencies for {plugin_name}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@generate.command()
@click.option('--all-plugins', '-a', is_flag=True, help='Install dependencies for all plugins')
@click.option('--force', '-f', is_flag=True, help='Force reinstall dependencies')
def install_all_deps(all_plugins: bool, force: bool):
    """Install dependencies for all plugins."""
    try:
        from plugins.registry import plugin_registry
        
        click.echo("📦 Installing dependencies for all plugins...")
        
        # Enable auto-install
        plugin_registry.set_auto_install(True)
        
        if force:
            # Clear failed installations cache
            plugin_registry._failed_installations.clear()
        
        # Refresh registry with auto-install
        plugin_registry.refresh()
        
        # Show results
        stats = plugin_registry.get_installation_stats()
        plugins = plugin_registry.list_plugins()
        
        loaded_count = sum(1 for p in plugins if p.get('status') == 'loaded')
        total_count = len(plugins)
        
        click.echo(f"\n📊 Results:")
        click.echo(f"   Loaded plugins: {loaded_count}/{total_count}")
        
        if stats['installed_dependencies']:
            click.echo(f"   Installed: {', '.join(stats['installed_dependencies'])}")
        
        if stats['failed_installations']:
            click.echo(f"   Failed: {', '.join(stats['failed_installations'])}")
        
        if loaded_count == total_count:
            click.echo("🎉 All plugins loaded successfully!")
        else:
            failed_count = total_count - loaded_count
            click.echo(f"⚠️  {failed_count} plugins still have issues")
            click.echo("💡 Run with --verbose to see details")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@generate.command()
@click.option('--create-example', is_flag=True, help='Create example plugin structure')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
def setup_plugins(create_example: bool, force: bool):
    """Setup plugin directory structure."""
    plugins_dir = Path.cwd() / "plugins"
    
    # Create main plugins directory
    if not plugins_dir.exists():
        plugins_dir.mkdir(parents=True)
        click.echo(f"✅ Created plugins directory: {plugins_dir}")
    else:
        click.echo(f"📁 Plugins directory already exists: {plugins_dir}")
    
    # Create __init__.py
    init_file = plugins_dir / "__init__.py"
    if not init_file.exists() or force:
        init_file.write_text('"""Plugins directory."""\n')
        click.echo("✅ Created plugins/__init__.py")
    
    # Create contrib directory
    contrib_dir = plugins_dir / "contrib"
    if not contrib_dir.exists():
        contrib_dir.mkdir()
        (contrib_dir / "__init__.py").write_text('"""Contributed plugins."""\n')
        click.echo("✅ Created contrib/ directory for community plugins")
    
    if create_example:
        example_dir = plugins_dir / "example"
        if example_dir.exists() and not force:
            click.echo(f"❌ Example plugin already exists at {example_dir}")
            click.echo("   Use --force to overwrite")
        else:
            _create_example_plugin(example_dir)
            click.echo(f"✅ Created example plugin: {example_dir}")
            click.echo("📝 You can now use 'example.hello_world' in your workflows!")
    
    click.echo("")
    click.echo("🎯 Next steps:")
    click.echo("   1. Test: workflow-gen generate list-plugins")
    if create_example:
        click.echo("   2. Try auto-install: workflow-gen generate --auto-install list-plugins")
        click.echo("   3. Create workflow: workflow-gen generate init")
    click.echo("   4. Install all dependencies: workflow-gen generate install-all-deps")


def _create_example_plugin(plugin_dir: Path):
    """Create an example plugin."""
    plugin_dir.mkdir(exist_ok=True)
    
    # Manifest
    manifest_content = """name: example
version: 1.0.0
description: Example plugin for demonstration and testing
author: Workflow Orchestrator Team

states:
  hello_world:
    description: Simple hello world demonstration
    inputs:
      message:
        type: string
        default: "Hello, World!"
        description: Message to process and display
    outputs:
      result:
        type: string
        description: Processed message result
  
  echo:
    description: Echo input message with timestamp
    inputs:
      text:
        type: string
        required: true
        description: Text to echo
    outputs:
      echoed:
        type: string
        description: Echoed text with timestamp

resources:
  cpu_units: 0.1
  memory_mb: 50
  network_weight: 0.0

dependencies: []
"""
    
    (plugin_dir / "manifest.yaml").write_text(manifest_content)
    
    # States implementation
    states_content = '''"""Example plugin states implementation."""

import time
from datetime import datetime
from plugins.base import PluginState
from core.agent.context import Context
from core.agent.state import StateResult


class HelloWorldState(PluginState):
    """Simple hello world state for testing and demonstration."""
    
    async def execute(self, context: Context) -> StateResult:
        """Execute hello world action."""
        message = self.config.get("message", "Hello, World!")
        
        # Log the message
        print(f"🌍 Example plugin says: {message}")
        
        # Store result in context
        result = f"Processed at {datetime.now().isoformat()}: {message}"
        context.set_output("result", result)
        context.set_state("example_executed", True)
        context.set_state("last_message", message)
        
        return "hello_complete"
    
    def validate_inputs(self, context: Context) -> None:
        """Validate inputs - message is optional so nothing to validate."""
        pass
    
    def validate_outputs(self, context: Context) -> None:
        """Validate outputs were set correctly."""
        result = context.get_output("result")
        if not result:
            raise ValueError("Result output was not set")


class EchoState(PluginState):
    """Echo state that returns input with timestamp."""
    
    async def execute(self, context: Context) -> StateResult:
        """Echo the input text with current timestamp."""
        text = self.config.get("text", "")
        
        if not text:
            raise ValueError("Text input is required for echo state")
        
        # Create echoed message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        echoed = f"[{timestamp}] Echo: {text}"
        
        print(f"🔊 {echoed}")
        
        # Store outputs
        context.set_output("echoed", echoed)
        context.set_state("echo_executed", True)
        
        return "echo_complete"
    
    def validate_inputs(self, context: Context) -> None:
        """Validate required inputs."""
        if not self.config.get("text"):
            raise ValueError("Text input is required")
'''
    
    (plugin_dir / "states.py").write_text(states_content)
    
    # Plugin class
    init_content = '''"""Example plugin for demonstration and testing."""

from pathlib import Path
from typing import Dict, Type

from plugins.base import Plugin, PluginState
from .states import HelloWorldState, EchoState


class ExamplePlugin(Plugin):
    """Example plugin implementation showing basic plugin structure."""
    
    def __init__(self):
        manifest_path = Path(__file__).parent / "manifest.yaml"
        super().__init__(manifest_path)
    
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register all example states."""
        return {
            "hello_world": HelloWorldState,
            "echo": EchoState,
        }
'''
    
    (plugin_dir / "__init__.py").write_text(init_content)


@generate.command()
def init():
    """Initialize a new workflow YAML template with plugin examples."""
    template_content = '''---
name: my_workflow
version: 1.0.0
description: "My awesome workflow with plugin examples"
author: "Your Name <your.email@example.com>"

config:
  timeout: 300
  max_concurrent: 5
  retry_policy:
    max_retries: 3
    initial_delay: 1.0
    exponential_base: 2.0
    jitter: true

environment:
  variables:
    LOG_LEVEL: INFO
  secrets:
    - API_KEY

# Integrations will be auto-discovered from state types
# But you can explicitly configure them if needed:
# integrations:
#   - name: gmail
#     version: "1.0.0"
#     config: {}

states:
  - name: start
    type: builtin.start
    description: "Workflow entry point"
    transitions:
      - on_success: greet
      
  - name: greet
    type: example.hello_world  # Example plugin state
    description: "Say hello using example plugin"
    config:
      message: "Hello from my workflow!"
    dependencies:
      - name: start
        type: required
    transitions:
      - on_success: process_data
      - on_failure: error_handler
      
  - name: process_data
    type: builtin.transform
    description: "Process some data with custom logic"
    function: |
      async def process(context):
          # Access previous state results
          greeting_result = context.get_output("result")
          
          # Your custom logic here
          processed_data = {
              "greeting": greeting_result,
              "processed_at": time.time(),
              "status": "completed"
          }
          
          context.set_state("processed_data", processed_data)
          return "data_processed"
    dependencies:
      - name: greet
        type: required
    transitions:
      - on_success: echo_result
      - on_failure: error_handler
      
  - name: echo_result
    type: example.echo  # Another example plugin state
    description: "Echo the processing result"
    config:
      text: "Data processing completed successfully!"
    dependencies:
      - name: process_data
        type: required
    transitions:
      - on_success: end
      - on_failure: error_handler
      
  - name: error_handler
    type: builtin.error_handler
    description: "Handle any errors that occur"
    config:
      log_level: ERROR
      notify: true
    transitions:
      - on_complete: end
      
  - name: end
    type: builtin.end
    description: "Workflow completion"

# Optional: Schedule the workflow
# schedule:
#   cron: "0 9 * * *"  # Daily at 9 AM
#   timezone: "UTC"
#   enabled: true
'''
    
    output_file = Path("workflow.yaml")
    if output_file.exists():
        click.echo("❌ workflow.yaml already exists")
        click.echo("💡 Use a different name or remove the existing file")
        sys.exit(1)
    
    output_file.write_text(template_content.strip())
    
    click.echo("✅ Created workflow.yaml template")
    click.echo("")
    click.echo("📝 Next steps:")
    click.echo("   1. Setup plugins: workflow-gen generate setup-plugins --create-example")
    click.echo("   2. Install dependencies: workflow-gen generate --auto-install validate workflow.yaml")
    click.echo("   3. Generate code: workflow-gen generate --auto-install yaml-to-code workflow.yaml")
    click.echo("")
    click.echo("💡 Use --auto-install flag to automatically install missing dependencies")


@generate.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
def export_requirements(yaml_file: Path, output: Optional[Path]):
    """Export requirements.txt based on workflow plugins."""
    try:
        click.echo(f"📋 Analyzing requirements for: {yaml_file}")
        
        spec = parse_workflow_file(yaml_file)
        
        # Base requirements
        requirements = [
            "asyncio",
            "pyyaml>=6.0",
            "pydantic>=1.8.0", 
            "structlog>=21.0.0",
            "click>=8.0.0"
        ]
        
        # Add plugin-specific requirements
        from plugins.registry import plugin_registry
        plugin_registry.load_plugins()
        
        plugin_requirements = set()
        for integration in spec.integrations:
            plugin = plugin_registry.get_plugin(integration.name)
            if plugin:
                plugin_requirements.update(plugin.manifest.dependencies)
        
        all_requirements = sorted(set(requirements + list(plugin_requirements)))
        
        # Output requirements
        if output:
            output.write_text('\n'.join(all_requirements) + '\n')
            click.echo(f"✅ Requirements exported to: {output}")
        else:
            click.echo("📦 Required packages:")
            for req in all_requirements:
                click.echo(f"   {req}")
        
    except Exception as e:
        click.echo(f"❌ Error analyzing requirements: {str(e)}")
        sys.exit(1)


@generate.command()
def clean():
    """Clean generated files and caches."""
    patterns_to_clean = [
        "generated-*",
        "*.pyc",
        "__pycache__",
        ".pytest_cache",
        "*.egg-info"
    ]
    
    cleaned = []
    for pattern in patterns_to_clean:
        for path in Path.cwd().glob(pattern):
            if path.is_file():
                path.unlink()
                cleaned.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                cleaned.append(str(path))
    
    if cleaned:
        click.echo(f"🧹 Cleaned {len(cleaned)} items:")
        for item in cleaned:
            click.echo(f"   {item}")
    else:
        click.echo("✨ Nothing to clean")


@generate.command()
@click.pass_context
def doctor(ctx):
    """Diagnose common issues with the workflow system."""
    verbose = ctx.obj.get('verbose', False)
    
    click.echo("🔍 Running system diagnostics...")
    
    issues = []
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"❌ Python {python_version.major}.{python_version.minor} is too old. Requires Python 3.8+")
    else:
        click.echo(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required modules
    required_modules = ['yaml', 'click', 'jinja2', 'pathlib']
    for module in required_modules:
        try:
            __import__(module)
            if verbose:
                click.echo(f"✅ {module} module available")
        except ImportError:
            issues.append(f"❌ Required module '{module}' not found")
    
    # Check plugins directory
    plugins_dir = Path.cwd() / "plugins"
    if plugins_dir.exists():
        click.echo(f"✅ Plugins directory found: {plugins_dir}")
        
        # Count plugins
        plugin_count = 0
        for subdir in plugins_dir.iterdir():
            if subdir.is_dir() and (subdir / "manifest.yaml").exists():
                plugin_count += 1
        
        click.echo(f"📦 Found {plugin_count} plugins")
        
        if verbose and plugin_count > 0:
            click.echo("   Plugins:")
            for subdir in plugins_dir.iterdir():
                if subdir.is_dir() and (subdir / "manifest.yaml").exists():
                    click.echo(f"     • {subdir.name}")
    else:
        issues.append(f"⚠️  No plugins directory found at {plugins_dir}")
    
    # Check plugin registry
    try:
        from plugins.registry import plugin_registry
        plugin_registry.load_plugins()
        available_plugins = plugin_registry.list_plugins()
        loaded_count = sum(1 for p in available_plugins if p.get('status') == 'loaded')
        total_count = len(available_plugins)
        
        click.echo(f"✅ Plugin registry: {loaded_count}/{total_count} plugins loaded")
        
        if loaded_count < total_count:
            failed_count = total_count - loaded_count
            click.echo(f"⚠️  {failed_count} plugins failed to load (likely missing dependencies)")
            click.echo("💡 Try: workflow-gen generate --auto-install list-plugins")
        
    except Exception as e:
        issues.append(f"❌ Plugin registry error: {e}")
    
    # Check core modules
    try:
        from core.generator.parser import WorkflowParser
        from core.generator.engine import CodeGenerator
        click.echo("✅ Core generator modules available")
    except ImportError as e:
        issues.append(f"❌ Core module import error: {e}")
    
    # Test pip functionality
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            click.echo("✅ pip is available for dependency installation")
        else:
            issues.append("❌ pip is not working correctly")
    except Exception:
        issues.append("❌ pip is not available")
    
    # Summary
    click.echo("")
    if issues:
        click.echo(f"🚨 Found {len(issues)} issues:")
        for issue in issues:
            click.echo(f"   {issue}")
        
        click.echo("")
        click.echo("🔧 Suggested fixes:")
        click.echo("   1. Install requirements: pip install -r requirements.txt")
        click.echo("   2. Setup plugins: workflow-gen generate setup-plugins --create-example")
        click.echo("   3. Auto-install dependencies: workflow-gen generate --auto-install list-plugins")
        click.echo("   4. Check Python version (requires 3.8+)")
    else:
        click.echo("✅ All checks passed! System is ready.")
        click.echo("💡 Try: workflow-gen generate --auto-install init")


if __name__ == '__main__':
    generate()