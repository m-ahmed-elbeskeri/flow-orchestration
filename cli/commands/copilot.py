# cli/commands/copilot.py
"""AI Copilot CLI commands."""

import asyncio
import yaml
import functools
from pathlib import Path
import click

from enterprise.ai.flow_copilot import FlowCopilot


def async_command(f):
    """Decorator to run async functions with Click."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.group()
def copilot():
    """AI-powered workflow generation commands."""
    pass


@copilot.command()
@click.argument('request', nargs=-1, required=True)
@click.option('--model', default='anthropic/claude-3-sonnet', help='AI model to use')
@click.option('--output', '-o', help='Output file for workflow definition')
@click.option('--interactive/--no-interactive', default=True, help='Interactive mode')
@async_command
async def generate(request, model, output, interactive):
    """Generate a workflow from natural language request."""
    request_text = ' '.join(request)
    
    click.echo(f"🤖 Analyzing request: {request_text}")
    
    try:
        # Initialize copilot
        copilot_instance = FlowCopilot(model=model)
        
        # Analyze request
        click.echo("📊 Running analysis...")
        analysis = await copilot_instance.analyze_request(request_text)
        
        # Show analysis results
        click.echo(f"✅ Analysis complete!")
        click.echo(f"   Complexity: {analysis.complexity}")
        click.echo(f"   Clear enough: {'Yes' if analysis.clear_enough else 'No'}")
        
        if analysis.suggested_plugins:
            click.echo(f"   Suggested plugins: {', '.join(analysis.suggested_plugins)}")
        
        # Interactive clarifications
        clarifications = {}
        if interactive and not analysis.clear_enough:
            click.echo("\n❓ Clarifications needed:")
            for question in analysis.clarification_questions:
                click.echo(f"   • {question}")
            
            click.echo("\nPlease answer the following questions:")
            clarifications = await copilot_instance.ask_clarifications(analysis)
        
        # Generate workflow
        click.echo("\n🔨 Generating workflow...")
        result = await copilot_instance.generate_workflow(request_text, clarifications, analysis)
        
        if result.get("error"):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        # Save workflow
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(result['workflow_definition'], f, default_flow_style=False)
            
            click.echo(f"✅ Workflow saved to: {output_path}")
            
            # Save additional artifacts
            if result.get('python_code'):
                py_path = output_path.with_suffix('.py')
                with open(py_path, 'w') as f:
                    f.write(result['python_code'])
                click.echo(f"📄 Python code saved to: {py_path}")
            
            if result.get('mermaid_diagram'):
                mmd_path = output_path.with_suffix('.mmd')
                with open(mmd_path, 'w') as f:
                    f.write(result['mermaid_diagram'])
                click.echo(f"📊 Diagram saved to: {mmd_path}")
            
            if result.get('env_template'):
                env_path = output_path.parent / '.env.template'
                with open(env_path, 'w') as f:
                    f.write(result['env_template'])
                click.echo(f"🔐 Environment template saved to: {env_path}")
        else:
            # Display workflow
            click.echo("\n📋 Generated Workflow:")
            click.echo(yaml.dump(result['workflow_definition'], default_flow_style=False))
        
        # Display detected variables
        if result.get('detected_variables'):
            vars_info = result['detected_variables']
            if any(vars_info.values()):
                click.echo("\n🔍 Detected Variables:")
                for var_type, var_list in vars_info.items():
                    if var_list:
                        click.echo(f"  {var_type}: {', '.join(var_list)}")
    
    except ImportError as e:
        click.echo(f"❌ Missing dependencies: {e}", err=True)
        click.echo("Install enterprise AI dependencies to use this feature.")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@copilot.command()
@click.option('--format', type=click.Choice(['yaml', 'json', 'table']), default='yaml')
def list_plugins(format):
    """List all available plugins and their capabilities."""
    try:
        from enterprise.ai.plugin_discovery import PluginDiscovery
        
        discovery = PluginDiscovery()
        plugins = discovery.discover_all_plugins()
        
        if format == 'yaml':
            click.echo(yaml.dump(plugins, default_flow_style=False))
        elif format == 'json':
            import json
            click.echo(json.dumps(plugins, indent=2))
        else:  # table
            for name, info in plugins.items():
                click.echo(f"\n📦 {name} (v{info['version']})")
                click.echo(f"   {info['description']}")
                if info['states']:
                    click.echo("   States:")
                    for state_name, state_info in info['states'].items():
                        click.echo(f"     - {state_name}: {state_info['description']}")
    
    except ImportError:
        click.echo("❌ Plugin discovery system not available. Install enterprise dependencies.")


@copilot.command()
@click.argument('request')
@click.option('--dry-run', is_flag=True, help='Show what would be generated without calling AI')
def analyze(request, dry_run):
    """Analyze a request without generating the full workflow."""
    click.echo(f"🔍 Analyzing: {request}")
    
    if dry_run:
        click.echo("📝 Dry run mode - would analyze:")
        click.echo(f"   Request: {request}")
        click.echo("   Would detect variables, suggest plugins, assess complexity")
        return
    
    click.echo("💡 Use 'copilot generate' for full workflow generation")


@copilot.command()
def test_connection():
    """Test the AI API connection."""
    click.echo("🔌 Testing AI API connection...")
    
    try:
        from enterprise.ai.openrouter_api import OpenRouterAPI
        
        api = OpenRouterAPI()
        click.echo("✅ OpenRouter API client initialized successfully")
        click.echo(f"   Base URL: {api.base_url}")
        click.echo("   API Key: [REDACTED]" if api.api_key else "❌ No API key found")
        
        if not api.api_key:
            click.echo("\n💡 Set OPENROUTER_KEY environment variable to enable AI features")
    
    except ImportError:
        click.echo("❌ OpenRouter API not available. Install enterprise dependencies.")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@copilot.command()
def version():
    """Show copilot version and capabilities."""
    click.echo("🤖 AI Copilot for Workflow Orchestrator")
    click.echo("   Version: 1.0.0")
    
    # Check what's available
    features = []
    
    try:
        from enterprise.ai.openrouter_api import OpenRouterAPI
        features.append("✅ OpenRouter AI API")
    except ImportError:
        features.append("❌ OpenRouter AI API")
    
    try:
        from enterprise.ai.plugin_discovery import PluginDiscovery
        features.append("✅ Plugin Discovery")
    except ImportError:
        features.append("❌ Plugin Discovery")
    
    try:
        from enterprise.ai.flow_copilot import FlowCopilot
        features.append("✅ Flow Generation")
    except ImportError:
        features.append("❌ Flow Generation")
    
    click.echo("\n🔧 Available Features:")
    for feature in features:
        click.echo(f"   {feature}")
    
    if "❌" in '\n'.join(features):
        click.echo("\n💡 Install enterprise dependencies for full AI capabilities")