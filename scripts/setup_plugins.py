# scripts/setup_plugins.py
"""Setup script to create basic plugin structure."""

from pathlib import Path
import shutil


def setup_plugins_directory():
    """Setup the plugins directory with basic structure."""
    
    # Get project root (assuming script is in scripts/)
    project_root = Path(__file__).parent.parent
    plugins_dir = project_root / "plugins"
    
    # Create plugins directory
    plugins_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_file = plugins_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Plugins directory."""\n')
    
    # Create basic structure
    (plugins_dir / "contrib").mkdir(exist_ok=True)
    (plugins_dir / "contrib" / "__init__.py").write_text('"""Contributed plugins."""\n')
    
    print(f"✅ Plugin directory structure created at: {plugins_dir}")
    
    # Create example plugin if it doesn't exist
    example_dir = plugins_dir / "example"
    if not example_dir.exists():
        create_example_plugin(example_dir)
        print(f"✅ Example plugin created at: {example_dir}")


def create_example_plugin(plugin_dir: Path):
    """Create an example plugin."""
    plugin_dir.mkdir(exist_ok=True)
    
    # Manifest
    manifest_content = """name: example
version: 1.0.0
description: Example plugin for demonstration
author: Workflow Orchestrator Team

states:
  hello_world:
    description: Simple hello world demonstration
    inputs:
      message:
        type: string
        default: "Hello, World!"
        description: Message to process
    outputs:
      result:
        type: string
        description: Processed message

resources:
  cpu_units: 0.1
  memory_mb: 50

dependencies: []
"""
    
    (plugin_dir / "manifest.yaml").write_text(manifest_content)
    
    # States
    states_content = '''"""Example plugin states."""

from plugins.base import PluginState
from core.agent.context import Context
from core.agent.state import StateResult


class HelloWorldState(PluginState):
    """Simple hello world state for testing."""
    
    async def execute(self, context: Context) -> StateResult:
        """Execute hello world action."""
        message = self.config.get("message", "Hello, World!")
        
        print(f"Example plugin says: {message}")
        
        # Store result
        context.set_output("result", f"Processed: {message}")
        context.set_state("example_executed", True)
        
        return "hello_complete"
    
    def validate_inputs(self, context: Context) -> None:
        """Validate inputs."""
        # Message is optional, so no validation needed
        pass
'''
    
    (plugin_dir / "states.py").write_text(states_content)
    
    # Plugin class
    init_content = '''"""Example plugin."""

from pathlib import Path
from typing import Dict, Type

from plugins.base import Plugin, PluginState
from .states import HelloWorldState


class ExamplePlugin(Plugin):
    """Example plugin for demonstration."""
    
    def __init__(self):
        manifest_path = Path(__file__).parent / "manifest.yaml"
        super().__init__(manifest_path)
    
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register all example states."""
        return {
            "hello_world": HelloWorldState,
        }
'''
    
    (plugin_dir / "__init__.py").write_text(init_content)


if __name__ == "__main__":
    setup_plugins_directory()