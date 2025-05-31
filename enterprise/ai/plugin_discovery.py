# enterprise/ai/plugin_discovery.py
"""Plugin discovery system for AI copilot."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import importlib.util
import inspect

from plugins.base import Plugin, PluginState
from core.agent.state import StateResult
from core.agent.context import Context


class PluginDiscovery:
    """Discover and analyze available plugins."""
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        """Initialize plugin discovery.
        
        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent.parent / "plugins"
        self._plugins_cache: Dict[str, Dict[str, Any]] = {}
    
    def discover_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available plugins and their capabilities.
        
        Returns:
            Dictionary mapping plugin names to their metadata and capabilities
        """
        if self._plugins_cache:
            return self._plugins_cache
        
        plugins = {}
        
        # Scan built-in plugins
        builtin_dir = self.plugin_dir / "contrib"
        if builtin_dir.exists():
            for plugin_path in builtin_dir.iterdir():
                if plugin_path.is_dir() and (plugin_path / "manifest.yaml").exists():
                    plugin_info = self._analyze_plugin(plugin_path)
                    if plugin_info:
                        plugins[plugin_info["name"]] = plugin_info
        
        # Scan custom plugins
        for plugin_path in self.plugin_dir.iterdir():
            if (plugin_path.is_dir() and 
                plugin_path.name != "contrib" and 
                (plugin_path / "manifest.yaml").exists()):
                plugin_info = self._analyze_plugin(plugin_path)
                if plugin_info:
                    plugins[plugin_info["name"]] = plugin_info
        
        self._plugins_cache = plugins
        return plugins
    
    def _analyze_plugin(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single plugin.
        
        Args:
            plugin_path: Path to plugin directory
            
        Returns:
            Plugin information dictionary
        """
        try:
            # Load manifest
            with open(plugin_path / "manifest.yaml", 'r') as f:
                manifest = yaml.safe_load(f)
            
            plugin_info = {
                "name": manifest["name"],
                "version": manifest.get("version", "1.0.0"),
                "description": manifest.get("description", ""),
                "author": manifest.get("author", ""),
                "states": {},
                "inputs": manifest.get("inputs", {}),
                "outputs": manifest.get("outputs", {}),
                "resources": manifest.get("resources", {}),
                "dependencies": manifest.get("dependencies", [])
            }
            
            # Analyze states
            states = manifest.get("states", {})
            for state_name, state_config in states.items():
                state_info = {
                    "description": state_config.get("description", ""),
                    "inputs": state_config.get("inputs", {}),
                    "outputs": state_config.get("outputs", {}),
                    "resources": state_config.get("resources", {}),
                    "examples": state_config.get("examples", [])
                }
                
                # Try to load the actual state class for more info
                try:
                    module_path = plugin_path / "__init__.py"
                    if not module_path.exists():
                        module_path = plugin_path / "states.py"
                    
                    if module_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            f"plugins.{manifest['name']}", 
                            module_path
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find state classes
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (inspect.isclass(attr) and 
                                    issubclass(attr, PluginState) and 
                                    attr != PluginState):
                                    # Extract docstring
                                    if attr.__doc__:
                                        state_info["docstring"] = attr.__doc__.strip()
                except Exception:
                    pass  # Continue without runtime info
                
                plugin_info["states"][state_name] = state_info
            
            return plugin_info
            
        except Exception as e:
            print(f"Error analyzing plugin {plugin_path}: {e}")
            return None
    
    def get_plugin_capabilities_prompt(self) -> str:
        """Generate a formatted prompt describing all plugin capabilities.
        
        Returns:
            Formatted string for AI prompts
        """
        plugins = self.discover_all_plugins()
        
        capabilities = []
        
        for plugin_name, plugin_info in plugins.items():
            plugin_section = [
                f"Plugin: {plugin_name}",
                f"  Description: {plugin_info['description']}",
                f"  Version: {plugin_info['version']}",
                "  States:"
            ]
            
            for state_name, state_info in plugin_info["states"].items():
                full_state_name = f"{plugin_name}.{state_name}"
                state_desc = state_info.get("description", "No description")
                plugin_section.append(f"    - {full_state_name}: {state_desc}")
                
                # Add inputs
                if state_info.get("inputs"):
                    plugin_section.append("      Inputs:")
                    for input_name, input_config in state_info["inputs"].items():
                        input_type = input_config.get("type", "any")
                        required = input_config.get("required", False)
                        desc = input_config.get("description", "")
                        default = input_config.get("default")
                        
                        req_text = "required" if required else "optional"
                        default_text = f", default: {default}" if default is not None else ""
                        
                        plugin_section.append(
                            f"        - {input_name} ({input_type}, {req_text}{default_text}): {desc}"
                        )
                
                # Add outputs
                if state_info.get("outputs"):
                    plugin_section.append("      Outputs:")
                    for output_name, output_config in state_info["outputs"].items():
                        output_type = output_config.get("type", "any")
                        desc = output_config.get("description", "")
                        plugin_section.append(
                            f"        - {output_name} ({output_type}): {desc}"
                        )
                
                # Add examples if available
                if state_info.get("examples"):
                    plugin_section.append("      Examples:")
                    for example in state_info["examples"][:2]:  # Limit to 2 examples
                        plugin_section.append(f"        - {example}")
            
            capabilities.append("\n".join(plugin_section))
        
        return "\n\n".join(capabilities)
    
if __name__ == "__main__":
    discovery = PluginDiscovery()
    print("=== Plugin Capabilities ===")
    print(discovery.get_plugin_capabilities_prompt())
