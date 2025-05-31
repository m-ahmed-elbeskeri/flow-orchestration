# core/generator/discovery.py
"""Enhanced plugin discovery system."""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import yaml
import importlib.util
import inspect
import logging

from plugins.base import Plugin, PluginState
from plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Discover and analyze available plugins for code generation."""
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """Initialize plugin discovery.
        
        Args:
            plugin_dirs: Additional directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or []
        self._capabilities_cache: Optional[str] = None
        self._plugins_cache: Optional[Dict[str, Dict[str, Any]]] = None
    
    def discover_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available plugins and their capabilities.
        
        Returns:
            Dictionary mapping plugin names to their metadata and capabilities
        """
        if self._plugins_cache:
            return self._plugins_cache
        
        # Ensure plugins are loaded
        if self.plugin_dirs:
            plugin_registry.load_plugins(self.plugin_dirs)
        else:
            plugin_registry.load_plugins()
        
        plugins = {}
        
        # Get plugin information from registry
        plugin_list = plugin_registry.list_plugins()
        
        for plugin_info in plugin_list:
            plugin_name = plugin_info['name']
            
            # Get additional detailed information
            try:
                plugin = plugin_registry.get_plugin(plugin_name)
                if plugin:
                    detailed_info = self._analyze_plugin_detailed(plugin)
                    # Merge basic info with detailed analysis
                    combined_info = {**plugin_info, **detailed_info}
                    plugins[plugin_name] = combined_info
                else:
                    plugins[plugin_name] = plugin_info
            except Exception as e:
                logger.error(f"Error analyzing plugin {plugin_name}: {e}")
                plugins[plugin_name] = plugin_info
        
        self._plugins_cache = plugins
        return plugins
    
    def _analyze_plugin_detailed(self, plugin: Plugin) -> Dict[str, Any]:
        """Perform detailed analysis of a plugin.
        
        Args:
            plugin: Plugin instance to analyze
            
        Returns:
            Detailed plugin information
        """
        detailed_info = {
            "states_detailed": {},
            "capabilities": [],
            "integration_type": self._determine_integration_type(plugin),
            "config_schema": {},
            "examples": []
        }
        
        try:
            # Analyze each state in detail
            states = plugin.register_states()
            
            for state_name, state_class in states.items():
                state_info = self._analyze_state_class(state_class, plugin.manifest.name, state_name)
                detailed_info["states_detailed"][state_name] = state_info
                
                # Extract capabilities
                capability = f"{plugin.manifest.name}.{state_name}"
                if hasattr(state_class, '__doc__') and state_class.__doc__:
                    capability += f": {state_class.__doc__.strip().split('.')[0]}"
                detailed_info["capabilities"].append(capability)
        
        except Exception as e:
            logger.error(f"Error analyzing states for plugin {plugin.manifest.name}: {e}")
        
        return detailed_info
    
    def _analyze_state_class(self, state_class: type, plugin_name: str, state_name: str) -> Dict[str, Any]:
        """Analyze a state class for detailed information.
        
        Args:
            state_class: State class to analyze
            plugin_name: Name of the plugin
            state_name: Name of the state
            
        Returns:
            Detailed state information
        """
        state_info = {
            "class_name": state_class.__name__,
            "docstring": "",
            "methods": [],
            "parameters": [],
            "examples": [],
            "error_handling": []
        }
        
        # Extract docstring
        if hasattr(state_class, '__doc__') and state_class.__doc__:
            state_info["docstring"] = state_class.__doc__.strip()
        
        # Analyze methods
        for method_name in dir(state_class):
            if not method_name.startswith('_') and callable(getattr(state_class, method_name)):
                method = getattr(state_class, method_name)
                if hasattr(method, '__doc__') and method.__doc__:
                    state_info["methods"].append({
                        "name": method_name,
                        "doc": method.__doc__.strip()
                    })
        
        # Extract parameters from __init__ if available
        try:
            init_signature = inspect.signature(state_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name not in ['self', 'config']:
                    state_info["parameters"].append({
                        "name": param_name,
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    })
        except Exception:
            pass
        
        return state_info
    
    def _determine_integration_type(self, plugin: Plugin) -> str:
        """Determine the type of integration.
        
        Args:
            plugin: Plugin to analyze
            
        Returns:
            Integration type string
        """
        plugin_name = plugin.manifest.name.lower()
        
        if 'email' in plugin_name or 'gmail' in plugin_name or 'smtp' in plugin_name:
            return "communication"
        elif 'slack' in plugin_name or 'teams' in plugin_name or 'discord' in plugin_name:
            return "messaging"
        elif 'database' in plugin_name or 'sql' in plugin_name or 'mongo' in plugin_name:
            return "storage"
        elif 'http' in plugin_name or 'api' in plugin_name or 'rest' in plugin_name:
            return "api"
        elif 'file' in plugin_name or 'storage' in plugin_name or 's3' in plugin_name:
            return "file_storage"
        else:
            return "utility"
    
    def get_plugin_capabilities_prompt(self) -> str:
        """Generate a formatted prompt describing all plugin capabilities.
        
        Returns:
            Formatted string for AI prompts
        """
        if self._capabilities_cache:
            return self._capabilities_cache
        
        plugins = self.discover_all_plugins()
        
        if not plugins:
            return "No plugins available. Only built-in states (builtin.start, builtin.end, builtin.transform, builtin.conditional, builtin.delay, builtin.error_handler) can be used."
        
        capabilities = ["Available workflow integrations and states:"]
        
        # Group by integration type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for plugin_name, plugin_info in plugins.items():
            integration_type = plugin_info.get('integration_type', 'utility')
            if integration_type not in by_type:
                by_type[integration_type] = []
            by_type[integration_type].append(plugin_info)
        
        for integration_type, type_plugins in by_type.items():
            capabilities.append(f"\n{integration_type.upper()} INTEGRATIONS:")
            
            for plugin_info in type_plugins:
                plugin_name = plugin_info['name']
                capabilities.append(f"\n  📦 {plugin_name} v{plugin_info['version']}")
                capabilities.append(f"     {plugin_info['description']}")
                
                # List states with detailed info
                states_detailed = plugin_info.get('states_detailed', {})
                regular_states = plugin_info.get('states', [])
                
                if states_detailed:
                    capabilities.append("     States:")
                    for state_name, state_info in states_detailed.items():
                        full_name = f"{plugin_name}.{state_name}"
                        doc = state_info.get('docstring', 'No description available')
                        # Take first sentence of docstring
                        short_doc = doc.split('.')[0] if doc else 'No description'
                        capabilities.append(f"       • {full_name}: {short_doc}")
                elif regular_states:
                    capabilities.append(f"     States: {', '.join(f'{plugin_name}.{s}' for s in regular_states)}")
                
                # Add dependencies if any
                if plugin_info.get('dependencies'):
                    deps = plugin_info['dependencies'][:3]  # Show first 3
                    capabilities.append(f"     Dependencies: {', '.join(deps)}")
        
        # Add built-in states
        capabilities.extend([
            "\nBUILT-IN STATES:",
            "  • builtin.start: Workflow entry point",
            "  • builtin.end: Workflow exit point", 
            "  • builtin.transform: Custom data transformation with Python code",
            "  • builtin.conditional: Conditional branching based on expressions",
            "  • builtin.delay: Add delays to workflow execution",
            "  • builtin.error_handler: Handle errors and exceptions"
        ])
        
        self._capabilities_cache = "\n".join(capabilities)
        return self._capabilities_cache
    
    def get_integration_suggestions(self, description: str) -> List[str]:
        """Get integration suggestions based on workflow description.
        
        Args:
            description: Workflow description
            
        Returns:
            List of suggested integration state types
        """
        description_lower = description.lower()
        plugins = self.discover_all_plugins()
        suggestions = []
        
        # Keyword mapping
        keywords = {
            'email': ['gmail', 'smtp', 'email'],
            'slack': ['slack'],
            'database': ['database', 'sql', 'mongo'],
            'http': ['http', 'api', 'rest', 'webhook'],
            'file': ['file', 'storage', 's3', 'upload', 'download']
        }
        
        for keyword, plugin_names in keywords.items():
            if keyword in description_lower:
                for plugin_name in plugin_names:
                    if plugin_name in plugins:
                        plugin_info = plugins[plugin_name]
                        for state_name in plugin_info.get('states', []):
                            suggestions.append(f"{plugin_name}.{state_name}")
        
        return suggestions
    
    def clear_cache(self):
        """Clear cached data."""
        self._capabilities_cache = None
        self._plugins_cache = None


# Global discovery instance
plugin_discovery = PluginDiscovery()