from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NodeAppearance:
    icon: str
    color: str
    category: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    ui_config: Dict[str, Any]

class NodeRegistry:
    def __init__(self):
        self._node_appearances: Dict[str, NodeAppearance] = {}
        self._load_builtin_nodes()
        self._load_plugin_nodes()
    
    def _load_builtin_nodes(self):
        """Load default node appearances for builtin types"""
        builtin_nodes = {
            "builtin.start": NodeAppearance(
                icon="play-circle",
                color="#10b981",
                category="control",
                description="Workflow entry point",
                inputs={},
                outputs={"next": {"type": "signal", "description": "Next state signal"}},
                ui_config={"shape": "circle", "size": "small", "hideable": True}
            ),
            "builtin.end": NodeAppearance(
                icon="check-circle",
                color="#ef4444",
                category="control", 
                description="Workflow completion",
                inputs={"prev": {"type": "signal", "description": "Previous state signal"}},
                outputs={},
                ui_config={"shape": "circle", "size": "small", "hideable": True}
            ),
            "builtin.conditional": NodeAppearance(
                icon="git-branch",
                color="#f59e0b",
                category="logic",
                description="Conditional branching",
                inputs={
                    "condition": {
                        "type": "boolean", 
                        "required": True,
                        "description": "Condition to evaluate"
                    }
                },
                outputs={
                    "true": {"type": "signal", "description": "True path"},
                    "false": {"type": "signal", "description": "False path"}
                },
                ui_config={"shape": "diamond", "size": "medium"}
            ),
            "builtin.transform": NodeAppearance(
                icon="zap",
                color="#8b5cf6",
                category="processing",
                description="Data transformation",
                inputs={
                    "data": {
                        "type": "any", 
                        "required": False,
                        "description": "Input data to transform"
                    }
                },
                outputs={
                    "result": {"type": "any", "description": "Transformed data"}
                },
                ui_config={"shape": "rectangle", "size": "medium"}
            ),
            "builtin.delay": NodeAppearance(
                icon="clock",
                color="#6b7280",
                category="control",
                description="Add delay/wait",
                inputs={
                    "seconds": {
                        "type": "number",
                        "required": True,
                        "default": 1,
                        "description": "Delay in seconds"
                    }
                },
                outputs={
                    "completed": {"type": "signal", "description": "Delay completed"}
                },
                ui_config={"shape": "rectangle", "size": "small"}
            ),
            "builtin.error_handler": NodeAppearance(
                icon="alert-triangle",
                color="#dc2626",
                category="control",
                description="Error handling",
                inputs={
                    "error": {
                        "type": "error",
                        "required": False,
                        "description": "Error to handle"
                    }
                },
                outputs={
                    "handled": {"type": "signal", "description": "Error handled"}
                },
                ui_config={"shape": "rectangle", "size": "medium"}
            )
        }
        self._node_appearances.update(builtin_nodes)
    
    def _load_plugin_nodes(self):
        """Load node appearances from plugin manifests"""
        plugins_dir = Path(__file__).parent
        
        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith('.'):
                continue
                
            # Skip non-plugin directories
            if plugin_dir.name in ['__pycache__', 'node_registry.py']:
                continue
                
            manifest_file = plugin_dir / "manifest.yaml"
            if not manifest_file.exists():
                continue
            
            try:
                with open(manifest_file) as f:
                    manifest = yaml.safe_load(f)
                
                plugin_name = manifest.get('name')
                if not plugin_name:
                    continue
                    
                states = manifest.get('states', {})
                
                for state_name, state_config in states.items():
                    node_type = f"{plugin_name}.{state_name}"
                    
                    # Get UI config from manifest
                    ui_config = state_config.get('ui', {})
                    
                    # Default icon based on plugin type
                    default_icons = {
                        'gmail': 'mail',
                        'slack': 'message-circle',
                        'webhook': 'globe',
                        'openai': 'brain',
                        'database': 'database',
                        'file': 'file',
                        'http': 'globe'
                    }
                    
                    default_icon = default_icons.get(plugin_name, 'box')
                    
                    appearance = NodeAppearance(
                        icon=ui_config.get('icon', default_icon),
                        color=ui_config.get('color', '#6b7280'),
                        category=ui_config.get('category', plugin_name),
                        description=state_config.get('description', ''),
                        inputs=state_config.get('inputs', {}),
                        outputs=state_config.get('outputs', {}),
                        ui_config={
                            **ui_config,
                            'shape': ui_config.get('shape', 'rectangle'),
                            'size': ui_config.get('size', 'medium')
                        }
                    )
                    
                    self._node_appearances[node_type] = appearance
                    
            except Exception as e:
                logger.warning(f"Failed to load node appearances from {plugin_dir}: {e}")
    
    def get_node_appearance(self, node_type: str) -> Optional[NodeAppearance]:
        """Get appearance for a specific node type"""
        return self._node_appearances.get(node_type)
    
    def get_available_nodes(self) -> Dict[str, NodeAppearance]:
        """Get all available node types and their appearances"""
        return self._node_appearances.copy()
    
    def get_nodes_by_category(self) -> Dict[str, List[str]]:
        """Get nodes grouped by category"""
        categories = {}
        for node_type, appearance in self._node_appearances.items():
            category = appearance.category
            if category not in categories:
                categories[category] = []
            categories[category].append(node_type)
        return categories
    
    def refresh(self):
        """Refresh the registry by reloading all nodes"""
        self._node_appearances.clear()
        self._load_builtin_nodes()
        self._load_plugin_nodes()

# Global registry instance
node_registry = NodeRegistry()