# core/visual/engine.py
"""Visual workflow editor engine with bidirectional conversion."""

import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from core.visual.nodes import NodeLibrary, NodeType
from core.visual.flow import FlowDefinition, FlowNode, FlowEdge
from generator.core import WorkflowSpec, CodeGenerator


@dataclass
class VisualWorkflow:
    """Visual workflow representation."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    nodes: List[FlowNode] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)
    viewport: Dict[str, Any] = field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class VisualWorkflowEngine:
    """Core engine for visual workflow editing and conversion."""
    
    def __init__(self):
        self.node_library = NodeLibrary()
        self.code_generator = None
        self._workflows: Dict[str, VisualWorkflow] = {}
    
    def create_workflow(self, name: str, description: str = "") -> str:
        """Create a new visual workflow."""
        workflow_id = str(uuid.uuid4())
        
        workflow = VisualWorkflow(
            id=workflow_id,
            name=name,
            description=description
        )
        
        # Add default start and end nodes
        start_node = FlowNode(
            id="start",
            type="builtin.start",
            position={"x": 100, "y": 100},
            data={"label": "Start"}
        )
        
        end_node = FlowNode(
            id="end", 
            type="builtin.end",
            position={"x": 500, "y": 100},
            data={"label": "End"}
        )
        
        workflow.nodes = [start_node, end_node]
        self._workflows[workflow_id] = workflow
        
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[VisualWorkflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> None:
        """Update workflow with changes."""
        if workflow_id in self._workflows:
            workflow = self._workflows[workflow_id]
            
            if 'nodes' in updates:
                workflow.nodes = [FlowNode(**node) for node in updates['nodes']]
            
            if 'edges' in updates:
                workflow.edges = [FlowEdge(**edge) for edge in updates['edges']]
            
            if 'viewport' in updates:
                workflow.viewport = updates['viewport']
            
            workflow.updated_at = datetime.utcnow()
    
    def visual_to_yaml(self, workflow_id: str) -> str:
        """Convert visual workflow to YAML."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Build YAML structure
        yaml_data = {
            'name': workflow.name,
            'version': workflow.version,
            'description': workflow.description,
            'author': workflow.author,
            'config': {
                'timeout': 300,
                'max_concurrent': 10,
                'retry_policy': {
                    'max_retries': 3,
                    'initial_delay': 1.0,
                    'exponential_base': 2.0
                }
            },
            'environment': {
                'variables': {},
                'secrets': []
            },
            'integrations': [],
            'states': []
        }
        
        # Extract integrations from nodes
        integrations = set()
        for node in workflow.nodes:
            if '.' in node.type and not node.type.startswith('builtin.'):
                integration_name = node.type.split('.')[0]
                integrations.add(integration_name)
        
        yaml_data['integrations'] = [
            {'name': name, 'version': '1.0.0'} 
            for name in sorted(integrations)
        ]
        
        # Convert nodes to states
        node_map = {node.id: node for node in workflow.nodes}
        
        for node in workflow.nodes:
            state = {
                'name': node.id,
                'type': node.type,
                'description': node.data.get('description', ''),
            }
            
            # Add configuration
            if node.data.get('config'):
                state['config'] = node.data['config']
            
            # Add resources if specified
            if node.data.get('resources'):
                state['resources'] = node.data['resources']
            
            # Find dependencies from incoming edges
            dependencies = []
            for edge in workflow.edges:
                if edge.target == node.id:
                    dependencies.append(edge.source)
            
            if dependencies:
                state['dependencies'] = dependencies
            
            # Find transitions from outgoing edges
            transitions = []
            for edge in workflow.edges:
                if edge.source == node.id:
                    target_node = node_map.get(edge.target)
                    if target_node:
                        transition_type = edge.data.get('type', 'on_success')
                        transitions.append({
                            transition_type: edge.target
                        })
            
            if transitions:
                state['transitions'] = transitions
            
            yaml_data['states'].append(state)
        
        return yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def yaml_to_visual(self, yaml_content: str) -> str:
        """Convert YAML to visual workflow."""
        data = yaml.safe_load(yaml_content)
        
        # Create new workflow
        workflow_id = self.create_workflow(
            name=data.get('name', 'Untitled'),
            description=data.get('description', '')
        )
        
        workflow = self.get_workflow(workflow_id)
        workflow.version = data.get('version', '1.0.0')
        workflow.author = data.get('author', '')
        
        # Clear default nodes
        workflow.nodes = []
        workflow.edges = []
        
        # Convert states to nodes
        states = data.get('states', [])
        node_positions = self._calculate_node_positions(states)
        
        for i, state in enumerate(states):
            node = FlowNode(
                id=state['name'],
                type=state['type'],
                position=node_positions[i],
                data={
                    'label': state['name'],
                    'description': state.get('description', ''),
                    'config': state.get('config', {}),
                    'resources': state.get('resources', {})
                }
            )
            workflow.nodes.append(node)
        
        # Convert dependencies and transitions to edges
        for state in states:
            state_name = state['name']
            
            # Handle transitions
            for transition in state.get('transitions', []):
                for transition_type, target_state in transition.items():
                    edge = FlowEdge(
                        id=f"{state_name}-{target_state}",
                        source=state_name,
                        target=target_state,
                        data={'type': transition_type}
                    )
                    workflow.edges.append(edge)
        
        return workflow_id
    
    def visual_to_code(self, workflow_id: str, output_dir: str) -> None:
        """Convert visual workflow to standalone code."""
        # Convert to YAML first
        yaml_content = self.visual_to_yaml(workflow_id)
        
        # Create WorkflowSpec from YAML
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            spec = WorkflowSpec.from_yaml(f.name)
        
        # Generate code
        if not self.code_generator:
            from pathlib import Path
            templates_dir = Path(__file__).parent / "templates"
            self.code_generator = CodeGenerator(templates_dir)
        
        from pathlib import Path
        self.code_generator.generate(spec, Path(output_dir))
    
    def _calculate_node_positions(self, states: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Calculate positions for nodes in a flow layout."""
        positions = []
        
        # Simple grid layout
        nodes_per_row = 3
        node_width = 200
        node_height = 100
        spacing_x = 250
        spacing_y = 150
        
        for i, state in enumerate(states):
            row = i // nodes_per_row
            col = i % nodes_per_row
            
            x = 100 + col * spacing_x
            y = 100 + row * spacing_y
            
            positions.append({"x": x, "y": y})
        
        return positions
    
    def export_workflow(self, workflow_id: str, format: str = "json") -> str:
        """Export workflow in specified format."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if format == "json":
            return json.dumps({
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'version': workflow.version,
                'author': workflow.author,
                'nodes': [node.__dict__ for node in workflow.nodes],
                'edges': [edge.__dict__ for edge in workflow.edges],
                'viewport': workflow.viewport,
                'metadata': workflow.metadata,
                'created_at': workflow.created_at.isoformat(),
                'updated_at': workflow.updated_at.isoformat()
            }, indent=2)
        
        elif format == "yaml":
            return self.visual_to_yaml(workflow_id)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_workflow(self, content: str, format: str = "json") -> str:
        """Import workflow from specified format."""
        if format == "json":
            data = json.loads(content)
            
            workflow = VisualWorkflow(
                id=data.get('id', str(uuid.uuid4())),
                name=data['name'],
                description=data['description'],
                version=data.get('version', '1.0.0'),
                author=data.get('author', ''),
                nodes=[FlowNode(**node) for node in data['nodes']],
                edges=[FlowEdge(**edge) for edge in data['edges']],
                viewport=data.get('viewport', {"x": 0, "y": 0, "zoom": 1}),
                metadata=data.get('metadata', {})
            )
            
            self._workflows[workflow.id] = workflow
            return workflow.id
        
        elif format == "yaml":
            return self.yaml_to_visual(content)
        
        else:
            raise ValueError(f"Unsupported import format: {format}")