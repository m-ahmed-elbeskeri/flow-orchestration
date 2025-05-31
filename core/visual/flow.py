# core/visual/flow.py
"""Flow data models for visual workflows."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class FlowNode:
    """Represents a node in the visual flow."""
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any] = field(default_factory=dict)
    width: Optional[float] = None
    height: Optional[float] = None
    selected: bool = False
    dragging: bool = False
    
    def __post_init__(self):
        """Set default dimensions."""
        if self.width is None:
            self.width = 200
        if self.height is None:
            self.height = 100


@dataclass 
class FlowEdge:
    """Represents an edge (connection) in the visual flow."""
    id: str
    source: str
    target: str
    type: str = "default"
    data: Dict[str, Any] = field(default_factory=dict)
    animated: bool = False
    selected: bool = False
    
    def __post_init__(self):
        """Set default edge properties."""
        if 'label' not in self.data:
            self.data['label'] = self.data.get('type', 'success')


@dataclass
class FlowDefinition:
    """Complete flow definition with metadata."""
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    viewport: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    
    def get_node(self, node_id: str) -> Optional[FlowNode]:
        """Get node by ID."""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_edges_for_node(self, node_id: str) -> List[FlowEdge]:
        """Get all edges connected to a node."""
        return [
            edge for edge in self.edges 
            if edge.source == node_id or edge.target == node_id
        ]
    
    def add_node(self, node: FlowNode) -> None:
        """Add a node to the flow."""
        if not self.get_node(node.id):
            self.nodes.append(node)
    
    def remove_node(self, node_id: str) -> None:
        """Remove node and connected edges."""
        self.nodes = [node for node in self.nodes if node.id != node_id]
        self.edges = [
            edge for edge in self.edges
            if edge.source != node_id and edge.target != node_id
        ]
    
    def add_edge(self, edge: FlowEdge) -> None:
        """Add an edge to the flow."""
        # Remove existing edge with same ID
        self.edges = [e for e in self.edges if e.id != edge.id]
        self.edges.append(edge)
    
    def remove_edge(self, edge_id: str) -> None:
        """Remove edge by ID."""
        self.edges = [edge for edge in self.edges if edge.id != edge_id]
    
    def validate(self) -> List[str]:
        """Validate flow for common issues."""
        errors = []
        
        # Check for orphaned edges
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.id} references unknown source node: {edge.source}")
            if edge.target not in node_ids:
                errors.append(f"Edge {edge.id} references unknown target node: {edge.target}")
        
        # Check for cycles (simple detection)
        # TODO: Implement proper cycle detection
        
        return errors