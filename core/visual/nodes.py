# core/visual/nodes.py
"""Node library for visual workflow editor."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class NodeCategory(Enum):
    """Node categories for organization."""
    BUILTIN = "builtin"
    INTEGRATIONS = "integrations"
    CUSTOM = "custom"
    CONTROL = "control"
    DATA = "data"


@dataclass
class NodeProperty:
    """Node property definition."""
    name: str
    type: str  # string, number, boolean, array, object
    required: bool = False
    default: Any = None
    description: str = ""
    options: Optional[List[str]] = None  # For enum-like properties


@dataclass
class NodeType:
    """Node type definition."""
    id: str
    name: str
    category: NodeCategory
    description: str
    icon: str = "⚡"
    color: str = "#3b82f6"
    inputs: List[NodeProperty] = field(default_factory=list)
    outputs: List[NodeProperty] = field(default_factory=list)
    config_schema: Dict[str, NodeProperty] = field(default_factory=dict)
    
    def create_node_data(self) -> Dict[str, Any]:
        """Create default node data."""
        return {
            'label': self.name,
            'type': self.id,
            'config': {
                prop.name: prop.default 
                for prop in self.config_schema.values()
                if prop.default is not None
            }
        }


class NodeLibrary:
    """Library of available node types."""
    
    def __init__(self):
        self._node_types: Dict[str, NodeType] = {}
        self._register_builtin_nodes()
        self._register_integration_nodes()
    
    def _register_builtin_nodes(self):
        """Register built-in node types."""
        
        # Start node
        self.register_node_type(NodeType(
            id="builtin.start",
            name="Start",
            category=NodeCategory.BUILTIN,
            description="Workflow entry point",
            icon="▶️",
            color="#22c55e",
            outputs=[NodeProperty("success", "trigger")]
        ))
        
        # End node  
        self.register_node_type(NodeType(
            id="builtin.end",
            name="End",
            category=NodeCategory.BUILTIN,
            description="Workflow exit point",
            icon="⏹️", 
            color="#ef4444",
            inputs=[NodeProperty("input", "trigger")]
        ))
        
        # Transform node
        self.register_node_type(NodeType(
            id="builtin.transform",
            name="Transform",
            category=NodeCategory.DATA,
            description="Transform data using custom function",
            icon="🔄",
            color="#8b5cf6",
            inputs=[NodeProperty("input", "any")],
            outputs=[NodeProperty("output", "any")],
            config_schema={
                'function': NodeProperty(
                    "function", "string", required=True,
                    description="Python function to execute",
                    default="async def process(context):\n    return 'processed'"
                )
            }
        ))
        
        # Conditional node
        self.register_node_type(NodeType(
            id="builtin.conditional",
            name="Condition",
            category=NodeCategory.CONTROL,
            description="Conditional branching",
            icon="❓",
            color="#f59e0b",
            inputs=[NodeProperty("input", "any")],
            outputs=[
                NodeProperty("true", "trigger"),
                NodeProperty("false", "trigger")
            ],
            config_schema={
                'condition': NodeProperty(
                    "condition", "string", required=True,
                    description="Boolean expression to evaluate",
                    default="True"
                )
            }
        ))
        
        # Delay node
        self.register_node_type(NodeType(
            id="builtin.delay",
            name="Delay",
            category=NodeCategory.CONTROL,
            description="Add delay to workflow",
            icon="⏰",
            color="#6b7280",
            inputs=[NodeProperty("input", "trigger")],
            outputs=[NodeProperty("output", "trigger")],
            config_schema={
                'seconds': NodeProperty(
                    "seconds", "number", required=True,
                    description="Delay in seconds",
                    default=1
                )
            }
        ))
        
        # Error handler
        self.register_node_type(NodeType(
            id="builtin.error_handler",
            name="Error Handler",
            category=NodeCategory.CONTROL,
            description="Handle errors and exceptions",
            icon="⚠️",
            color="#dc2626",
            inputs=[NodeProperty("error", "error")],
            outputs=[NodeProperty("handled", "trigger")],
            config_schema={
                'log_level': NodeProperty(
                    "log_level", "string", required=False,
                    description="Logging level",
                    default="ERROR",
                    options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                ),
                'notify': NodeProperty(
                    "notify", "boolean", required=False,
                    description="Send error notification",
                    default=False
                )
            }
        ))
    
    def _register_integration_nodes(self):
        """Register integration node types."""
        
        # Gmail nodes
        self.register_node_type(NodeType(
            id="gmail.read_emails",
            name="Read Emails",
            category=NodeCategory.INTEGRATIONS,
            description="Read emails from Gmail",
            icon="📧",
            color="#ea4335",
            inputs=[NodeProperty("trigger", "trigger")],
            outputs=[NodeProperty("emails", "array")],
            config_schema={
                'query': NodeProperty(
                    "query", "string", required=False,
                    description="Gmail search query",
                    default="is:unread"
                ),
                'max_results': NodeProperty(
                    "max_results", "number", required=False,
                    description="Maximum number of emails to fetch",
                    default=10
                ),
                'mark_as_read': NodeProperty(
                    "mark_as_read", "boolean", required=False,
                    description="Mark emails as read after fetching",
                    default=False
                )
            }
        ))
        
        self.register_node_type(NodeType(
            id="gmail.send_email",
            name="Send Email",
            category=NodeCategory.INTEGRATIONS,
            description="Send email via Gmail",
            icon="📤",
            color="#ea4335",
            inputs=[NodeProperty("trigger", "trigger")],
            outputs=[NodeProperty("sent", "trigger")],
            config_schema={
                'to': NodeProperty(
                    "to", "array", required=True,
                    description="Recipient email addresses"
                ),
                'subject': NodeProperty(
                    "subject", "string", required=True,
                    description="Email subject"
                ),
                'body': NodeProperty(
                    "body", "string", required=True,
                    description="Email body content"
                ),
                'cc': NodeProperty(
                    "cc", "array", required=False,
                    description="CC email addresses"
                ),
                'bcc': NodeProperty(
                    "bcc", "array", required=False,
                    description="BCC email addresses"
                )
            }
        ))
        
        # Slack nodes
        self.register_node_type(NodeType(
            id="slack.send_message",
            name="Send Message",
            category=NodeCategory.INTEGRATIONS,
            description="Send message to Slack",
            icon="💬",
            color="#4a154b",
            inputs=[NodeProperty("trigger", "trigger")],
            outputs=[NodeProperty("sent", "trigger")],
            config_schema={
                'channel': NodeProperty(
                    "channel", "string", required=True,
                    description="Slack channel or user"
                ),
                'message': NodeProperty(
                    "message", "string", required=True,
                    description="Message content"
                ),
                'thread_ts': NodeProperty(
                    "thread_ts", "string", required=False,
                    description="Thread timestamp for replies"
                )
            }
        ))
        
        # HTTP nodes
        self.register_node_type(NodeType(
            id="http.request",
            name="HTTP Request",
            category=NodeCategory.INTEGRATIONS,
            description="Make HTTP request",
            icon="🌐",
            color="#059669",
            inputs=[NodeProperty("trigger", "trigger")],
            outputs=[
                NodeProperty("success", "trigger"),
                NodeProperty("error", "trigger")
            ],
            config_schema={
                'url': NodeProperty(
                    "url", "string", required=True,
                    description="Request URL"
                ),
                'method': NodeProperty(
                    "method", "string", required=False,
                    description="HTTP method",
                    default="GET",
                    options=["GET", "POST", "PUT", "DELETE", "PATCH"]
                ),
                'headers': NodeProperty(
                    "headers", "object", required=False,
                    description="Request headers"
                ),
                'body': NodeProperty(
                    "body", "string", required=False,
                    description="Request body"
                )
            }
        ))
    
    def register_node_type(self, node_type: NodeType):
        """Register a new node type."""
        self._node_types[node_type.id] = node_type
    
    def get_node_type(self, type_id: str) -> Optional[NodeType]:
        """Get node type by ID."""
        return self._node_types.get(type_id)
    
    def get_node_types_by_category(self, category: NodeCategory) -> List[NodeType]:
        """Get all node types in a category."""
        return [
            node_type for node_type in self._node_types.values()
            if node_type.category == category
        ]
    
    def get_all_node_types(self) -> Dict[str, NodeType]:
        """Get all registered node types."""
        return self._node_types.copy()
    
    def search_node_types(self, query: str) -> List[NodeType]:
        """Search node types by name or description."""
        query_lower = query.lower()
        results = []
        
        for node_type in self._node_types.values():
            if (query_lower in node_type.name.lower() or 
                query_lower in node_type.description.lower()):
                results.append(node_type)
        
        return results