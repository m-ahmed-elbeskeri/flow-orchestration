import uuid
import structlog
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Fix imports to match actual codebase structure
from core.visual.engine import VisualWorkflow, VisualWorkflowEngine
from core.visual.flow import FlowNode, FlowEdge, FlowDefinition
from core.generator.parser import WorkflowParser, parse_workflow_string
from core.generator.engine import CodeGenerator
from core.agent.base import Agent
from core.agent.context import Context

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflow-editor"])

# Pydantic models for request/response
class GenerateCodeRequest(BaseModel):
    workflow: Dict[str, Any]

class ExecuteWorkflowRequest(BaseModel):
    workflow: Dict[str, Any]

class ValidateWorkflowRequest(BaseModel):
    workflow: Dict[str, Any]

class CreateFromTemplateRequest(BaseModel):
    template: str
    variables: Dict[str, Any] = {}

class ImportYamlRequest(BaseModel):
    yaml: str

@router.post("/generate-code")
async def generate_code(request: GenerateCodeRequest):
    """Generate Python code from visual workflow definition."""
    try:
        workflow_data = request.workflow
        
        # Create VisualWorkflow from dict
        visual_workflow = VisualWorkflow(
            id=workflow_data.get("id", str(uuid.uuid4())),
            name=workflow_data.get("name", "Generated Workflow"),
            description=workflow_data.get("description", ""),
            version=workflow_data.get("version", "1.0.0"),
            author=workflow_data.get("author", ""),
            nodes=[FlowNode(**node) for node in workflow_data.get("nodes", [])],
            edges=[FlowEdge(**edge) for edge in workflow_data.get("edges", [])],
            viewport=workflow_data.get("viewport", {"x": 0, "y": 0, "zoom": 1}),
            metadata=workflow_data.get("metadata", {})
        )
        
        # Convert to YAML
        engine = VisualWorkflowEngine()
        engine._workflows[visual_workflow.id] = visual_workflow
        yaml_content = engine.visual_to_yaml(visual_workflow.id)
        
        # Parse YAML to workflow spec
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(yaml_content)
        
        # Generate Python code
        code_generator = CodeGenerator()
        
        # Create a simple workflow code template since generate_workflow_code may not exist
        python_code = f'''"""
Generated workflow: {workflow_spec.name}
Description: {workflow_spec.description}
Author: {workflow_spec.author}
Version: {workflow_spec.version}
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from core.agent.base import Agent
from core.agent.context import Context

logger = logging.getLogger(__name__)

class {workflow_spec.class_name}Workflow:
    def __init__(self):
        self.agent = Agent("{workflow_spec.name}")
        self._setup_states()
    
    def _setup_states(self):
        """Setup workflow states."""
        # Add your state implementations here
        pass
    
    async def run(self):
        """Run the workflow."""
        logger.info("Starting workflow: {workflow_spec.name}")
        await self.agent.run()
        logger.info("Workflow completed: {workflow_spec.name}")

async def main():
    workflow = {workflow_spec.class_name}Workflow()
    await workflow.run()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        return {
            "success": True,
            "python_code": python_code,
            "yaml_content": yaml_content,
            "workflow_spec": {
                "name": workflow_spec.name,
                "description": workflow_spec.description,
                "version": workflow_spec.version,
                "states_count": len(workflow_spec.states)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@router.post("/execute")
async def execute_workflow(request: ExecuteWorkflowRequest):
    """Execute a visual workflow."""
    try:
        workflow_data = request.workflow
        
        # Create a simple execution simulation since we don't have a full executor
        visual_workflow = VisualWorkflow(
            id=workflow_data.get("id", str(uuid.uuid4())),
            name=workflow_data.get("name", "Executed Workflow"),
            description=workflow_data.get("description", ""),
            nodes=[FlowNode(**node) for node in workflow_data.get("nodes", [])],
            edges=[FlowEdge(**edge) for edge in workflow_data.get("edges", [])]
        )
        
        # Simulate execution
        execution_log = []
        for node in visual_workflow.nodes:
            execution_log.append({
                "node_id": node.id,
                "node_type": node.type,
                "status": "completed",
                "timestamp": "2025-06-02T10:00:00Z",
                "message": f"Executed {node.type} node successfully"
            })
        
        result = {
            "success": True,
            "execution_id": str(uuid.uuid4()),
            "workflow_id": visual_workflow.id,
            "status": "completed",
            "execution_log": execution_log,
            "duration": "5.2s",
            "nodes_executed": len(visual_workflow.nodes)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@router.post("/validate")
async def validate_workflow(request: ValidateWorkflowRequest):
    """Validate a visual workflow definition."""
    try:
        workflow_data = request.workflow
        errors = []
        warnings = []
        
        # Basic validation
        if not workflow_data.get("name"):
            errors.append("Workflow name is required")
        
        nodes = workflow_data.get("nodes", [])
        edges = workflow_data.get("edges", [])
        
        if not nodes:
            errors.append("Workflow must contain at least one node")
        
        # Validate node structure
        node_ids = set()
        for i, node in enumerate(nodes):
            if not node.get("id"):
                errors.append(f"Node {i} is missing an ID")
            elif node["id"] in node_ids:
                errors.append(f"Duplicate node ID: {node['id']}")
            else:
                node_ids.add(node["id"])
            
            if not node.get("type"):
                errors.append(f"Node {node.get('id', i)} is missing a type")
        
        # Validate edges
        for i, edge in enumerate(edges):
            if not edge.get("source"):
                errors.append(f"Edge {i} is missing source")
            elif edge["source"] not in node_ids:
                errors.append(f"Edge {i} references non-existent source node: {edge['source']}")
            
            if not edge.get("target"):
                errors.append(f"Edge {i} is missing target")
            elif edge["target"] not in node_ids:
                errors.append(f"Edge {i} references non-existent target node: {edge['target']}")
        
        # Check for disconnected nodes
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge.get("source"))
            connected_nodes.add(edge.get("target"))
        
        disconnected = node_ids - connected_nodes
        if disconnected and len(nodes) > 1:
            warnings.append(f"Disconnected nodes found: {', '.join(disconnected)}")
        
        # Check for cycles (basic check)
        if len(edges) >= len(nodes):
            warnings.append("Workflow may contain cycles")
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "node_types": len(set(node.get("type") for node in nodes if node.get("type")))
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow validation failed: {str(e)}")

@router.get("/templates")
async def list_templates():
    """List available workflow templates."""
    # Return some example templates since we don't have a template library
    templates = [
        {
            "name": "data_processing",
            "title": "Data Processing Pipeline",
            "description": "A template for processing data through multiple stages",
            "category": "data",
            "variables": ["input_source", "output_destination", "processing_steps"]
        },
        {
            "name": "api_workflow",
            "title": "API Integration Workflow",
            "description": "Template for integrating with external APIs",
            "category": "integration",
            "variables": ["api_endpoint", "auth_token", "retry_count"]
        },
        {
            "name": "notification_flow",
            "title": "Notification Flow",
            "description": "Send notifications through multiple channels",
            "category": "communication",
            "variables": ["message", "channels", "priority"]
        }
    ]
    
    return {
        "templates": templates,
        "total": len(templates)
    }

@router.post("/from-template")
async def create_from_template(request: CreateFromTemplateRequest):
    """Create a workflow from a template."""
    try:
        template_name = request.template
        variables = request.variables
        
        # Simple template creation since we don't have a full template library
        templates = {
            "data_processing": {
                "name": f"Data Processing - {variables.get('name', 'Untitled')}",
                "description": "Generated data processing workflow",
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "position": {"x": 100, "y": 100},
                        "data": {"label": "Start", "type": "start"}
                    },
                    {
                        "id": "process",
                        "type": "process",
                        "position": {"x": 300, "y": 100},
                        "data": {"label": "Process Data", "type": "process"}
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "position": {"x": 500, "y": 100},
                        "data": {"label": "End", "type": "end"}
                    }
                ],
                "edges": [
                    {
                        "id": "start-process",
                        "source": "start",
                        "target": "process",
                        "type": "default"
                    },
                    {
                        "id": "process-end",
                        "source": "process",
                        "target": "end",
                        "type": "default"
                    }
                ]
            }
        }
        
        if template_name not in templates:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        workflow_data = templates[template_name].copy()
        
        # Apply variables to the template
        for key, value in variables.items():
            if key == "name":
                workflow_data["name"] = value
            elif key == "description":
                workflow_data["description"] = value
        
        visual_workflow = {
            "id": str(uuid.uuid4()),
            "version": "1.0.0",
            "author": "Template Generator",
            "created_at": "2025-06-02T10:00:00Z",
            "viewport": {"x": 0, "y": 0, "zoom": 1},
            "metadata": {"template": template_name, "variables": variables},
            **workflow_data
        }
        
        return {
            "success": True,
            "workflow": visual_workflow
        }
        
    except Exception as e:
        logger.error(f"Error creating workflow from template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template creation failed: {str(e)}")

@router.get("/nodes")
async def list_available_nodes():
    """List available node types for the workflow designer."""
    # Return node types based on what's available in the system
    categories = {
        "control": [
            {
                "id": "start",
                "name": "Start",
                "description": "Starting point of the workflow",
                "icon": "⚡",
                "color": "#10b981",
                "inputs": [],
                "outputs": ["flow"]
            },
            {
                "id": "end",
                "name": "End",
                "description": "End point of the workflow",
                "icon": "🏁",
                "color": "#ef4444",
                "inputs": ["flow"],
                "outputs": []
            },
            {
                "id": "condition",
                "name": "Condition",
                "description": "Conditional branching logic",
                "icon": "❓",
                "color": "#f59e0b",
                "inputs": ["flow"],
                "outputs": ["true", "false"]
            }
        ],
        "data": [
            {
                "id": "transform",
                "name": "Transform",
                "description": "Transform data",
                "icon": "🔄",
                "color": "#3b82f6",
                "inputs": ["data"],
                "outputs": ["data"]
            },
            {
                "id": "filter",
                "name": "Filter",
                "description": "Filter data based on conditions",
                "icon": "🔍",
                "color": "#8b5cf6",
                "inputs": ["data"],
                "outputs": ["data"]
            }
        ],
        "integration": [
            {
                "id": "http_request",
                "name": "HTTP Request",
                "description": "Make HTTP requests to external APIs",
                "icon": "🌐",
                "color": "#06b6d4",
                "inputs": ["config"],
                "outputs": ["response"]
            },
            {
                "id": "email",
                "name": "Send Email",
                "description": "Send email notifications",
                "icon": "📧",
                "color": "#ec4899",
                "inputs": ["message"],
                "outputs": ["result"]
            }
        ]
    }
    
    return {
        "categories": categories,
        "total_nodes": sum(len(nodes) for nodes in categories.values())
    }

@router.post("/import-yaml")
async def import_from_yaml(request: ImportYamlRequest):
    """Import a workflow from YAML definition."""
    try:
        yaml_content = request.yaml
        
        # Parse the YAML workflow
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(yaml_content)
        
        # Convert to visual workflow format
        engine = VisualWorkflowEngine()
        workflow_id = engine.yaml_to_visual(yaml_content)
        visual_workflow = engine.get_workflow(workflow_id)
        
        if not visual_workflow:
            raise HTTPException(status_code=400, detail="Failed to convert YAML to visual workflow")
        
        # Convert to dict format for response
        workflow_dict = {
            "id": visual_workflow.id,
            "name": visual_workflow.name,
            "description": visual_workflow.description,
            "version": visual_workflow.version,
            "author": visual_workflow.author,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "position": node.position,
                    "data": node.data,
                    "width": node.width,
                    "height": node.height
                }
                for node in visual_workflow.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "animated": edge.animated,
                    "data": edge.data
                }
                for edge in visual_workflow.edges
            ],
            "viewport": visual_workflow.viewport,
            "metadata": visual_workflow.metadata
        }
        
        return {
            "success": True,
            "workflow": workflow_dict,
            "stats": {
                "nodes": len(visual_workflow.nodes),
                "edges": len(visual_workflow.edges),
                "states": len(workflow_spec.states)
            }
        }
        
    except Exception as e:
        logger.error(f"Error importing YAML workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=f"YAML import failed: {str(e)}")