# core/api/rest/workflow_editor.py
"""API endpoints for workflow editor."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog

from core.workflow.flow import VisualWorkflow, WorkflowExecutor
from core.dag.builder import CodeGenerator, YAMLDAGBuilder
from core.dag.parser import YAMLParser

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflow-editor"])


@router.post("/generate-code")
async def generate_code(request: Dict[str, Any]):
    """Generate Python code from visual workflow."""
    try:
        workflow_data = request["workflow"]
        
        # Convert to VisualWorkflow
        visual_workflow = VisualWorkflow.from_dict(workflow_data)
        
        # Convert to YAML format
        yaml_content = visual_workflow.to_yaml()
        
        # Parse YAML to WorkflowDefinition
        workflow_def = YAMLParser.parse_string(yaml_content)
        
        # Generate Python code
        code_generator = CodeGenerator()
        python_code = code_generator.generate_workflow_code(workflow_def)
        
        return {
            "code": python_code,
            "yaml": yaml_content
        }
        
    except Exception as e:
        logger.error("code_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_workflow(request: Dict[str, Any]):
    """Execute a visual workflow."""
    try:
        workflow_data = request["workflow"]
        
        # Convert to VisualWorkflow
        visual_workflow = VisualWorkflow.from_dict(workflow_data)
        
        # Create executor
        executor = WorkflowExecutor()
        
        # Execute workflow
        result = await executor.execute(visual_workflow)
        
        return {
            "execution_id": result.execution_id,
            "status": result.status,
            "node_executions": {
                node_id: {
                    "status": exec_data["status"],
                    "data": exec_data.get("data"),
                    "error": exec_data.get("error")
                }
                for node_id, exec_data in result.node_executions.items()
            },
            "output_data": result.data
        }
        
    except Exception as e:
        logger.error("workflow_execution_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_workflow(request: Dict[str, Any]):
    """Validate a visual workflow."""
    try:
        workflow_data = request["workflow"]
        
        # Convert to VisualWorkflow
        visual_workflow = VisualWorkflow.from_dict(workflow_data)
        
        # Build DAG to validate
        from core.workflow.flow import WorkflowBuilder
        builder = WorkflowBuilder()
        dag = builder.build_dag(visual_workflow)
        
        # Validate DAG
        dag.validate()
        
        # Validate node parameters
        nodes = builder.create_node_instances(visual_workflow)
        errors = []
        
        for node_id, node in nodes.items():
            node_errors = node.validate_parameters()
            if node_errors:
                errors.extend([f"{node_id}: {error}" for error in node_errors])
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }


@router.get("/templates")
async def list_templates():
    """List available workflow templates."""
    from core.dag.templates import TemplateLibrary
    
    library = TemplateLibrary()
    templates = []
    
    for template_name in library.list_templates():
        template = library.get_template(template_name)
        if template:
            templates.append({
                "name": template.name,
                "description": template.description,
                "variables": list(template.variables.keys())
            })
    
    return {"templates": templates}


@router.post("/from-template")
async def create_from_template(request: Dict[str, Any]):
    """Create workflow from template."""
    from core.dag.templates import TemplateLibrary
    
    template_name = request["template"]
    variables = request.get("variables", {})
    
    library = TemplateLibrary()
    
    try:
        # Render template
        workflow_def = library.render_template(template_name, **variables)
        
        # Convert to visual workflow format
        visual_workflow = {
            "id": f"workflow_{int(datetime.utcnow().timestamp())}",
            "name": workflow_def.name,
            "description": workflow_def.description,
            "nodes": {},
            "connections": []
        }
        
        # Position nodes in a grid
        x, y = 100, 100
        x_spacing, y_spacing = 250, 120
        
        for i, (state_name, state) in enumerate(workflow_def.states.items()):
            visual_workflow["nodes"][state_name] = {
                "type": state.config.get("plugin", state.type),
                "position": {"x": x + (i % 3) * x_spacing, "y": y + (i // 3) * y_spacing},
                "parameters": state.config,
                "name": state_name
            }
            
            # Add connections from transitions
            for transition in state.transitions:
                visual_workflow["connections"].append({
                    "source": {"node": state_name, "output": "main"},
                    "target": {"node": transition.target, "input": "main"}
                })
        
        return {"workflow": visual_workflow}
        
    except Exception as e:
        logger.error("template_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes")
async def list_available_nodes():
    """List all available node types from plugins."""
    from core.workflow.integrations import get_registry
    
    registry = get_registry()
    nodes = registry.list_nodes()
    
    # Group by category
    categories = {}
    for node in nodes:
        category = node["group"][0] if node["group"] else "other"
        if category not in categories:
            categories[category] = []
        categories[category].append(node)
    
    return {"categories": categories}


@router.post("/import-yaml")
async def import_from_yaml(request: Dict[str, Any]):
    """Import workflow from YAML."""
    try:
        yaml_content = request["yaml"]
        
        # Parse YAML
        visual_workflow = VisualWorkflow.from_yaml(yaml_content, str(uuid.uuid4()))
        
        return {"workflow": visual_workflow.to_dict()}
        
    except Exception as e:
        logger.error("yaml_import_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


import uuid
from datetime import datetime