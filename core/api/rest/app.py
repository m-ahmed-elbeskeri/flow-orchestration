import uuid
import json
import random
import asyncio
import base64
import tempfile
import yaml
import structlog
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import get_settings, Features
from core.storage.backends.sqlite import SQLiteBackend
from core.monitoring.metrics import MetricsCollector
from core.monitoring.alerts import AlertSeverity
from core.generator.engine import CodeGenerator
from core.generator.parser import WorkflowParser

logger = structlog.get_logger(__name__)
settings = get_settings()
features = Features(settings)
security = HTTPBearer()

app_state = {
    "workflows": {},
    "executions": {},
    "metrics_collector": None,
    "storage": None
}

class WorkflowCreate(BaseModel):
    name: str
    description: str = ""
    yaml_content: str
    auto_start: bool = False
    metadata: Optional[Dict[str, Any]] = None

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    yaml_content: Optional[str] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    description: str
    status: str
    created_at: str
    updated_at: str
    states_count: int
    last_execution: Optional[str] = None

class ExecutionRequest(BaseModel):
    parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    priority: int = 1

class ExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    started_at: str
    status: str

class ValidationRequest(BaseModel):
    yaml: str

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}

class CodeGenerationRequest(BaseModel):
    workflow_id: Optional[str] = None
    yaml_content: str

class CodeGenerationResult(BaseModel):
    success: bool
    files: Dict[str, str] = {}
    zip_content: Optional[str] = None
    message: Optional[str] = None

class ScheduleRequest(BaseModel):
    cron: Optional[str] = None
    timezone: str = "UTC"
    enabled: bool = True
    max_instances: int = 1

class AlertRequest(BaseModel):
    rule_name: str
    metric: str
    operator: str
    threshold: float
    severity: AlertSeverity
    message: str

class EnvironmentUpdate(BaseModel):
    variables: Dict[str, str]

class SecretRequest(BaseModel):
    key: str
    value: str

async def get_current_user(credentials = Depends(security)):
    # Simplified auth for demo - in production, validate JWT token
    return {"user_id": "demo_user", "username": "demo"}

app = FastAPI(
    title="Workflow Orchestrator API",
    description="AI-native workflow orchestration system",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {"detail": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"detail": "Internal server error", "status_code": 500}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Workflow Orchestrator API")
    storage = SQLiteBackend(settings.database_url)
    await storage.initialize()
    app_state["storage"] = storage
    app_state["metrics_collector"] = MetricsCollector()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Workflow Orchestrator API")
    if app_state.get("storage"):
        await app_state["storage"].close()

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/v1/system/info")
async def get_system_info():
    collector = app_state["metrics_collector"]
    return {
        "version": "0.1.0",
        "environment": settings.environment,
        "features": [],
        "uptime": 0,
        "stats": {
            "total_workflows": len(app_state["workflows"]),
            "active_workflows": 0,
            "total_executions": len(app_state["executions"]),
            "successful_executions": 0,
            "failed_executions": 0
        }
    }

@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Demo auth - accept any credentials
    token = str(uuid.uuid4())
    return {"token": token, "user": {"username": username}}

@app.post("/api/v1/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    return {"message": "Logged out successfully"}

@app.get("/api/v1/workflows")
async def list_workflows(limit: int = 10, offset: int = 0):
    workflows = []
    for workflow_id, workflow_data in app_state["workflows"].items():
        workflows.append({
            "workflow_id": workflow_id,
            "name": workflow_data["name"],
            "description": workflow_data["description"],
            "status": "created",
            "created_at": workflow_data["created_at"],
            "updated_at": workflow_data["updated_at"],
            "states_count": workflow_data.get("states_count", 0)
        })
    
    total = len(workflows)
    workflows = workflows[offset:offset + limit]
    
    return {
        "items": workflows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }

@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    workflow_data = app_state["workflows"].get(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow_data

@app.get("/api/v1/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get current status of a workflow including running executions."""
    try:
        # Check if workflow exists
        workflow_data = app_state["workflows"].get(workflow_id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Get running executions for this workflow
        running_executions = [
            exec_data for exec_id, exec_data in app_state["executions"].items()
            if exec_data.get("workflow_id") == workflow_id and exec_data.get("status") == "running"
        ]
        
        # Get latest execution
        latest_execution = None
        executions = [
            exec_data for exec_id, exec_data in app_state["executions"].items()
            if exec_data.get("workflow_id") == workflow_id
        ]
        if executions:
            latest_execution = max(executions, key=lambda x: x.get("started_at", ""))
        
        # Determine workflow status
        if running_executions:
            status = "running"
        elif latest_execution:
            status = latest_execution.get("status", "idle")
        else:
            status = "idle"
        
        return {
            "workflow_id": workflow_id,
            "status": status,
            "running_executions": len(running_executions),
            "total_executions": len(executions),
            "latest_execution": latest_execution,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate, background_tasks: BackgroundTasks):
    workflow_id = str(uuid.uuid4())
    parser = WorkflowParser()
    
    try:
        workflow_spec = parser.parse_string(workflow.yaml_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
    
    workflow_data = {
        "workflow_id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "yaml_content": workflow.yaml_content,
        "status": "created",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "states_count": len(workflow_spec.states),
        "metadata": workflow.metadata or {}
    }
    
    app_state["workflows"][workflow_id] = workflow_data
    
    return {
        "workflow_id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "status": "created",
        "created_at": workflow_data["created_at"],
        "updated_at": workflow_data["updated_at"],
        "states_count": workflow_data["states_count"]
    }

@app.post("/api/v1/workflows/from-yaml")
async def create_workflow_from_yaml(request: dict):
    name = request.get("name")
    yaml_content = request.get("yaml_content")
    auto_start = request.get("auto_start", False)
    
    if not name or not yaml_content:
        raise HTTPException(status_code=400, detail="Name and yaml_content are required")
    
    workflow = WorkflowCreate(
        name=name,
        yaml_content=yaml_content,
        auto_start=auto_start
    )
    
    return await create_workflow(workflow, BackgroundTasks())

@app.put("/api/v1/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, updates: WorkflowUpdate):
    workflow_data = app_state["workflows"].get(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if updates.yaml_content:
        parser = WorkflowParser()
        try:
            workflow_spec = parser.parse_string(updates.yaml_content)
            workflow_data["states_count"] = len(workflow_spec.states)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
    
    if updates.name:
        workflow_data["name"] = updates.name
    if updates.description is not None:
        workflow_data["description"] = updates.description
    if updates.yaml_content:
        workflow_data["yaml_content"] = updates.yaml_content
    
    workflow_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    return workflow_data

@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del app_state["workflows"][workflow_id]
    return {"message": "Workflow deleted successfully"}

@app.post("/api/v1/workflows/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, request: dict):
    original = app_state["workflows"].get(workflow_id)
    if not original:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    new_name = request.get("name", f"{original['name']} (Copy)")
    new_workflow_id = str(uuid.uuid4())
    
    new_workflow = original.copy()
    new_workflow.update({
        "workflow_id": new_workflow_id,
        "name": new_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z"
    })
    
    app_state["workflows"][new_workflow_id] = new_workflow
    
    return {
        "workflow_id": new_workflow_id,
        "name": new_name,
        "description": new_workflow["description"],
        "status": "created",
        "created_at": new_workflow["created_at"],
        "updated_at": new_workflow["updated_at"],
        "states_count": new_workflow["states_count"]
    }

@app.post("/api/v1/workflows/validate-yaml")
async def validate_yaml_workflow(request: ValidationRequest):
    parser = WorkflowParser()
    try:
        spec = parser.parse_string(request.yaml)
        return ValidationResult(is_valid=True, info={"states_count": len(spec.states)})
    except Exception as e:
        error_info = {
            "message": str(e),
            "path": "workflow",
            "code": "PARSE_ERROR"
        }
        return ValidationResult(is_valid=False, errors=[error_info])

@app.post("/api/v1/workflows/{workflow_id}/validate")
async def validate_workflow_by_id(workflow_id: str):
    workflow_data = app_state["workflows"].get(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    request = ValidationRequest(yaml=workflow_data["yaml_content"])
    return await validate_yaml_workflow(request)

@app.post("/api/v1/workflows/{workflow_id}/generate-code")
async def generate_code_from_workflow(workflow_id: str):
    workflow_data = app_state["workflows"].get(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(parents=True)
            
            generator = CodeGenerator()
            generator.generate_from_string(workflow_data["yaml_content"], output_dir)
            
            # Create zip file
            zip_path = Path(temp_dir) / f"{workflow_data['name']}.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)
            
            # Read zip content
            with open(zip_path, 'rb') as f:
                zip_content = base64.b64encode(f.read()).decode('utf-8')
            
            return CodeGenerationResult(
                success=True,
                zip_content=zip_content,
                message="Code generated successfully"
            )
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

@app.post("/api/v1/workflows/generate-code-from-yaml")
async def generate_code_from_yaml_direct(request: CodeGenerationRequest):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(parents=True)
            
            generator = CodeGenerator()
            generator.generate_from_string(request.yaml_content, output_dir)
            
            files = {}
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(output_dir))
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[relative_path] = f.read()
            
            return CodeGenerationResult(
                success=True,
                files=files,
                message="Code generated successfully"
            )
    except Exception as e:
        logger.error(f"Error generating code from YAML: {str(e)}")
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecutionRequest, background_tasks: BackgroundTasks):
    """Execute a workflow."""
    try:
        # Check if workflow exists
        workflow_data = app_state["workflows"].get(workflow_id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution_data = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": None,
            "current_state": None,
            "parameters": request.parameters,
            "error": None,
            "states": {},
            "completed_states": 0,
            "total_states": 0
        }
        
        app_state["executions"][execution_id] = execution_data
        
        # Start background execution
        background_tasks.add_task(execute_workflow_background, workflow_id, execution_id)
        
        # Return execution info immediately
        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": execution_data["started_at"],
            "parameters": request.parameters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start workflow execution")

@app.get("/api/v1/workflows/{workflow_id}/executions")
async def list_executions(workflow_id: str, limit: int = 10, offset: int = 0):
    executions = [
        exec_data for exec_id, exec_data in app_state["executions"].items()
        if exec_data.get("workflow_id") == workflow_id
    ]
    
    # Sort by started_at desc
    executions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    total = len(executions)
    executions = executions[offset:offset + limit]
    
    return {
        "items": executions,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}")
async def get_execution(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    if execution.get("workflow_id") != workflow_id:
        raise HTTPException(status_code=404, detail="Execution not found for this workflow")
    
    return execution

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/status")
async def get_execution_status(workflow_id: str, execution_id: str):
    """Get detailed status of a specific execution."""
    try:
        execution = app_state["executions"].get(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        if execution.get("workflow_id") != workflow_id:
            raise HTTPException(status_code=404, detail="Execution not found for this workflow")
        
        # Add some mock state information for now
        states = execution.get("states", {})
        
        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": execution.get("status", "unknown"),
            "started_at": execution.get("started_at"),
            "completed_at": execution.get("completed_at"),
            "current_state": execution.get("current_state"),
            "states": states,
            "progress": {
                "total_states": len(states),
                "completed_states": len([s for s in states.values() if s.get("status") == "completed"]),
                "failed_states": len([s for s in states.values() if s.get("status") == "failed"]),
                "running_states": len([s for s in states.values() if s.get("status") == "running"])
            },
            "error": execution.get("error"),
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states")
async def get_execution_states(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    states = execution.get("states", {})
    return list(states.values())

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/metrics")
async def get_execution_metrics(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    states = execution.get("states", {})
    total_states = len(states)
    completed_states = len([s for s in states.values() if s.get("status") == "completed"])
    failed_states = len([s for s in states.values() if s.get("status") == "failed"])
    active_states = len([s for s in states.values() if s.get("status") == "running"])
    
    start_time = datetime.fromisoformat(execution["started_at"].replace('Z', '+00:00'))
    total_execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    avg_state_time = 0
    if completed_states > 0:
        total_duration = sum(s.get("duration", 0) for s in states.values() if s.get("duration"))
        avg_state_time = total_duration / completed_states
    
    return {
        "totalStates": total_states,
        "completedStates": completed_states,
        "failedStates": failed_states,
        "activeStates": active_states,
        "totalExecutionTime": total_execution_time,
        "avgStateTime": avg_state_time,
        "resourceUtilization": {
            "cpu": random.uniform(10, 80),
            "memory": random.uniform(20, 70),
            "network": random.uniform(5, 50)
        },
        "throughput": completed_states / (total_execution_time / 1000) if total_execution_time > 0 else 0,
        "errorRate": failed_states / total_states if total_states > 0 else 0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/events")
async def get_execution_events(workflow_id: str, execution_id: str, limit: int = 100, level: str = None, event_type: str = None):
    # Mock events for demo
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "state_started",
            "state": "start",
            "message": "Execution started",
            "level": "info"
        }
    ]
    
    if level:
        events = [e for e in events if e["level"] == level]
    if event_type:
        events = [e for e in events if e["type"] == event_type]
    
    return events[:limit]

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/alerts")
async def get_execution_alerts(workflow_id: str, execution_id: str):
    # Mock alerts for demo
    return []

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/logs")
async def get_execution_logs(workflow_id: str, execution_id: str, limit: int = 100, level: str = None, state: str = None):
    # Mock logs for demo
    logs = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "info",
            "message": "Execution started",
            "state": "start"
        }
    ]
    
    if level:
        logs = [l for l in logs if l["level"] == level]
    if state:
        logs = [l for l in logs if l.get("state") == state]
    
    return logs[:limit]

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/pause")
async def pause_execution(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution["status"] = "paused"
    return {"success": True, "message": "Execution paused"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/resume")
async def resume_execution(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution["status"] = "running"
    return {"success": True, "message": "Execution resumed"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/cancel")
async def cancel_execution(workflow_id: str, execution_id: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution["status"] = "cancelled"
    execution["completed_at"] = datetime.utcnow().isoformat() + "Z"
    return {"success": True, "message": "Execution cancelled"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/retry")
async def retry_execution(workflow_id: str, execution_id: str):
    original_execution = app_state["executions"].get(execution_id)
    if not original_execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    new_execution_id = str(uuid.uuid4())
    new_execution = {
        "execution_id": new_execution_id,
        "workflow_id": workflow_id,
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "completed_at": None,
        "current_state": None,
        "parameters": original_execution.get("parameters"),
        "error": None,
        "states": {},
        "completed_states": 0,
        "total_states": 0
    }
    
    app_state["executions"][new_execution_id] = new_execution
    return {"execution_id": new_execution_id}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states/{state_name}/retry")
async def retry_state(workflow_id: str, execution_id: str, state_name: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    states = execution.get("states", {})
    if state_name in states:
        state = states[state_name]
        state["status"] = "pending"
        state["attempts"] = state.get("attempts", 0) + 1
        state["error"] = None
    
    return {"success": True}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states/{state_name}/skip")
async def skip_state(workflow_id: str, execution_id: str, state_name: str):
    execution = app_state["executions"].get(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    states = execution.get("states", {})
    if state_name in states:
        state = states[state_name]
        state["status"] = "skipped"
        state["completed_at"] = datetime.utcnow().isoformat() + "Z"
    
    return {"success": True}

@app.get("/api/v1/workflows/templates")
async def list_templates():
    return []

@app.get("/api/v1/workflows/templates/{template_id}")
async def get_template(template_id: str):
    templates = {
        "basic": {
            "id": "basic",
            "name": "Basic Workflow",
            "description": "A simple workflow template",
            "yaml_content": "name: basic_workflow\nversion: '1.0'\nstates:\n  - name: start\n    type: builtin.start"
        }
    }
    
    template = templates.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

@app.post("/api/v1/workflows/from-template")
async def create_workflow_from_template(request: dict):
    template_id = request.get("template")
    variables = request.get("variables", {})
    
    template = await get_template(template_id)
    yaml_content = template["yaml_content"]
    
    # Apply variables
    for var_name, var_value in variables.items():
        yaml_content = yaml_content.replace(f"${{{var_name}}}", str(var_value))
    
    return await create_workflow_from_yaml({
        "name": template["name"],
        "yaml_content": yaml_content
    })

@app.get("/api/v1/workflows/examples")
async def get_workflow_examples():
    examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
    examples = []
    
    if examples_dir.exists():
        for yaml_file in examples_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()
                    yaml_data = yaml.safe_load(content)
                    
                    examples.append({
                        "name": yaml_data.get("name", yaml_file.stem),
                        "description": yaml_data.get("description", ""),
                        "yaml_content": content
                    })
            except Exception as e:
                logger.warning(f"Failed to load example {yaml_file}: {e}")
    
    return examples

@app.get("/api/v1/workflows/nodes")
async def get_available_nodes():
    # Mock plugin discovery
    try:
        from enterprise.ai.plugin_discovery import PluginDiscovery
        discovery = PluginDiscovery()
        plugins = discovery.discover_all_plugins()
        nodes = []
        
        for plugin_name, plugin_info in plugins.items():
            states = plugin_info.get("states_detailed", {})
            for state_name, state_info in states.items():
                nodes.append({
                    "id": f"{plugin_name}.{state_name}",
                    "name": f"{plugin_name}.{state_name}",
                    "category": "integration",
                    "description": state_info.get("description", ""),
                    "inputs": [],
                    "outputs": []
                })
    except ImportError:
        nodes = []
    
    # Add builtin nodes
    builtin_nodes = [
        {
            "id": "builtin.start",
            "name": "Start",
            "category": "builtin",
            "description": "Workflow start state",
            "inputs": [],
            "outputs": []
        },
        {
            "id": "builtin.end",
            "name": "End",
            "category": "builtin",
            "description": "Workflow end state",
            "inputs": [],
            "outputs": []
        }
    ]
    
    return builtin_nodes + nodes

@app.get("/api/v1/plugins")
async def list_plugins():
    return []

@app.get("/api/v1/plugins/{plugin_name}")
async def get_plugin_info(plugin_name: str):
    from plugins.registry import plugin_registry
    
    plugin = plugin_registry.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {
        "name": plugin.manifest.name,
        "version": plugin.manifest.version,
        "description": plugin.manifest.description,
        "states": list(plugin.register_states().keys())
    }

@app.post("/api/v1/files")
async def upload_file(file_data: dict):
    file_id = str(uuid.uuid4())
    file_info = {
        "file_id": file_id,
        "filename": file_data.get("filename"),
        "size": len(file_data.get("content", "")),
        "content_type": file_data.get("content_type", "application/octet-stream"),
        "uploaded_at": datetime.utcnow().isoformat() + "Z"
    }
    return file_info

@app.get("/api/v1/files/{file_id}/download")
async def download_file(file_id: str):
    raise HTTPException(status_code=404, detail="File not found")

@app.websocket("/api/v1/workflows/{workflow_id}/executions/{execution_id}/ws")
async def execution_websocket(websocket: WebSocket, workflow_id: str, execution_id: str):
    """WebSocket endpoint for real-time execution updates."""
    await websocket.accept()
    
    try:
        # Verify execution exists
        execution = app_state["executions"].get(execution_id)
        if not execution or execution.get("workflow_id") != workflow_id:
            await websocket.send_json({
                "type": "error",
                "message": "Execution not found",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            await websocket.close()
            return
        
        logger.info(f"WebSocket connected for execution {execution_id}")
        
        # Send initial status
        await websocket.send_json({
            "type": "execution_status",
            "data": {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": execution.get("status", "unknown"),
                "started_at": execution.get("started_at"),
                "current_state": execution.get("current_state")
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for either a message from client or timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                
                # Handle client messages (like ping)
                try:
                    msg_data = json.loads(message)
                    if msg_data.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        })
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send periodic status update
                current_execution = app_state["executions"].get(execution_id)
                if current_execution:
                    await websocket.send_json({
                        "type": "execution_update",
                        "data": {
                            "execution_id": execution_id,
                            "status": current_execution.get("status", "unknown"),
                            "current_state": current_execution.get("current_state"),
                            "progress": {
                                "completed": current_execution.get("completed_states", 0),
                                "total": current_execution.get("total_states", 0)
                            }
                        },
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    
                    # Close if execution is complete
                    if current_execution.get("status") in ["completed", "failed", "cancelled"]:
                        await websocket.send_json({
                            "type": "execution_complete",
                            "data": {
                                "execution_id": execution_id,
                                "final_status": current_execution.get("status")
                            },
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        })
                        break
                else:
                    # Execution no longer exists
                    break
                    
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error", 
                "message": "Connection error",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"WebSocket disconnected for execution {execution_id}")

async def execute_workflow_background(workflow_id: str, execution_id: str = None):
    """Background task to simulate workflow execution."""
    try:
        if not execution_id:
            execution_id = str(uuid.uuid4())
        
        workflow_data = app_state["workflows"][workflow_id]
        execution = app_state["executions"][execution_id]
        
        logger.info(f"Starting background execution for workflow {workflow_id}, execution {execution_id}")
        
        # Parse workflow to get states
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(workflow_data["yaml_content"])
        
        # Initialize states
        states = {}
        for state_def in workflow_spec.states:
            states[state_def.name] = {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "duration": None,
                "attempts": 0,
                "error": None
            }
        
        execution["states"] = states
        execution["total_states"] = len(states)
        execution["completed_states"] = 0
        
        # Simulate execution of each state
        for i, state_def in enumerate(workflow_spec.states):
            state_name = state_def.name
            
            # Update execution current state
            execution["current_state"] = state_name
            
            # Update state to running
            states[state_name]["status"] = "running"
            states[state_name]["started_at"] = datetime.utcnow().isoformat() + "Z"
            states[state_name]["attempts"] = 1
            
            logger.info(f"Executing state {state_name} in workflow {workflow_id}")
            
            # Simulate work (1-3 seconds per state)
            await asyncio.sleep(random.uniform(1, 3))
            
            # Simulate occasional failures (10% chance)
            if random.random() < 0.1:
                states[state_name]["status"] = "failed"
                states[state_name]["error"] = f"Simulated failure in state {state_name}"
                states[state_name]["completed_at"] = datetime.utcnow().isoformat() + "Z"
                states[state_name]["duration"] = 2000  # 2 seconds in ms
                
                # Mark execution as failed
                execution["status"] = "failed"
                execution["completed_at"] = datetime.utcnow().isoformat() + "Z"
                execution["error"] = f"Workflow failed at state {state_name}"
                
                logger.error(f"Workflow {workflow_id} failed at state {state_name}")
                return
            else:
                # State completed successfully
                states[state_name]["status"] = "completed"
                states[state_name]["completed_at"] = datetime.utcnow().isoformat() + "Z"
                states[state_name]["duration"] = random.randint(1000, 3000)  # 1-3 seconds in ms
                execution["completed_states"] += 1
        
        # Mark execution as completed
        execution["status"] = "completed"
        execution["completed_at"] = datetime.utcnow().isoformat() + "Z"
        execution["current_state"] = None
        
        logger.info(f"Workflow {workflow_id} execution {execution_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background execution: {str(e)}")
        execution["status"] = "failed"
        execution["completed_at"] = datetime.utcnow().isoformat() + "Z"
        execution["error"] = str(e)