# core/api/rest/app.py
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import zipfile
import base64

from fastapi import (
    FastAPI, HTTPException, Depends, BackgroundTasks, Request, 
    WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
import structlog
import yaml

from core.config import get_settings, Features
from core.storage.backends.sqlite import SQLiteBackend
from core.execution.engine import WorkflowEngine
from core.agent.base import Agent, RetryPolicy
from core.agent.context import Context
from core.monitoring.metrics import MetricsCollector, RealTimeMetrics
from core.monitoring.events import get_event_stream, EventStream, EventFilter
from core.monitoring.alerts import AlertManager, AlertRule, AlertSeverity
from core.resources.requirements import ResourceRequirements
from core.generator.engine import CodeGenerator
from core.generator.parser import WorkflowParser, ValidationError
from core.scheduler.cron import CronScheduler
from plugins.registry import plugin_registry
from enterprise.ai.plugin_discovery import PluginDiscovery

# Configure logging
logger = structlog.get_logger(__name__)
settings = get_settings()
features = Features(settings)

# Security
security = HTTPBearer()

# Global state
app_state = {
    "engine": None,
    "agents": {},
    "workflows": {},
    "executions": {},
    "websocket_connections": {},
    "metrics_collector": None,
    "alert_manager": None,
    "event_stream": None,
    "scheduler": None
}

# Pydantic Models
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
    metadata: Optional[Dict[str, Any]] = None

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
    status: str
    started_at: str
    parameters: Optional[Dict[str, Any]] = None

class ValidationRequest(BaseModel):
    yaml: str

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}

class CodeGenerationRequest(BaseModel):
    workflow_id: Optional[str] = None
    yaml_content: Optional[str] = None

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

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simplified auth - in production, validate JWT token
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {"user_id": "system", "username": "admin"}  # Mock user

# Create FastAPI app
app = FastAPI(
    title="Workflow Orchestrator API",
    description="AI-native workflow orchestration system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {
        "error": {
            "message": exc.detail,
            "code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unexpected error", error=str(exc), path=request.url.path)
    return {
        "error": {
            "message": "Internal server error",
            "code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize storage
        storage = SQLiteBackend(settings.database_url)
        await storage.initialize()
        
        # Initialize workflow engine
        app_state["engine"] = WorkflowEngine(storage)
        await app_state["engine"].start()
        
        # Initialize monitoring
        app_state["metrics_collector"] = MetricsCollector()
        app_state["alert_manager"] = AlertManager()
        await app_state["alert_manager"].start()
        
        # Initialize event stream
        app_state["event_stream"] = get_event_stream()
        
        # Initialize scheduler
        if features.scheduling:
            app_state["scheduler"] = CronScheduler()
            await app_state["scheduler"].start()
        
        # Load plugins
        plugin_registry.load_plugins()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if app_state["engine"]:
            await app_state["engine"].stop()
        
        if app_state["alert_manager"]:
            await app_state["alert_manager"].stop()
        
        if app_state["scheduler"]:
            await app_state["scheduler"].stop()
        
        # Close websocket connections
        for connections in app_state["websocket_connections"].values():
            for ws in connections:
                try:
                    await ws.close()
                except:
                    pass
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Shutdown error", error=str(e))

# Health and System endpoints
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "engine": app_state["engine"] is not None,
            "metrics": app_state["metrics_collector"] is not None,
            "alerts": app_state["alert_manager"] is not None
        }
    }

@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "version": "1.0.0",
        "features": {
            "enterprise": features.enterprise_auth,
            "ai_optimizer": features.ai_optimizer,
            "marketplace": features.marketplace,
            "multitenancy": features.multitenancy
        },
        "plugins": {
            "total": len(plugin_registry.list_plugins()),
            "loaded": len([p for p in plugin_registry.list_plugins() if p.get("status") == "loaded"])
        },
        "settings": {
            "max_concurrent": settings.worker_concurrency,
            "worker_timeout": settings.worker_timeout
        }
    }

@app.get("/api/v1/system/metrics")
async def get_system_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metrics: Optional[List[str]] = Query(None)
):
    """Get system-wide metrics"""
    collector = app_state["metrics_collector"]
    if not collector:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    
    # Implementation would fetch metrics based on parameters
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "active_workflows": len(app_state["workflows"]),
        "active_executions": len(app_state["executions"]),
        "total_events": 1247,
        "error_rate": 2.1
    }

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    """Authenticate user"""
    # Simplified auth - implement proper authentication
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username == "admin" and password == "admin":
        token = str(uuid.uuid4())  # Generate proper JWT in production
        return {
            "token": token,
            "user": {
                "id": "1",
                "username": username,
                "role": "admin"
            }
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/v1/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    """Logout user"""
    return {"message": "Logged out successfully"}

# Workflow endpoints
@app.get("/api/v1/workflows")
async def list_workflows(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    author: Optional[str] = None,
    search: Optional[str] = None
):
    """List workflows with filtering and pagination"""
    workflows = []
    
    # Get workflows from storage or app state
    for workflow_id, workflow_data in app_state["workflows"].items():
        if status and workflow_data.get("status") != status:
            continue
        if author and workflow_data.get("author") != author:
            continue
        if search and search.lower() not in workflow_data.get("name", "").lower():
            continue
        
        workflows.append({
            "workflow_id": workflow_id,
            "name": workflow_data.get("name", ""),
            "description": workflow_data.get("description", ""),
            "status": workflow_data.get("status", "created"),
            "created_at": workflow_data.get("created_at", datetime.utcnow().isoformat()),
            "states_count": workflow_data.get("states_count", 0),
            "author": workflow_data.get("author", ""),
            "last_execution": workflow_data.get("last_execution")
        })
    
    # Apply pagination
    total = len(workflows)
    workflows = workflows[offset:offset + limit]
    
    return {
        "workflows": workflows,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow details"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return app_state["workflows"][workflow_id]

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate, background_tasks: BackgroundTasks):
    """Create a new workflow"""
    try:
        workflow_id = str(uuid.uuid4())
        
        # Parse and validate YAML
        parser = WorkflowParser()
        try:
            workflow_spec = parser.parse_string(workflow.yaml_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
        
        # Store workflow
        workflow_data = {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "yaml_content": workflow.yaml_content,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "states_count": len(workflow_spec.states),
            "author": "system",  # Get from auth context
            "metadata": workflow.metadata or {}
        }
        
        app_state["workflows"][workflow_id] = workflow_data
        
        # Auto-start if requested
        if workflow.auto_start:
            background_tasks.add_task(execute_workflow_background, workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": "created",
            "message": "Workflow created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/from-yaml")
async def create_workflow_from_yaml(request: dict):
    """Create workflow from YAML content"""
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
    """Update workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    if updates.name is not None:
        workflow_data["name"] = updates.name
    if updates.description is not None:
        workflow_data["description"] = updates.description
    if updates.yaml_content is not None:
        workflow_data["yaml_content"] = updates.yaml_content
        # Re-parse and validate
        try:
            parser = WorkflowParser()
            workflow_spec = parser.parse_string(updates.yaml_content)
            workflow_data["states_count"] = len(workflow_spec.states)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")
    if updates.metadata is not None:
        workflow_data["metadata"] = updates.metadata
    
    workflow_data["updated_at"] = datetime.utcnow().isoformat()
    
    return workflow_data

@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Cancel any running executions
    for execution_id, execution in app_state["executions"].items():
        if execution.get("workflow_id") == workflow_id and execution.get("status") == "running":
            execution["status"] = "cancelled"
    
    # Remove workflow
    del app_state["workflows"][workflow_id]
    
    return {"message": "Workflow deleted successfully"}

@app.post("/api/v1/workflows/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, request: dict):
    """Duplicate a workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    original = app_state["workflows"][workflow_id]
    new_name = request.get("name", f"{original['name']} (Copy)")
    
    workflow = WorkflowCreate(
        name=new_name,
        description=original.get("description", ""),
        yaml_content=original["yaml_content"],
        auto_start=False
    )
    
    return await create_workflow(workflow, BackgroundTasks())

# Workflow validation
@app.post("/api/v1/workflows/validate-yaml")
async def validate_yaml_workflow(request: ValidationRequest):
    """Validate workflow YAML"""
    try:
        parser = WorkflowParser()
        spec = parser.parse_string(request.yaml)
        
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info={
                "states_count": len(spec.states),
                "integrations": len(spec.integrations),
                "name": spec.name,
                "version": spec.version
            }
        )
        
    except Exception as e:
        error_info = {
            "message": str(e),
            "type": type(e).__name__,
            "line": getattr(e, 'line', None)
        }
        
        return ValidationResult(
            is_valid=False,
            errors=[error_info],
            warnings=[],
            info={}
        )

@app.post("/api/v1/workflows/{workflow_id}/validate")
async def validate_workflow_by_id(workflow_id: str):
    """Validate existing workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    request = ValidationRequest(yaml=workflow_data["yaml_content"])
    
    return await validate_yaml_workflow(request)

# Code generation
@app.post("/api/v1/workflows/{workflow_id}/generate-code")
async def generate_code_from_workflow(workflow_id: str):
    """Generate code from workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    try:
        # Create temporary directory for generation
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(exist_ok=True)
            
            # Generate code
            generator = CodeGenerator()
            generator.generate_from_string(workflow_data["yaml_content"], output_dir)
            
            # Create zip file
            zip_path = Path(temp_dir) / f"{workflow_data['name']}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)
            
            # Read zip content as base64
            with open(zip_path, 'rb') as f:
                zip_content = base64.b64encode(f.read()).decode('utf-8')
            
            return CodeGenerationResult(
                success=True,
                zip_content=zip_content,
                message="Code generated successfully"
            )
            
    except Exception as e:
        logger.error("Code generation failed", error=str(e))
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

@app.post("/api/v1/workflows/generate-code-from-yaml")
async def generate_code_from_yaml_direct(request: CodeGenerationRequest):
    """Generate code directly from YAML"""
    if not request.yaml_content:
        raise HTTPException(status_code=400, detail="yaml_content is required")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(exist_ok=True)
            
            generator = CodeGenerator()
            generator.generate_from_string(request.yaml_content, output_dir)
            
            # Read generated files
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
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

# Workflow execution
@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecutionRequest):
    """Execute a workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    execution_id = str(uuid.uuid4())
    
    try:
        # Create execution record
        execution_data = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "workflow_name": workflow_data["name"],
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "parameters": request.parameters or {},
            "timeout": request.timeout,
            "priority": request.priority,
            "states": {},
            "metrics": {
                "totalStates": 0,
                "completedStates": 0,
                "failedStates": 0,
                "activeStates": 0
            }
        }
        
        app_state["executions"][execution_id] = execution_data
        workflow_data["last_execution"] = execution_id
        
        # Start execution in background
        asyncio.create_task(execute_workflow_background(workflow_id, execution_id))
        
        return ExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status="running",
            started_at=execution_data["started_at"],
            parameters=request.parameters
        )
        
    except Exception as e:
        logger.error("Failed to start workflow execution", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows/{workflow_id}/executions")
async def get_workflow_executions(
    workflow_id: str,
    limit: int = 50,
    offset: int = 0
):
    """Get workflow executions"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    executions = [
        execution for execution in app_state["executions"].values()
        if execution.get("workflow_id") == workflow_id
    ]
    
    # Sort by start time (newest first)
    executions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    total = len(executions)
    executions = executions[offset:offset + limit]
    
    return {
        "executions": executions,
        "total": total
    }

# Execution monitoring
@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}")
async def get_execution(workflow_id: str, execution_id: str):
    """Get execution details"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    if execution.get("workflow_id") != workflow_id:
        raise HTTPException(status_code=404, detail="Execution not found for this workflow")
    
    return execution

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states")
async def get_execution_states(workflow_id: str, execution_id: str):
    """Get execution states"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    return list(execution.get("states", {}).values())

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/metrics")
async def get_execution_metrics(workflow_id: str, execution_id: str):
    """Get execution metrics"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    
    # Calculate real-time metrics
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
            "cpu": 45.2,  # Mock data - implement real monitoring
            "memory": 67.8,
            "network": 23.1
        },
        "throughput": completed_states / max(total_execution_time / 1000, 1),
        "errorRate": (failed_states / max(total_states, 1)) * 100
    }

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/events")
async def get_execution_events(
    workflow_id: str,
    execution_id: str,
    limit: int = 100,
    level: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None
):
    """Get execution events"""
    # Mock events - implement real event tracking
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "type": "state_started",
            "level": "info",
            "message": "State 'start' began execution",
            "state": "start",
            "metadata": {}
        },
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "type": "state_completed",
            "level": "success",
            "message": "State 'start' completed successfully",
            "state": "start",
            "metadata": {"duration": 150}
        }
    ]
    
    # Apply filters
    if level and level != "all":
        events = [e for e in events if e["level"] == level]
    if event_type and event_type != "all":
        events = [e for e in events if e["type"] == event_type]
    
    return events[:limit]

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/alerts")
async def get_execution_alerts(workflow_id: str, execution_id: str):
    """Get execution alerts"""
    # Mock alerts - implement real alert tracking
    return [
        {
            "id": str(uuid.uuid4()),
            "severity": "warning",
            "title": "High CPU Usage",
            "message": "CPU usage exceeded 80% for more than 2 minutes",
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": False,
            "source": "resource_monitor"
        }
    ]

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}/logs")
async def get_execution_logs(
    workflow_id: str,
    execution_id: str,
    limit: int = 100,
    level: Optional[str] = None,
    state: Optional[str] = None,
    since: Optional[str] = None
):
    """Get execution logs"""
    # Mock logs - implement real log aggregation
    logs = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "message": "Workflow execution started",
            "state": None,
            "execution_id": execution_id
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "debug",
            "message": "State 'start' initialized",
            "state": "start",
            "execution_id": execution_id
        }
    ]
    
    return logs[:limit]

# Execution control
@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/pause")
async def pause_execution(workflow_id: str, execution_id: str):
    """Pause execution"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    if execution["status"] != "running":
        raise HTTPException(status_code=400, detail="Execution is not running")
    
    execution["status"] = "paused"
    execution["paused_at"] = datetime.utcnow().isoformat()
    
    return {"status": "paused"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/resume")
async def resume_execution(workflow_id: str, execution_id: str):
    """Resume execution"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    if execution["status"] != "paused":
        raise HTTPException(status_code=400, detail="Execution is not paused")
    
    execution["status"] = "running"
    execution["resumed_at"] = datetime.utcnow().isoformat()
    
    return {"status": "resumed"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/cancel")
async def cancel_execution(workflow_id: str, execution_id: str):
    """Cancel execution"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    execution["status"] = "cancelled"
    execution["cancelled_at"] = datetime.utcnow().isoformat()
    
    return {"status": "cancelled"}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/retry")
async def retry_execution(workflow_id: str, execution_id: str):
    """Retry execution"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # Create new execution
    new_execution_id = str(uuid.uuid4())
    original_execution = app_state["executions"][execution_id]
    
    new_execution = {
        **original_execution,
        "execution_id": new_execution_id,
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "parent_execution": execution_id,
        "retry_count": original_execution.get("retry_count", 0) + 1
    }
    
    app_state["executions"][new_execution_id] = new_execution
    
    # Start new execution
    asyncio.create_task(execute_workflow_background(workflow_id, new_execution_id))
    
    return {
        "new_execution_id": new_execution_id,
        "status": "running"
    }

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states/{state_name}/retry")
async def retry_state(workflow_id: str, execution_id: str, state_name: str):
    """Retry a specific state"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    states = execution.get("states", {})
    
    if state_name not in states:
        raise HTTPException(status_code=404, detail="State not found")
    
    state = states[state_name]
    state["status"] = "running"
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["last_retry"] = datetime.utcnow().isoformat()
    
    return {"status": "retrying", "state": state_name}

@app.post("/api/v1/workflows/{workflow_id}/executions/{execution_id}/states/{state_name}/skip")
async def skip_state(workflow_id: str, execution_id: str, state_name: str):
    """Skip a state"""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    states = execution.get("states", {})
    
    if state_name not in states:
        raise HTTPException(status_code=404, detail="State not found")
    
    state = states[state_name]
    state["status"] = "skipped"
    state["skipped_at"] = datetime.utcnow().isoformat()
    
    return {"status": "skipped", "state": state_name}

# Templates
@app.get("/api/v1/workflows/templates")
async def list_templates():
    """List workflow templates"""
    return [
        {
            "id": "email-processor",
            "name": "Email Processor",
            "description": "Process incoming emails and send notifications",
            "category": "communication",
            "variables": ["EMAIL_QUERY", "SLACK_CHANNEL"],
            "preview": "A workflow that checks for urgent emails and sends Slack notifications"
        },
        {
            "id": "data-pipeline",
            "name": "Data Pipeline",
            "description": "ETL pipeline for data processing",
            "category": "data",
            "variables": ["SOURCE_URL", "OUTPUT_FORMAT"],
            "preview": "Extract, transform, and load data from various sources"
        }
    ]

@app.get("/api/v1/workflows/templates/{template_id}")
async def get_template(template_id: str):
    """Get template details"""
    templates = {
        "email-processor": {
            "id": "email-processor",
            "name": "Email Processor",
            "description": "Process incoming emails and send notifications",
            "yaml_content": """name: email_processor
version: 1.0.0
description: "Process urgent emails and send notifications"
states:
  - name: start
    type: builtin.start
  - name: fetch_emails
    type: gmail.read_emails
  - name: send_notification
    type: slack.send_message
  - name: end
    type: builtin.end""",
            "variables": ["EMAIL_QUERY", "SLACK_CHANNEL"]
        }
    }
    
    if template_id not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return templates[template_id]

@app.post("/api/v1/workflows/from-template")
async def create_workflow_from_template(request: dict):
    """Create workflow from template"""
    template_id = request.get("template")
    variables = request.get("variables", {})
    
    template = await get_template(template_id)
    
    # Replace variables in template
    yaml_content = template["yaml_content"]
    for var_name, var_value in variables.items():
        yaml_content = yaml_content.replace(f"${{{var_name}}}", str(var_value))
    
    workflow = WorkflowCreate(
        name=f"{template['name']} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description=template["description"],
        yaml_content=yaml_content
    )
    
    return await create_workflow(workflow, BackgroundTasks())

# Examples
@app.get("/api/v1/workflows/examples")
async def get_workflow_examples():
    """Get workflow examples"""
    examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
    examples = []
    
    if examples_dir.exists():
        for yaml_file in examples_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()
                    yaml_data = yaml.safe_load(content)
                    
                examples.append({
                    "id": yaml_file.stem,
                    "name": yaml_data.get("name", yaml_file.stem),
                    "description": yaml_data.get("description", ""),
                    "content": content
                })
            except Exception as e:
                logger.warning("Failed to load example", file=str(yaml_file), error=str(e))
    
    return examples

# Plugins and nodes
@app.get("/api/v1/workflows/nodes")
async def get_available_nodes():
    """Get available node types"""
    discovery = PluginDiscovery()
    plugins = discovery.discover_all_plugins()
    
    nodes = []
    for plugin_name, plugin_info in plugins.items():
        states = plugin_info.get("states_detailed", {})
        for state_name, state_info in states.items():
            nodes.append({
                "type": f"{plugin_name}.{state_name}",
                "name": state_name.replace("_", " ").title(),
                "description": state_info.get("description", ""),
                "category": plugin_info.get("integration_type", "utility"),
                "inputs": state_info.get("inputs", []),
                "outputs": state_info.get("outputs", []),
                "plugin": plugin_name
            })
    
    # Add built-in nodes
    builtin_nodes = [
        {
            "type": "builtin.start",
            "name": "Start",
            "description": "Workflow start point",
            "category": "control",
            "inputs": [],
            "outputs": ["next"]
        },
        {
            "type": "builtin.end",
            "name": "End",
            "description": "Workflow end point",
            "category": "control",
            "inputs": ["previous"],
            "outputs": []
        },
        {
            "type": "builtin.conditional",
            "name": "Conditional",
            "description": "Conditional branching",
            "category": "control",
            "inputs": ["condition"],
            "outputs": ["true", "false"]
        }
    ]
    
    return nodes + builtin_nodes

@app.get("/api/v1/plugins")
async def list_plugins():
    """List available plugins"""
    return plugin_registry.list_plugins()

@app.get("/api/v1/plugins/{plugin_name}")
async def get_plugin_info(plugin_name: str):
    """Get plugin information"""
    plugin = plugin_registry.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    discovery = PluginDiscovery()
    plugins = discovery.discover_all_plugins()
    
    return plugins.get(plugin_name, {})

# File operations
@app.post("/api/v1/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    workflow_id: Optional[str] = Form(None)
):
    """Upload a file"""
    file_id = str(uuid.uuid4())
    
    # Save file (implement proper file storage)
    file_info = {
        "file_id": file_id,
        "filename": file.filename,
        "size": file.size,
        "content_type": file.content_type,
        "workflow_id": workflow_id,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    return file_info

@app.get("/api/v1/files/{file_id}/download")
async def download_file(file_id: str):
    """Download a file"""
    # Implement file retrieval
    raise HTTPException(status_code=404, detail="File not found")

# WebSocket for real-time updates
@app.websocket("/api/v1/workflows/{workflow_id}/executions/{execution_id}/ws")
async def execution_websocket(websocket: WebSocket, workflow_id: str, execution_id: str):
    """WebSocket endpoint for real-time execution monitoring"""
    await websocket.accept()
    
    # Add to connections
    if execution_id not in app_state["websocket_connections"]:
        app_state["websocket_connections"][execution_id] = []
    app_state["websocket_connections"][execution_id].append(websocket)
    
    try:
        while True:
            # Send periodic updates
            if execution_id in app_state["executions"]:
                execution = app_state["executions"][execution_id]
                metrics = await get_execution_metrics(workflow_id, execution_id)
                events = await get_execution_events(workflow_id, execution_id, limit=5)
                
                update = {
                    "type": "execution_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "execution": execution,
                        "metrics": metrics,
                        "recent_events": events
                    }
                }
                
                await websocket.send_json(update)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        # Remove from connections
        if execution_id in app_state["websocket_connections"]:
            app_state["websocket_connections"][execution_id].remove(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close()

# Background task for workflow execution
async def execute_workflow_background(workflow_id: str, execution_id: str = None):
    """Execute workflow in background"""
    try:
        if not execution_id:
            execution_id = str(uuid.uuid4())
        
        execution = app_state["executions"][execution_id]
        workflow_data = app_state["workflows"][workflow_id]
        
        # Parse workflow
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(workflow_data["yaml_content"])
        
        # Simulate workflow execution
        execution["states"] = {}
        
        for i, state in enumerate(workflow_spec.states):
            if execution["status"] in ["cancelled", "paused"]:
                break
            
            state_data = {
                "name": state.name,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "attempts": 1
            }
            execution["states"][state.name] = state_data
            
            # Simulate state execution
            await asyncio.sleep(1)  # Simulate work
            
            # Random success/failure for demo
            import random
            if random.random() > 0.1:  # 90% success rate
                state_data["status"] = "completed"
                state_data["completed_at"] = datetime.utcnow().isoformat()
                state_data["duration"] = 1000  # Mock duration
            else:
                state_data["status"] = "failed"
                state_data["failed_at"] = datetime.utcnow().isoformat()
                state_data["error"] = "Simulated failure"
                break
        
        # Update execution status
        if execution["status"] == "cancelled":
            pass  # Already set
        elif any(s.get("status") == "failed" for s in execution["states"].values()):
            execution["status"] = "failed"
            execution["failed_at"] = datetime.utcnow().isoformat()
        else:
            execution["status"] = "completed"
            execution["completed_at"] = datetime.utcnow().isoformat()
        
        # Notify WebSocket connections
        if execution_id in app_state["websocket_connections"]:
            for ws in app_state["websocket_connections"][execution_id]:
                try:
                    await ws.send_json({
                        "type": "execution_completed",
                        "data": {"execution_id": execution_id, "status": execution["status"]}
                    })
                except:
                    pass
        
    except Exception as e:
        logger.error("Workflow execution failed", error=str(e))
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "failed"
            app_state["executions"][execution_id]["error"] = str(e)

# Serve static files (for UI) - commented out for development
# app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)