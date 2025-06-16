import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Core imports
from core.config import get_settings, get_features, Features
from core.agent.base import Agent
from core.agent.context import Context
from core.agent.state import StateStatus
from core.execution.engine import WorkflowEngine
from core.generator.parser import WorkflowParser
from core.monitoring.events import WorkflowEvent, EventType, get_event_stream
from core.storage.backends.sqlite import SQLiteBackend
from core.storage.events import WorkflowEvent as StorageWorkflowEvent

# Plugin imports
from plugins.node_registry import node_registry
from plugins.registry import plugin_registry

logger = structlog.get_logger(__name__)
settings = get_settings()
features = Features(settings)

# Global application state
app_state = {
    "workflows": {},
    "executions": {},
    "agents": {},
    "storage": None,
    "event_stream": None,
    "resource_pool": None,
    "alert_manager": None
}

# Storage files
WORKFLOWS_FILE = "workflows_storage.json"
EXECUTIONS_FILE = "executions_storage.json"

def load_workflows_from_storage():
    """Load workflows from JSON storage"""
    try:
        with open(WORKFLOWS_FILE, 'r') as f:
            workflows = json.load(f)
        app_state["workflows"].update(workflows)
        logger.info(f"Loaded {len(workflows)} workflows from storage")
    except FileNotFoundError:
        logger.info("No workflows storage file found, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load workflows: {e}")

def save_workflows_to_storage():
    """Save workflows to JSON storage"""
    try:
        with open(WORKFLOWS_FILE, 'w') as f:
            json.dump(app_state["workflows"], f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save workflows: {e}")

def load_executions_from_storage():
    """Load executions from JSON storage"""
    try:
        with open(EXECUTIONS_FILE, 'r') as f:
            executions = json.load(f)
        app_state["executions"].update(executions)
        logger.info(f"Loaded {len(executions)} executions from storage")
    except FileNotFoundError:
        logger.info("No executions storage file found, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load executions: {e}")

def save_executions_to_storage():
    """Save executions to JSON storage"""
    try:
        with open(EXECUTIONS_FILE, 'w') as f:
            json.dump(app_state["executions"], f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save executions: {e}")

# Pydantic models
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
    parameters: Dict[str, Any]

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

# FastAPI app setup
app = FastAPI(
    title="Workflow Orchestrator API",
    description="AI-native workflow orchestration system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint for real-time execution monitoring
@app.websocket("/api/v1/workflows/{workflow_id}/executions/{execution_id}/ws")
async def execution_websocket_standalone(websocket: WebSocket, workflow_id: str, execution_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates about execution status
            max_wait = 10
            wait_time = 0
            
            while wait_time < max_wait:
                await asyncio.sleep(1)
                wait_time += 1
                
                # Send update every few seconds
                update_count = 0
                start_time = time.time()
                
                if execution_id in app_state["executions"]:
                    execution_data = app_state["executions"][execution_id]
                    agent = app_state["agents"].get(execution_id)
                    current_time = time.time()
                    
                    update_data = {
                        "type": "execution_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_id": execution_id,
                        "workflow_id": workflow_id,
                        "status": execution_data.get("status"),
                        "current_time": current_time
                    }
                    
                    # Get metrics and states if available
                    try:
                        metrics_response = await get_execution_metrics(workflow_id, execution_id)
                        states_response = await get_execution_states(workflow_id, execution_id)
                        
                        update_data.update({
                            "metrics": metrics_response,
                            "states": states_response
                        })
                    except:
                        pass
                    
                    await websocket.send_text(json.dumps(update_data))
                
                # Check if execution is completed
                status = execution_data.get("status")
                if status in ["completed", "failed", "cancelled"]:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {e}")

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.4f}s"
    )
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "success": False}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Workflow Orchestrator API")
    
    # Load existing data
    stored_workflows = load_workflows_from_storage()
    stored_executions = load_executions_from_storage()
    
    # Initialize storage backend
    storage = SQLiteBackend(settings.database_url)
    await storage.initialize()
    app_state["storage"] = storage
    
    # Initialize event stream
    from core.monitoring.events import get_event_stream
    app_state["event_stream"] = get_event_stream()
    
    logger.info("Workflow Orchestrator API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Workflow Orchestrator API")
    
    # Save current state
    save_workflows_to_storage()
    save_executions_to_storage()
    
    # Close storage connections
    if app_state.get("storage"):
        await app_state["storage"].close()
    
    logger.info("Workflow Orchestrator API shutdown complete")

async def periodic_save():
    """Periodically save state to storage"""
    while True:
        await asyncio.sleep(60)  # Save every minute
        save_workflows_to_storage()
        save_executions_to_storage()

# Health and system endpoints
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/system/info")
async def get_system_info():
    return {
        "version": "1.0.0",
        "environment": settings.environment,
        "features": {
            "multitenancy": features.multitenancy,
            "ai_optimizer": features.ai_optimizer,
            "marketplace": features.marketplace,
            "enterprise_auth": features.enterprise_auth,
            "scheduling": features.scheduling,
            "webhooks": features.webhooks,
            "file_storage": features.file_storage,
            "metrics": features.metrics
        },
        "uptime": time.time(),
        "stats": {
            "total_workflows": len(app_state["workflows"]),
            "active_workflows": len([w for w in app_state["workflows"].values() if w.get("status") == "active"]),
            "total_executions": len(app_state["executions"]),
            "successful_executions": len([e for e in app_state["executions"].values() if e.get("status") == "completed"]),
            "failed_executions": len([e for e in app_state["executions"].values() if e.get("status") == "failed"])
        }
    }

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Simple auth for demo - replace with proper auth
    if username == "admin" and password == "admin":
        token = str(uuid.uuid4())
        return {"token": token, "user": {"username": username, "role": "admin"}}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/v1/auth/logout")
async def logout():
    return {"message": "Logged out successfully"}

# Node registry endpoints (NEW)
@app.get("/api/v1/nodes/available")
async def get_available_nodes():
    """Get all available node types and their appearances"""
    try:
        nodes = node_registry.get_available_nodes()
        return {
            "success": True,
            "data": {
                node_type: {
                    "icon": appearance.icon,
                    "color": appearance.color,
                    "category": appearance.category,
                    "description": appearance.description,
                    "inputs": appearance.inputs,
                    "outputs": appearance.outputs,
                    "ui_config": appearance.ui_config
                }
                for node_type, appearance in nodes.items()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get available nodes: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available nodes")

@app.get("/api/v1/nodes/categories")
async def get_node_categories():
    """Get nodes grouped by category"""
    try:
        categories = node_registry.get_nodes_by_category()
        return {
            "success": True,
            "data": categories
        }
    except Exception as e:
        logger.error(f"Failed to get node categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get node categories")

@app.get("/api/v1/nodes/{node_type}")
async def get_node_details(node_type: str):
    """Get detailed information about a specific node type"""
    try:
        appearance = node_registry.get_node_appearance(node_type)
        if not appearance:
            raise HTTPException(status_code=404, detail="Node type not found")
        
        return {
            "success": True,
            "data": {
                "icon": appearance.icon,
                "color": appearance.color,
                "category": appearance.category,
                "description": appearance.description,
                "inputs": appearance.inputs,
                "outputs": appearance.outputs,
                "ui_config": appearance.ui_config
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node details for {node_type}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get node details")

@app.post("/api/v1/nodes/refresh")
async def refresh_node_registry():
    """Refresh the node registry (reload from plugins)"""
    try:
        node_registry.refresh()
        return {
            "success": True,
            "message": "Node registry refreshed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to refresh node registry: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh node registry")

# Workflow status and control endpoints
@app.get("/api/v1/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    running_executions = []
    
    # Check for running agent
    agent_info = None
    if workflow_id in app_state["agents"]:
        agent = app_state["agents"][workflow_id]
        agent_info = {
            "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
            "completed_states": list(agent.completed_states),
            "running_states": list(agent._running_states) if hasattr(agent, '_running_states') else [],
            "total_states": len(agent.states)
        }
    
    # Get execution statistics
    all_executions = [e for e in app_state["executions"].values()
                     if e.get("workflow_id") == workflow_id]
    successful_executions = len([e for e in all_executions if e.get("status") == "completed"])
    failed_executions = len([e for e in all_executions if e.get("status") == "failed"])
    
    status_response = {
        "workflow_id": workflow_id,
        "name": workflow_data.get("name"),
        "status": workflow_data.get("status"),
        "agent_info": agent_info,
        "running_executions": running_executions,
        "total_executions": len(all_executions),
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "success_rate": (successful_executions / len(all_executions) * 100) if all_executions else 0
    }
    
    return status_response

@app.post("/api/v1/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    paused_executions = []
    
    # Pause any running agents for this workflow
    for execution_id, agent in app_state["agents"].items():
        execution = app_state["executions"].get(execution_id)
        if execution and execution.get("workflow_id") == workflow_id:
            try:
                await agent.pause()
                paused_executions.append(execution_id)
            except Exception as e:
                logger.error(f"Failed to pause execution {execution_id}: {e}")
    
    return {"paused_executions": paused_executions}

@app.post("/api/v1/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    resumed_executions = []
    
    # Resume any paused agents for this workflow
    for execution_id, agent in app_state["agents"].items():
        execution = app_state["executions"].get(execution_id)
        if execution and execution.get("workflow_id") == workflow_id:
            try:
                await agent.resume()
                resumed_executions.append(execution_id)
            except Exception as e:
                logger.error(f"Failed to resume execution {execution_id}: {e}")
    
    return {"resumed_executions": resumed_executions}

@app.post("/api/v1/workflows/{workflow_id}/stop")
async def stop_workflow(workflow_id: str):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    stopped_executions = []
    
    # Stop any running agents for this workflow
    for execution_id, agent in app_state["agents"].items():
        execution = app_state["executions"].get(execution_id)
        if execution and execution.get("workflow_id") == workflow_id:
            try:
                await agent.cancel_all()
                stopped_executions.append(execution_id)
                # Remove from active agents
                app_state["agents"].pop(execution_id, None)
            except Exception as e:
                logger.error(f"Failed to stop execution {execution_id}: {e}")
    
    return {"stopped_executions": stopped_executions}

# Workflow CRUD endpoints
@app.get("/api/v1/workflows")
async def list_workflows(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    author: Optional[str] = None,
    search: Optional[str] = None
):
    workflows = []
    
    for workflow_id, workflow_data in app_state["workflows"].items():
        # Get execution statistics
        executions = [e for e in app_state["executions"].values() if e.get("workflow_id") == workflow_id]
        successful_executions = len([e for e in executions if e.get("status") == "completed"])
        total_executions = len(executions)
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        # Get executions today
        today = datetime.utcnow().date()
        executions_today = len([e for e in executions
                               if datetime.fromisoformat(e.get("started_at", "").replace('Z', '')).date() == today])
        
        workflow_response_data = {
            "workflow_id": workflow_id,
            "name": workflow_data.get("name"),
            "description": workflow_data.get("description"),
            "status": workflow_data.get("status"),
            "created_at": workflow_data.get("created_at"),
            "updated_at": workflow_data.get("updated_at"),
            "states_count": workflow_data.get("states_count"),
            "last_execution": workflow_data.get("last_execution"),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "executions_today": executions_today
        }
        workflows.append(workflow_response_data)
    
    # Apply filters
    if status:
        workflows = [w for w in workflows if w["status"] == status]
    
    if search:
        search_lower = search.lower()
        workflows = [w for w in workflows if
                    search_lower in w["name"].lower() or
                    search_lower in w["description"].lower()]
    
    if author:
        workflows = [w for w in workflows if author.lower() in w.get("author", "").lower()]
    
    # Pagination
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
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return app_state["workflows"][workflow_id]

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate, background_tasks: BackgroundTasks):
    workflow_id = str(uuid.uuid4())
    
    # Validate YAML content
    try:
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(workflow.yaml_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
    
    current_time = datetime.utcnow().isoformat() + "Z"
    workflow_data = {
        "workflow_id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "yaml_content": workflow.yaml_content,
        "status": "created",
        "created_at": current_time,
        "updated_at": current_time,
        "states_count": len(workflow_spec.states),
        "metadata": workflow.metadata or {}
    }
    
    app_state["workflows"][workflow_id] = workflow_data
    
    # Save to storage
    background_tasks.add_task(save_workflows_to_storage)
    
    response = WorkflowResponse(**workflow_data)
    return response

@app.post("/api/v1/workflows/from-yaml")
async def create_workflow_from_yaml(request: dict):
    name = request.get("name")
    yaml_content = request.get("yaml_content")
    auto_start = request.get("auto_start", False)
    description = request.get("description", "")
    
    if not name or not yaml_content:
        raise HTTPException(status_code=400, detail="Name and yaml_content are required")
    
    workflow = WorkflowCreate(
        name=name,
        description=description,
        yaml_content=yaml_content,
        auto_start=auto_start
    )
    
    result = await create_workflow(workflow, BackgroundTasks())
    return result

@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecutionRequest):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    
    # Create execution record
    execution_data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "parameters": request.parameters or {},
        "priority": request.priority
    }
    
    app_state["executions"][execution_id] = execution_data
    
    # Start background execution
    asyncio.create_task(execute_workflow_background(workflow_id, execution_id))
    
    return {"execution_id": execution_id}

async def execute_workflow_background(workflow_id: str, execution_id: str = None):
    """Execute workflow in background"""
    try:
        workflow_data = app_state["workflows"].get(workflow_id)
        if not workflow_data:
            raise Exception(f"Workflow {workflow_id} not found")
        
        # Parse workflow
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(workflow_data["yaml_content"])
        
        # Create agent
        agent = Agent(f"workflow_{workflow_id}")
        
        # Store agent for monitoring
        if execution_id:
            app_state["agents"][execution_id] = agent
        
        # Build state functions from workflow spec
        state_functions = {}
        
        # Add builtin start state
        async def start_state(context: Context) -> str:
            logger.info("Starting workflow")
            context.set_constant("workflow_start_time", time.time())
            context.set_constant("workflow_name", workflow_spec.name)
            context.set_constant("workflow_version", workflow_spec.version)
            
            # Emit start event
            event_stream = app_state.get("event_stream")
            if event_stream:
                event = WorkflowEvent(
                    workflow_id=workflow_id,
                    event_type=EventType.WORKFLOW_STARTED,
                    timestamp=datetime.utcnow(),
                    data={"execution_id": execution_id}
                )
                await event_stream.publish(event)
            
            # Return first actual state (skip builtin start/end)
            for state in workflow_spec.states:
                if not state.type.startswith('builtin.'):
                    return state.name
            return None
        
        # Add builtin end state
        async def end_state(context: Context) -> str:
            logger.info("Ending workflow")
            start_time = context.get_constant("workflow_start_time", time.time())
            duration = time.time() - start_time
            logger.info(f"Workflow completed in {duration:.2f} seconds")
            return None
        
        # Find start and end states
        start_state_name = None
        end_state_name = None
        
        for state in workflow_spec.states:
            if state.type == 'builtin.start':
                start_state_name = state.name
            elif state.type == 'builtin.end':
                end_state_name = state.name
        
        # Add states to agent
        if start_state_name:
            agent.add_state(start_state_name, start_state)
        if end_state_name:
            agent.add_state(end_state_name, end_state)
        
        # Add other states as simple pass-through for demo
        for state in workflow_spec.states:
            if not state.type.startswith('builtin.'):
                async def state_func(context: Context, state_name=state.name) -> str:
                    logger.info(f"Executing state: {state_name}")
                    await asyncio.sleep(1)  # Simulate work
                    return None
                
                agent.add_state(state.name, state_func)
        
        # Execute workflow - start from entry point (no dependencies)
        entry_states = [s for s in workflow_spec.states if not s.dependencies]
        if entry_states:
            current_state = entry_states[0].name
        elif start_state_name:
            current_state = start_state_name
        else:
            current_state = workflow_spec.states[0].name if workflow_spec.states else None
        
        if not current_state:
            raise Exception("No entry point found for workflow")
        
        # Simple sequential execution for demo
        max_iterations = 10
        iterations = 0
        
        while current_state and iterations < max_iterations:
            iterations += 1
            
            # Create context and execute state
            context = Context({})
            state_func = state_functions.get(current_state)
            
            if state_func:
                next_state = await state_func(context)
                current_state = next_state
            else:
                # Use agent's state execution
                await agent.run_state(current_state)
                break
            
            # Small delay between states to make transitions visible
            if current_state:  # Don't delay if workflow is ending
                await asyncio.sleep(0.5)
        
        # Mark execution as completed
        if execution_id and execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "completed"
            app_state["executions"][execution_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Update workflow last execution
            app_state["workflows"][workflow_id]["last_execution"] = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"Workflow {workflow_id} execution {execution_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        
        # Mark execution as failed
        if execution_id and execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "failed"
            app_state["executions"][execution_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            app_state["executions"][execution_id]["error"] = str(e)
    
    finally:
        # Cleanup
        async def cleanup_agent():
            await asyncio.sleep(1)
            if execution_id in app_state["agents"]:
                app_state["agents"].pop(execution_id, None)
        
        asyncio.create_task(cleanup_agent())

@app.get("/api/v1/workflows/{workflow_id}/executions")
async def get_workflow_executions(workflow_id: str, limit: int = 50, offset: int = 0):
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Filter executions for this workflow
    executions = [
        execution for execution in app_state["executions"].values()
        if execution.get("workflow_id") == workflow_id
    ]
    
    # Sort by started_at descending
    executions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    # Pagination
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
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    
    # Verify it belongs to the workflow
    if execution.get("workflow_id") != workflow_id:
        raise HTTPException(status_code=404, detail="Execution not found for this workflow")
    
    return execution

# Validation endpoints
@app.post("/api/v1/workflows/validate-yaml")
async def validate_yaml_workflow(request: ValidationRequest):
    try:
        parser = WorkflowParser()
        spec = parser.parse_string(request.yaml)
        
        return ValidationResult(
            is_valid=True,
            info={
                "name": spec.name,
                "states_count": len(spec.states),
                "integrations": spec.required_integrations
            }
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[{
                "message": str(e),
                "path": "workflow",
                "code": "VALIDATION_ERROR"
            }]
        )

# Template endpoints
@app.get("/api/v1/workflows/templates")
async def list_templates():
    # Return some example templates since we don't have a template library yet
    templates = [
        {
            "id": "basic",
            "name": "Basic Workflow",
            "description": "A simple workflow template",
            "category": "general",
            "yaml_content": """name: basic_workflow
version: "1.0.0"
description: "A basic workflow template"
author: "Template System"

config:
  timeout: 300
  max_concurrent: 5

environment:
  variables:
    LOG_LEVEL: INFO
  secrets: []

states:
  - name: start
    type: builtin.start
    description: "Workflow starting point"
    transitions:
      - on_success: process

  - name: process
    type: builtin.transform
    description: "Main processing step"
    config:
      message: "Processing data"
    transitions:
      - on_success: end

  - name: end
    type: builtin.end
    description: "Workflow completion"
""",
            "variables": []
        }
    ]
    return templates

# Import these execution monitoring functions from execution_monitor.py
from core.api.rest.execution_monitor import get_execution_states, get_execution_metrics

# Add the execution monitor router
from core.api.rest.execution_monitor import router as execution_router
app.include_router(execution_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)