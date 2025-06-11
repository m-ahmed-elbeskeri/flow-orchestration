"""
Complete fixed app.py with persistent storage and enhanced debugging
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Core imports
from core.config import Features, get_settings
from core.storage.backends.sqlite import SQLiteBackend
from core.monitoring.metrics import MetricsCollector
from core.monitoring.events import EventStream
from core.generator.engine import CodeGenerator
from core.generator.parser import WorkflowParser
from core.monitoring.alerts import AlertSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)

# Settings and features
settings = get_settings()
features = Features(settings)

# Global app state
app_state = {
    "workflows": {},
    "executions": {},
    "files": {},
    "storage": None,
    "metrics_collector": None,
    "event_stream": None
}

# Persistent storage configuration
WORKFLOWS_FILE = "workflows_storage.json"
EXECUTIONS_FILE = "executions_storage.json"

# Storage utility functions
def load_workflows_from_storage():
    """Load workflows from persistent storage."""
    try:
        if os.path.exists(WORKFLOWS_FILE):
            with open(WORKFLOWS_FILE, 'r') as f:
                workflows = json.load(f)
                logger.info(f"Loaded {len(workflows)} workflows from storage")
                return workflows
        else:
            logger.info(f"No existing workflows file found at {WORKFLOWS_FILE}")
    except Exception as e:
        logger.error(f"Failed to load workflows from storage: {str(e)}")
    return {}

def save_workflows_to_storage():
    """Save workflows to persistent storage."""
    try:
        with open(WORKFLOWS_FILE, 'w') as f:
            json.dump(app_state["workflows"], f, indent=2, default=str)
        logger.info(f"Saved {len(app_state['workflows'])} workflows to storage")
    except Exception as e:
        logger.error(f"Failed to save workflows to storage: {str(e)}")

def load_executions_from_storage():
    """Load executions from persistent storage."""
    try:
        if os.path.exists(EXECUTIONS_FILE):
            with open(EXECUTIONS_FILE, 'r') as f:
                executions = json.load(f)
                logger.info(f"Loaded {len(executions)} executions from storage")
                return executions
        else:
            logger.info(f"No existing executions file found at {EXECUTIONS_FILE}")
    except Exception as e:
        logger.error(f"Failed to load executions from storage: {str(e)}")
    return {}

def save_executions_to_storage():
    """Save executions to persistent storage."""
    try:
        with open(EXECUTIONS_FILE, 'w') as f:
            json.dump(app_state["executions"], f, indent=2, default=str)
        logger.info(f"Saved {len(app_state['executions'])} executions to storage")
    except Exception as e:
        logger.error(f"Failed to save executions to storage: {str(e)}")

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

# Create FastAPI app
app = FastAPI(
    title="Workflow Orchestrator API",
    description="AI-native workflow orchestration system",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for debugging."""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.4f}s")
    
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return {"detail": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {"detail": "Internal server error", "status_code": 500}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Enhanced startup with persistent storage."""
    logger.info("Starting Workflow Orchestrator API")
    
    # Load existing workflows and executions from storage
    stored_workflows = load_workflows_from_storage()
    app_state["workflows"].update(stored_workflows)
    
    stored_executions = load_executions_from_storage()
    app_state["executions"].update(stored_executions)
    
    logger.info(f"Loaded {len(stored_workflows)} workflows and {len(stored_executions)} executions from persistent storage")
    
    # Initialize storage backend
    try:
        storage = SQLiteBackend(settings.database_url)
        await storage.initialize()
        app_state["storage"] = storage
        logger.info("Storage backend initialized successfully")
    except Exception as e:
        logger.error(f"Storage backend initialization failed: {str(e)}")
        # Continue without storage backend for now
    
    # Initialize other components
    app_state["metrics_collector"] = MetricsCollector()
    app_state["event_stream"] = EventStream()
    
    logger.info(f"App state initialized with {len(app_state['workflows'])} workflows")
    
    # Start periodic save task
    asyncio.create_task(periodic_save())
    logger.info("Periodic save task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Save data on shutdown."""
    logger.info("Shutting down Workflow Orchestrator API")
    
    # Save all data to persistent storage
    save_workflows_to_storage()
    save_executions_to_storage()
    
    # Close storage backend
    if app_state.get("storage"):
        await app_state["storage"].close()
        logger.info("Storage backend closed")

async def periodic_save():
    """Periodically save data to prevent loss."""
    while True:
        await asyncio.sleep(300)  # Save every 5 minutes
        try:
            save_workflows_to_storage()
            save_executions_to_storage()
            logger.debug("Periodic save completed")
        except Exception as e:
            logger.error(f"Periodic save failed: {str(e)}")

# Health check endpoints
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "workflows_count": len(app_state["workflows"]),
        "executions_count": len(app_state["executions"]),
        "version": "1.0.0"
    }

@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information."""
    collector = app_state.get("metrics_collector")
    
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
            "total_executions": len(app_state["executions"])
        }
    }

# Debug endpoints
@app.get("/api/v1/debug/workflows")
async def debug_workflows():
    """Debug endpoint to inspect workflow storage."""
    return {
        "total_workflows": len(app_state["workflows"]),
        "workflow_ids": list(app_state["workflows"].keys()),
        "workflows": app_state["workflows"],
        "app_state_keys": list(app_state.keys()),
        "storage_file_exists": os.path.exists(WORKFLOWS_FILE)
    }

@app.post("/api/v1/debug/reset-workflows")
async def reset_workflows():
    """Reset all workflows (debug endpoint)."""
    app_state["workflows"].clear()
    app_state["executions"].clear()
    save_workflows_to_storage()
    save_executions_to_storage()
    logger.info("All workflows and executions reset")
    return {"message": "All data reset", "total_workflows": 0, "total_executions": 0}

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    """Simple login endpoint."""
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Simple authentication for demo purposes
    if username and password:
        token = str(uuid.uuid4())
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/v1/auth/logout")
async def logout():
    """Logout endpoint."""
    return {"message": "Logged out successfully"}

# Workflow management endpoints
@app.get("/api/v1/workflows")
@app.get("/api/v1/workflows")
async def list_workflows(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    search: Optional[str] = None,
    author: Optional[str] = None
) -> dict:
    """List all workflows with pagination and filtering."""
    try:
        workflows = []
        
        for workflow_id, workflow_data in app_state["workflows"].items():
            workflow_response_data = {
                "workflow_id": workflow_id,
                "name": workflow_data["name"],
                "description": workflow_data.get("description", ""),
                "status": workflow_data.get("status", "created"),
                "created_at": workflow_data.get("created_at", ""),
                "updated_at": workflow_data.get("updated_at", ""),
                "states_count": workflow_data.get("states_count", 0),
                "last_execution": workflow_data.get("last_execution"),
                # Add missing fields that frontend expects
                "success_rate": workflow_data.get("success_rate", 0.0),
                "executions_today": workflow_data.get("executions_today", 0),
                "avg_duration": workflow_data.get("avg_duration", 0),
                "author": workflow_data.get("author", "Unknown"),
                "is_scheduled": workflow_data.get("is_scheduled", False),
                "schedule_info": workflow_data.get("schedule_info", None)
            }
            workflows.append(workflow_response_data)
        
        # Apply filters (same as before)
        if status:
            workflows = [w for w in workflows if w["status"] == status]
        
        if search:
            search_lower = search.lower()
            workflows = [w for w in workflows if 
                       search_lower in w["name"].lower() or 
                       search_lower in w["description"].lower()]
        
        if author:
            workflows = [w for w in workflows if author.lower() in w["description"].lower()]
        
        # Apply pagination
        total = len(workflows)
        workflows = workflows[offset:offset + limit]
        
        # Fix: Return 'workflows' instead of 'items'
        response = {
            "workflows": workflows,  # ← Changed from 'items' to 'workflows'
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(workflows) < total
        }
        
        logger.info(f"Returning {len(workflows)} workflows out of {total}")
        return response
        
    except Exception as e:
        logger.error(f"Error in list_workflows: {e}")
        return {
            "workflows": [],  # ← Changed from 'items' to 'workflows'
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False
        }

@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get a specific workflow."""
    logger.info(f"Getting workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        logger.error(f"Workflow {workflow_id} not found")
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    # Ensure response format
    workflow_response_data = {
        "workflow_id": workflow_data.get("workflow_id", workflow_id),
        "name": workflow_data.get("name", "Unknown"),
        "description": workflow_data.get("description", ""),
        "status": workflow_data.get("status", "created"),
        "created_at": workflow_data.get("created_at", datetime.utcnow().isoformat() + "Z"),
        "updated_at": workflow_data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
        "states_count": workflow_data.get("states_count", 0),
        "last_execution": workflow_data.get("last_execution"),
        "yaml_content": workflow_data.get("yaml_content", ""),
        "metadata": workflow_data.get("metadata", {})
    }
    
    return workflow_response_data

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate, background_tasks: BackgroundTasks):
    """Create a new workflow with persistent storage."""
    workflow_id = str(uuid.uuid4())
    logger.info(f"Creating workflow {workflow_id} with name: {workflow.name}")
    
    try:
        # Parse and validate YAML
        parser = WorkflowParser()
        workflow_spec = parser.parse_string(workflow.yaml_content)
        logger.info(f"YAML parsing successful for workflow {workflow_id}")
        
    except Exception as e:
        logger.error(f"YAML parsing failed for workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
    
    # Create workflow data
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
    
    # Store workflow in memory
    app_state["workflows"][workflow_id] = workflow_data
    
    # Save to persistent storage
    save_workflows_to_storage()
    
    logger.info(f"Workflow {workflow_id} stored and persisted. Total workflows: {len(app_state['workflows'])}")
    
    # Auto start if requested
    if workflow.auto_start:
        logger.info(f"Auto-starting workflow {workflow_id}")
        background_tasks.add_task(execute_workflow_background, workflow_id)
    
    response = WorkflowResponse(**workflow_data)
    logger.info(f"Returning workflow response: {response.workflow_id}")
    
    return response

@app.post("/api/v1/workflows/from-yaml")
async def create_workflow_from_yaml(request: dict):
    """Create workflow from YAML with enhanced debugging."""
    logger.info(f"Received workflow creation request: {request}")
    
    try:
        name = request.get("name")
        yaml_content = request.get("yaml_content")
        auto_start = request.get("auto_start", False)
        description = request.get("description", "")
        
        # Validate required fields
        if not name:
            logger.error("Workflow name is required but not provided")
            raise HTTPException(status_code=400, detail="Workflow name is required")
        
        if not yaml_content:
            logger.error("YAML content is required but not provided")
            raise HTTPException(status_code=400, detail="YAML content is required")
        
        logger.info(f"Creating workflow: name='{name}', auto_start={auto_start}")
        
        workflow = WorkflowCreate(
            name=name,
            description=description,
            yaml_content=yaml_content,
            auto_start=auto_start
        )
        
        result = await create_workflow(workflow, BackgroundTasks())
        logger.info(f"Workflow created successfully: {result.workflow_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating workflow: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.put("/api/v1/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, updates: WorkflowUpdate):
    """Update a workflow."""
    logger.info(f"Updating workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    # Update fields
    if updates.name is not None:
        workflow_data["name"] = updates.name
    if updates.description is not None:
        workflow_data["description"] = updates.description
    if updates.yaml_content is not None:
        # Re-parse YAML if content is updated
        try:
            parser = WorkflowParser()
            workflow_spec = parser.parse_string(updates.yaml_content)
            workflow_data["yaml_content"] = updates.yaml_content
            workflow_data["states_count"] = len(workflow_spec.states)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid workflow YAML: {str(e)}")
    if updates.metadata is not None:
        workflow_data["metadata"] = updates.metadata
    
    workflow_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    # Save to persistent storage
    save_workflows_to_storage()
    
    logger.info(f"Workflow {workflow_id} updated successfully")
    
    return workflow_data

@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow with persistent storage."""
    logger.info(f"Deleting workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Remove from memory
    del app_state["workflows"][workflow_id]
    
    # Also remove related executions
    executions_to_remove = [exec_id for exec_id, exec_data in app_state["executions"].items() 
                           if exec_data.get("workflow_id") == workflow_id]
    for exec_id in executions_to_remove:
        del app_state["executions"][exec_id]
    
    # Save to persistent storage
    save_workflows_to_storage()
    save_executions_to_storage()
    
    logger.info(f"Workflow {workflow_id} and {len(executions_to_remove)} executions deleted")
    
    return {"message": "Workflow deleted successfully"}

@app.post("/api/v1/workflows/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, request: dict):
    """Duplicate a workflow."""
    logger.info(f"Duplicating workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    original = app_state["workflows"][workflow_id]
    new_name = request.get("name", f"{original['name']} (Copy)")
    
    # Create new workflow
    new_workflow = WorkflowCreate(
        name=new_name,
        description=original.get("description", ""),
        yaml_content=original.get("yaml_content", ""),
        auto_start=False,
        metadata=original.get("metadata", {})
    )
    
    result = await create_workflow(new_workflow, BackgroundTasks())
    logger.info(f"Workflow duplicated: {workflow_id} -> {result.workflow_id}")
    
    return result

# Workflow validation endpoints
@app.post("/api/v1/workflows/validate-yaml")
async def validate_yaml_workflow(request: ValidationRequest):
    """Validate workflow YAML."""
    logger.info("Validating workflow YAML")
    
    try:
        parser = WorkflowParser()
        spec = parser.parse_string(request.yaml)
        
        return ValidationResult(
            is_valid=True,
            info={
                "workflow_name": spec.name,
                "states_count": len(spec.states),
                "integrations": len(spec.integrations)
            }
        )
        
    except Exception as e:
        logger.error(f"YAML validation failed: {str(e)}")
        
        error_info = {
            "message": str(e),
            "path": "yaml",
            "code": "VALIDATION_ERROR"
        }
        
        return ValidationResult(
            is_valid=False,
            errors=[error_info]
        )

@app.post("/api/v1/workflows/{workflow_id}/validate")
async def validate_workflow_by_id(workflow_id: str):
    """Validate workflow by ID."""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    request = ValidationRequest(yaml=workflow_data["yaml_content"])
    
    return await validate_yaml_workflow(request)

# Code generation endpoints
@app.post("/api/v1/workflows/{workflow_id}/generate-code")
async def generate_code_from_workflow(workflow_id: str):
    """Generate code from workflow."""
    logger.info(f"Generating code for workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(exist_ok=True)
            
            # Generate code
            generator = CodeGenerator()
            generator.generate_from_string(workflow_data["yaml_content"], output_dir)
            
            # Create ZIP file
            zip_path = Path(temp_dir) / f"{workflow_data['name']}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)
            
            # Read ZIP content
            with open(zip_path, 'rb') as f:
                zip_content = base64.b64encode(f.read()).decode('utf-8')
            
            # Read individual files
            files = {}
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(output_dir))
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[relative_path] = f.read()
            
            return CodeGenerationResult(
                success=True,
                files=files,
                zip_content=zip_content,
                message="Code generated successfully"
            )
            
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

@app.post("/api/v1/workflows/generate-code-from-yaml")
async def generate_code_from_yaml_direct(request: CodeGenerationRequest):
    """Generate code from YAML directly."""
    logger.info("Generating code from YAML directly")
    
    if not request.yaml_content:
        raise HTTPException(status_code=400, detail="YAML content is required")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "generated"
            output_dir.mkdir(exist_ok=True)
            
            # Generate code
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
        logger.error(f"Code generation failed: {str(e)}")
        return CodeGenerationResult(
            success=False,
            message=f"Code generation failed: {str(e)}"
        )

# Workflow execution endpoints
@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecutionRequest):
    """Execute a workflow."""
    logger.info(f"Executing workflow: {workflow_id}")
    
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    current_time = datetime.utcnow().isoformat() + "Z"
    
    execution_data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "status": "running",
        "started_at": current_time,
        "parameters": request.parameters or {},
        "priority": request.priority
    }
    
    # Store execution
    app_state["executions"][execution_id] = execution_data
    save_executions_to_storage()
    
    # Start background execution
    asyncio.create_task(execute_workflow_background(workflow_id, execution_id))
    
    logger.info(f"Workflow execution started: {execution_id}")
    
    return ExecutionResponse(**execution_data)

async def execute_workflow_background(workflow_id: str, execution_id: str = None):
    """Execute workflow in background."""
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    
    logger.info(f"Background execution starting: {workflow_id} -> {execution_id}")
    
    try:
        # Simulate workflow execution
        await asyncio.sleep(2)  # Simulate processing time
        
        # Update execution status
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })
        
        # Update workflow last execution
        if workflow_id in app_state["workflows"]:
            app_state["workflows"][workflow_id]["last_execution"] = datetime.utcnow().isoformat() + "Z"
        
        save_workflows_to_storage()
        save_executions_to_storage()
        
        logger.info(f"Background execution completed: {execution_id}")
        
    except Exception as e:
        logger.error(f"Background execution failed: {str(e)}")
        
        # Update execution status to failed
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "error": str(e)
            })
        
        save_executions_to_storage()

@app.get("/api/v1/workflows/{workflow_id}/executions")
async def get_workflow_executions(workflow_id: str, limit: int = 50, offset: int = 0):
    """Get executions for a workflow."""
    logger.info(f"Getting executions for workflow: {workflow_id}")
    
    # Filter executions for this workflow
    executions = [
        exec_data for exec_data in app_state["executions"].values()
        if exec_data.get("workflow_id") == workflow_id
    ]
    
    # Sort by started_at descending
    executions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    # Apply pagination
    total = len(executions)
    executions = executions[offset:offset + limit]
    
    return {
        "items": executions,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(executions) < total
    }

@app.get("/api/v1/workflows/{workflow_id}/executions/{execution_id}")
async def get_execution(workflow_id: str, execution_id: str):
    """Get specific execution details."""
    if execution_id not in app_state["executions"]:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = app_state["executions"][execution_id]
    
    if execution.get("workflow_id") != workflow_id:
        raise HTTPException(status_code=404, detail="Execution not found for this workflow")
    
    return execution

# Template and example endpoints
@app.get("/api/v1/workflows/templates")
async def list_templates():
    """List available workflow templates."""
    templates = [
        {
            "id": "simple-notification",
            "name": "Simple Notification",
            "description": "A basic workflow that sends notifications",
            "category": "Communication",
            "variables": ["email_recipient", "message"]
        },
        {
            "id": "data-pipeline",
            "name": "Data Processing Pipeline",
            "description": "Process and transform data with validation",
            "category": "Data",
            "variables": ["input_source", "output_destination"]
        }
    ]
    return templates

@app.get("/api/v1/workflows/examples")
async def get_workflow_examples():
    """Get workflow examples."""
    examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
    examples = []
    
    if examples_dir.exists():
        for example_file in examples_dir.glob("*.yaml"):
            try:
                with open(example_file, 'r') as f:
                    content = f.read()
                    yaml_data = yaml.safe_load(content)
                    
                    examples.append({
                        "name": yaml_data.get("name", example_file.stem),
                        "description": yaml_data.get("description", ""),
                        "filename": example_file.name,
                        "content": content
                    })
            except Exception as e:
                logger.error(f"Error reading example file {example_file}: {str(e)}")
    
    return examples

# File upload endpoints
@app.post("/api/v1/files/upload")
async def upload_file(file: bytes, filename: str):
    """Upload a file."""
    file_id = str(uuid.uuid4())
    
    file_info = {
        "file_id": file_id,
        "filename": filename,
        "size": len(file),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "content": base64.b64encode(file).decode('utf-8')
    }
    
    app_state["files"][file_id] = file_info
    
    return {"file_id": file_id, "filename": filename, "size": len(file)}

@app.get("/api/v1/files/{file_id}/download")
async def download_file(file_id: str):
    """Download a file."""
    if file_id not in app_state["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = app_state["files"][file_id]
    content = base64.b64decode(file_info["content"])
    
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file_info['filename']}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)