"""FastAPI REST API application."""

import uuid
import asyncio
import tempfile
import zipfile
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import structlog
import yaml
import base64

from core.config import get_settings
from core.execution.engine import WorkflowEngine
from core.storage.backends.sqlite import SQLiteBackend
from core.agent.base import Agent
from core.api.rest.models import (
    WorkflowCreate, WorkflowResponse, WorkflowStatus,
    StateAdd, WorkflowPause, WorkflowResume
)
from core.api.rest.workflow_editor import router as workflow_editor_router

logger = structlog.get_logger(__name__)
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Workflow Orchestrator API",
    version="0.1.0",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(workflow_editor_router)

# Global instances
engine: Optional[WorkflowEngine] = None
agents: Dict[str, Agent] = {}

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging."""
    logger.error(
        "validation_error",
        errors=exc.errors(),
        body=exc.body,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Request validation failed. Check the 'detail' field for specific errors."
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the workflow engine on startup."""
    global engine
    
    try:
        # Initialize workflows storage
        if not hasattr(app.state, 'workflows'):
            app.state.workflows = {}
            logger.info("workflows_storage_initialized")
            
        # Initialize storage
        storage = SQLiteBackend(settings.database_url)
        await storage.initialize()
        
        # Create workflow engine
        engine = WorkflowEngine(storage)
        
        # Start engine if it has a start method
        if hasattr(engine, 'start'):
            await engine.start()
        
        logger.info("workflow_engine_started")
        logger.info("api_started", host=settings.api_host, port=settings.api_port)
        
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        # Initialize empty workflows storage even if engine fails
        if not hasattr(app.state, 'workflows'):
            app.state.workflows = {}
        engine = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global engine
    
    try:
        if engine and hasattr(engine, 'stop'):
            await engine.stop()
        logger.info("api_stopped")
    except Exception as e:
        logger.error("shutdown_failed", error=str(e))

@app.get(f"{settings.api_prefix}/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.environment,
        "engine_status": "connected" if engine else "disconnected",
        "workflows_count": len(getattr(app.state, 'workflows', {}))
    }

# Debug endpoint to test what the UI is sending
@app.post(f"{settings.api_prefix}/debug-workflows")
async def debug_workflow_creation(request: Request):
    """Debug endpoint to see what data is being sent."""
    try:
        body = await request.body()
        logger.info("debug_request_body", raw_body=body.decode() if body else "empty")
        
        json_body = await request.json()
        logger.info("debug_request_json", json_body=json_body)
        
        return {
            "success": True,
            "received_data": json_body,
            "data_types": {k: type(v).__name__ for k, v in json_body.items()},
            "required_fields": ["name", "agent_name"],
            "optional_fields": ["max_concurrent", "timeout", "metadata", "auto_start"]
        }
    except Exception as e:
        logger.error("debug_request_failed", error=str(e))
        return {"error": str(e)}

# Simple test endpoint
@app.post(f"{settings.api_prefix}/test-create")
async def test_create():
    """Simple test endpoint for UI debugging."""
    return {
        "success": True,
        "message": "API is working",
        "timestamp": datetime.utcnow().isoformat()
    }

# =============================================================================
# YAML WORKFLOW ENDPOINTS
# =============================================================================

@app.get(f"{settings.api_prefix}/workflows")
async def list_workflows(limit: int = 100, offset: int = 0):
    """List all workflows - simplified version."""
    try:
        logger.info("list_workflows_called", limit=limit, offset=offset)
        
        # Get workflows from in-memory storage
        workflows = []
        
        if hasattr(app.state, 'workflows') and app.state.workflows:
            # Get workflows from in-memory storage
            all_workflows = list(app.state.workflows.values())
            workflows = all_workflows[offset:offset+limit]
            logger.info("workflows_from_memory", count=len(workflows))
        else:
            # Return empty list if no workflows
            logger.info("no_workflows_found")
            workflows = []
        
        return workflows
        
    except Exception as e:
        logger.error("list_workflows_failed", error=str(e))
        # Return empty list instead of failing
        return []

@app.post(f"{settings.api_prefix}/workflows/from-yaml")
async def create_workflow_from_yaml(request: Request):
    """Create workflow from YAML content."""
    try:
        data = await request.json()
        logger.info("create_from_yaml_called", data_keys=list(data.keys()))
        
        name = data.get('name')
        yaml_content = data.get('yaml_content')
        auto_start = data.get('auto_start', False)
        
        if not name or not yaml_content:
            raise HTTPException(status_code=400, detail="Name and yaml_content are required")
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Try to parse YAML workflow (optional - fallback if parser fails)
        workflow_info = {
            'name': name,
            'description': 'YAML-based workflow',
            'version': '1.0.0',
            'author': 'User'
        }
        
        states_count = 0
        
        try:
            # Try to parse with our workflow parser first
            from core.generator.parser import WorkflowParser
            parser = WorkflowParser()
            workflow_spec = parser.parse_string(yaml_content)
            workflow_info = {
                'name': workflow_spec.name,
                'description': workflow_spec.description,
                'version': workflow_spec.version,
                'author': workflow_spec.author
            }
            states_count = len(workflow_spec.states)
            logger.info("yaml_parsed_with_workflow_parser", states_count=states_count)
        except Exception as parse_error:
            logger.warning("workflow_parser_failed", error=str(parse_error))
            
            # Fallback: try basic YAML parsing
            try:
                parsed_yaml = yaml.safe_load(yaml_content)
                if isinstance(parsed_yaml, dict):
                    workflow_info.update({
                        'name': parsed_yaml.get('name', name),
                        'description': parsed_yaml.get('description', 'YAML-based workflow'),
                        'version': parsed_yaml.get('version', '1.0.0'),
                        'author': parsed_yaml.get('author', 'User')
                    })
                    states_count = len(parsed_yaml.get('states', []))
                    logger.info("yaml_parsed_basic", states_count=states_count)
            except Exception as basic_parse_error:
                logger.warning("basic_yaml_parse_failed", error=str(basic_parse_error))
                # Continue with provided name and defaults
        
        # Store workflow data
        workflow_data = {
            'workflow_id': workflow_id,
            'name': workflow_info['name'],
            'description': workflow_info['description'],
            'version': workflow_info['version'],
            'author': workflow_info['author'],
            'yaml_content': yaml_content,
            'status': 'created',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'states_count': states_count
        }
        
        # Initialize storage if needed
        if not hasattr(app.state, 'workflows'):
            app.state.workflows = {}
        
        # Store workflow
        app.state.workflows[workflow_id] = workflow_data
        
        # Create basic agent
        try:
            agent = Agent(name=f"{workflow_info['name'].replace(' ', '_')}_agent")
            agents[workflow_id] = agent
            logger.info("agent_created", workflow_id=workflow_id)
        except Exception as agent_error:
            logger.warning("agent_creation_failed", error=str(agent_error))
        
        logger.info("yaml_workflow_created", workflow_id=workflow_id, name=workflow_info['name'])
        
        return {
            "workflow_id": workflow_id,
            "name": workflow_info['name'],
            "status": "created",
            "message": f"Workflow '{workflow_info['name']}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("yaml_workflow_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@app.post(f"{settings.api_prefix}/workflows/validate-yaml")
async def validate_yaml_workflow(request: Request):
    """Validate YAML workflow content."""
    try:
        data = await request.json()
        yaml_content = data.get('yaml')
        
        if not yaml_content:
            raise HTTPException(status_code=400, detail="yaml content is required")
        
        # Basic YAML syntax check
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            if not isinstance(parsed_yaml, dict):
                raise ValueError("YAML must be a dictionary")
            
            # Basic structure validation
            errors = []
            warnings = []
            
            # Check required fields
            if 'name' not in parsed_yaml:
                errors.append("Missing required field: name")
            if 'states' not in parsed_yaml:
                errors.append("Missing required field: states")
            elif not isinstance(parsed_yaml['states'], list):
                errors.append("Field 'states' must be a list")
            elif len(parsed_yaml['states']) == 0:
                warnings.append("Workflow has no states defined")
            
            # Check optional but recommended fields
            if 'description' not in parsed_yaml:
                warnings.append("Missing recommended field: description")
            if 'version' not in parsed_yaml:
                warnings.append("Missing recommended field: version")
            
            # Validate states structure
            if 'states' in parsed_yaml and isinstance(parsed_yaml['states'], list):
                for i, state in enumerate(parsed_yaml['states']):
                    if not isinstance(state, dict):
                        errors.append(f"State {i} must be a dictionary")
                        continue
                    
                    if 'name' not in state:
                        errors.append(f"State {i} missing required field: name")
                    if 'type' not in state:
                        errors.append(f"State {i} missing required field: type")
            
            if errors:
                return {
                    "valid": False,
                    "message": "YAML structure validation failed",
                    "errors": errors,
                    "warnings": warnings
                }
            
            # Try advanced validation with workflow parser
            try:
                from core.generator.parser import WorkflowParser
                parser = WorkflowParser()
                workflow_spec = parser.parse_string(yaml_content)
                
                return {
                    "valid": True,
                    "message": "YAML is valid and well-formed",
                    "workflow_info": {
                        "name": workflow_spec.name,
                        "description": workflow_spec.description,
                        "states_count": len(workflow_spec.states),
                        "version": workflow_spec.version,
                        "author": workflow_spec.author
                    },
                    "errors": [],
                    "warnings": warnings
                }
            except Exception as parser_error:
                logger.warning("advanced_validation_failed", error=str(parser_error))
                # Fall back to basic validation result
                return {
                    "valid": True,
                    "message": "YAML is syntactically valid (advanced validation unavailable)",
                    "workflow_info": {
                        "name": parsed_yaml.get('name', 'Unknown'),
                        "description": parsed_yaml.get('description', ''),
                        "states_count": len(parsed_yaml.get('states', [])),
                        "version": parsed_yaml.get('version', '1.0.0'),
                        "author": parsed_yaml.get('author', 'Unknown')
                    },
                    "errors": [],
                    "warnings": warnings + [f"Advanced validation failed: {str(parser_error)}"]
                }
            
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "message": f"YAML syntax error: {str(e)}",
                "errors": [f"YAML parsing failed: {str(e)}"],
                "warnings": []
            }
        
    except Exception as e:
        logger.error("yaml_validation_failed", error=str(e))
        return {
            "valid": False,
            "message": f"Validation failed: {str(e)}",
            "errors": [str(e)],
            "warnings": []
        }

# =============================================================================
# CODE GENERATION ENDPOINTS
# =============================================================================

def _snake_case(text: str) -> str:
    """Convert text to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _pascal_case(text: str) -> str:
    """Convert text to PascalCase."""
    return ''.join(word.capitalize() for word in re.split(r'[^a-zA-Z0-9]', text) if word)

def _generate_env_setup(workflow_spec):
    """Generate environment setup code."""
    if not hasattr(workflow_spec, 'environment') or not workflow_spec.environment.secrets:
        return "        # No environment setup needed"
    
    lines = []
    for secret in workflow_spec.environment.secrets:
        lines.append(f'        self.context.set_secret("{secret.lower()}", os.getenv("{secret}"))')
    
    return "\n".join(lines)

def _generate_states_setup(workflow_spec):
    """Generate states setup code."""
    lines = []
    for state in workflow_spec.states:
        method_name = f"_state_{_snake_case(state.name)}"
        lines.append(f'        self.states["{state.name}"] = self.{method_name}')
    
    return "\n".join(lines)

def _get_next_state(state, workflow_spec, transition_type='on_success'):
    """Get the next state from transitions."""
    if hasattr(state, 'transitions') and state.transitions:
        for transition in state.transitions:
            if hasattr(transition, 'condition') and transition.condition == transition_type:
                return transition.target
            elif hasattr(transition, 'target'):
                return transition.target
    
    # Find next state in sequence
    state_names = [s.name for s in workflow_spec.states]
    try:
        current_index = state_names.index(state.name)
        if current_index + 1 < len(state_names):
            return state_names[current_index + 1]
    except ValueError:
        pass
    
    return None

def _generate_state_methods(workflow_spec):
    """Generate state method implementations."""
    methods = []
    
    for state in workflow_spec.states:
        method_name = f"_state_{_snake_case(state.name)}"
        
        # Generate method based on state type
        if state.type == 'builtin.start':
            method_body = f'''        """Start state implementation."""
        logger.info("Starting workflow")
        self.context.set_constant("workflow_start_time", time.time())
        self.context.set_constant("workflow_name", "{workflow_spec.name}")
        return "{_get_next_state(state, workflow_spec)}"'''
        
        elif state.type == 'builtin.end':
            method_body = f'''        """End state implementation."""
        start_time = self.context.get_constant("workflow_start_time", time.time())
        duration = time.time() - start_time
        logger.info(f"Workflow completed in {{duration:.2f}} seconds")
        return None'''
        
        elif state.type == 'builtin.transform':
            config_msg = state.config.get('message', 'Processing...') if hasattr(state, 'config') and state.config else 'Processing...'
            method_body = f'''        """Transform state implementation."""
        logger.info("{config_msg}")
        # Add your custom logic here
        self.context.set_state("last_transform", "{state.name}")
        return "{_get_next_state(state, workflow_spec)}"'''
        
        elif state.type == 'builtin.conditional':
            method_body = f'''        """Conditional state implementation."""
        # Add your condition logic here
        condition_met = True  # Replace with actual condition
        
        if condition_met:
            return "{_get_next_state(state, workflow_spec, 'on_true')}"
        else:
            return "{_get_next_state(state, workflow_spec, 'on_false')}"'''
        
        else:
            # Generic state
            method_body = f'''        """Custom state implementation for {state.type}."""
        logger.info("Executing custom state: {state.name}")
        # TODO: Implement {state.type} logic
        self.context.set_state("last_executed", "{state.name}")
        return "{_get_next_state(state, workflow_spec)}"'''
        
        method = f'''    async def {method_name}(self, context: Context) -> Optional[str]:
{method_body}'''
        
        methods.append(method)
    
    return "\n\n".join(methods)

@app.post(f"{settings.api_prefix}/workflows/{{workflow_id}}/generate-code")
async def generate_code_from_workflow(workflow_id: str):
    """Generate Python code from a stored workflow."""
    try:
        # Get workflow data
        if not hasattr(app.state, 'workflows') or workflow_id not in app.state.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_data = app.state.workflows[workflow_id]
        yaml_content = workflow_data.get('yaml_content')
        
        if not yaml_content:
            raise HTTPException(status_code=400, detail="Workflow has no YAML content")
        
        # Parse YAML to workflow spec
        try:
            from core.generator.parser import WorkflowParser
            parser = WorkflowParser()
            workflow_spec = parser.parse_string(yaml_content)
        except Exception as e:
            logger.error("yaml_parsing_failed", error=str(e))
            raise HTTPException(status_code=400, detail=f"Failed to parse YAML: {str(e)}")
        
        # Check if we have the full code generator
        try:
            from core.generator.engine import CodeGenerator
            generator = CodeGenerator()
            
            # Create temporary directory for generated files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                output_dir = temp_path / f"generated-{workflow_spec.snake_case_name}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate all files
                generator.generate_from_spec(workflow_spec, output_dir)
                
                # Create zip file
                zip_path = temp_path / f"{workflow_spec.snake_case_name}.zip"
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in output_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(output_dir)
                            zipf.write(file_path, arcname)
                
                # Read zip file content to return as base64
                with open(zip_path, 'rb') as f:
                    zip_content = f.read()
                
                zip_base64 = base64.b64encode(zip_content).decode('utf-8')
                
                return {
                    "success": True,
                    "workflow_name": workflow_spec.name,
                    "file_name": f"{workflow_spec.snake_case_name}.zip",
                    "content": zip_base64,
                    "content_type": "application/zip",
                    "files_generated": len(list(output_dir.rglob('*'))),
                    "message": f"Generated complete code package for workflow '{workflow_spec.name}'"
                }
                
        except ImportError:
            # Fallback to simple code generation
            logger.warning("full_code_generator_unavailable")
            # Generate simple Python code (same as generate_code_from_yaml_direct)
            return await _generate_simple_python_code(workflow_spec)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_code_from_workflow_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post(f"{settings.api_prefix}/workflows/generate-code-from-yaml")
async def generate_code_from_yaml_direct(request: Request):
    """Generate Python code directly from YAML content."""
    try:
        data = await request.json()
        yaml_content = data.get('yaml_content')
        
        if not yaml_content:
            raise HTTPException(status_code=400, detail="yaml_content is required")
        
        # Parse YAML to workflow spec
        try:
            from core.generator.parser import WorkflowParser
            parser = WorkflowParser()
            workflow_spec = parser.parse_string(yaml_content)
        except Exception as e:
            logger.error("yaml_parsing_failed", error=str(e))
            raise HTTPException(status_code=400, detail=f"Failed to parse YAML: {str(e)}")
        
        return await _generate_simple_python_code(workflow_spec)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_code_from_yaml_direct_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

async def _generate_simple_python_code(workflow_spec):
    """Generate simple Python code from workflow spec."""
    python_code = f'''#!/usr/bin/env python3
"""
Generated workflow: {workflow_spec.name}
Description: {workflow_spec.description}
Author: {workflow_spec.author}
Version: {workflow_spec.version}
Generated: {datetime.utcnow().isoformat()}

This is a standalone workflow generated from YAML.
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Context:
    """Simple context for state execution."""
    def __init__(self):
        self._data = {{}}
        self._outputs = {{}}
        self._constants = {{}}
        self._secrets = {{}}
    
    def get_state(self, key: str, default=None):
        return self._data.get(key, default)
    
    def set_state(self, key: str, value):
        self._data[key] = value
    
    def get_output(self, key: str, default=None):
        return self._outputs.get(key, default)
    
    def set_output(self, key: str, value):
        self._outputs[key] = value
    
    def get_constant(self, key: str, default=None):
        return self._constants.get(key, default)
    
    def set_constant(self, key: str, value):
        self._constants[key] = value
    
    def get_secret(self, key: str):
        return self._secrets.get(key) or os.getenv(key)
    
    def set_secret(self, key: str, value):
        self._secrets[key] = value

class {_pascal_case(workflow_spec.name)}Workflow:
    """Generated workflow class for {workflow_spec.name}."""
    
    def __init__(self):
        self.context = Context()
        self.states = {{}}
        self._setup_states()
    
    def _setup_states(self):
        """Setup all workflow states."""
        logger.info("Setting up workflow states...")
        
        # Initialize any environment variables
{_generate_env_setup(workflow_spec)}
        
        # Add states
{_generate_states_setup(workflow_spec)}
    
{_generate_state_methods(workflow_spec)}
    
    async def run(self):
        """Execute the workflow."""
        logger.info("Starting workflow: {workflow_spec.name}")
        
        try:
            # Find start state
            start_states = [name for name, func in self.states.items() if 'start' in name.lower()]
            if start_states:
                current_state = start_states[0]
            else:
                current_state = list(self.states.keys())[0] if self.states else None
            
            if not current_state:
                logger.error("No states defined in workflow")
                return
            
            # Execute states
            visited = set()
            while current_state and current_state not in visited:
                visited.add(current_state)
                logger.info(f"Executing state: {{current_state}}")
                
                if current_state in self.states:
                    try:
                        result = await self.states[current_state](self.context)
                        logger.info(f"State {{current_state}} completed")
                        
                        # Handle transitions (simplified)
                        if isinstance(result, str):
                            current_state = result
                        elif isinstance(result, list) and result:
                            current_state = result[0]
                        else:
                            # No more transitions
                            break
                            
                    except Exception as e:
                        logger.error(f"State {{current_state}} failed: {{e}}")
                        break
                else:
                    logger.error(f"State {{current_state}} not found")
                    break
            
            logger.info("Workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {{e}}")
            raise

async def main():
    """Main entry point."""
    workflow = {_pascal_case(workflow_spec.name)}Workflow()
    await workflow.run()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return {
        "success": True,
        "workflow_name": workflow_spec.name,
        "python_code": python_code,
        "file_name": f"{_snake_case(workflow_spec.name)}.py",
        "message": f"Generated Python code for workflow '{workflow_spec.name}'"
    }

@app.get(f"{settings.api_prefix}/workflows/examples")
async def get_workflow_examples():
    """Get example YAML workflows."""
    return {
        "examples": [
            {
                "name": "Simple Workflow",
                "description": "A basic workflow with start, process, and end states",
                "category": "basic",
                "id": "simple"
            },
            {
                "name": "Email Processor", 
                "description": "Process emails and send notifications",
                "category": "integration",
                "id": "email_processor"
            },
            {
                "name": "API Integration",
                "description": "Fetch and process data from external APIs", 
                "category": "integration",
                "id": "api_integration"
            },
            {
                "name": "Data Pipeline",
                "description": "Complete data processing pipeline",
                "category": "data",
                "id": "data_pipeline"
            }
        ]
    }

# =============================================================================
# LEGACY WORKFLOW ENDPOINTS (for backward compatibility)
# =============================================================================

@app.post(
    f"{settings.api_prefix}/workflows",
    response_model=WorkflowResponse
)
async def create_workflow(
    workflow: WorkflowCreate,
    background_tasks: BackgroundTasks
):
    """Create a new workflow (legacy endpoint)."""
    try:
        logger.info(
            "creating_legacy_workflow",
            name=workflow.name,
            agent_name=workflow.agent_name,
            max_concurrent=workflow.max_concurrent,
            auto_start=workflow.auto_start
        )
        
        # Create agent
        agent = Agent(
            name=workflow.agent_name,
            max_concurrent=workflow.max_concurrent or settings.worker_concurrency
        )
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Store agent for later use
        agents[workflow_id] = agent
        
        # Store basic workflow data
        workflow_data = {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'description': 'Legacy workflow',
            'version': '1.0.0',
            'author': 'API',
            'status': 'created' if not workflow.auto_start else 'starting',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'states_count': 0
        }
        
        if not hasattr(app.state, 'workflows'):
            app.state.workflows = {}
        app.state.workflows[workflow_id] = workflow_data
        
        # Try to create workflow in engine if available
        if engine:
            try:
                # Check if engine has the create_workflow method
                if hasattr(engine, 'create_workflow'):
                    engine_workflow_id = await engine.create_workflow(
                        name=workflow.name,
                        agent=agent,
                        metadata=workflow.metadata or {}
                    )
                    # Use engine's workflow ID if different
                    workflow_id = engine_workflow_id or workflow_id
                else:
                    logger.warning("engine_missing_create_workflow_method")
            except Exception as e:
                logger.error("engine_create_workflow_failed", error=str(e))
                # Continue without engine
        
        # Start execution in background if requested
        if workflow.auto_start:
            if engine and hasattr(engine, 'execute_workflow'):
                background_tasks.add_task(
                    execute_workflow_background,
                    workflow_id,
                    agent,
                    workflow.timeout
                )
            else:
                logger.warning("auto_start_requested_but_no_engine")
        
        logger.info("legacy_workflow_created", workflow_id=workflow_id, name=workflow.name)
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            name=workflow.name,
            agent_name=workflow.agent_name,
            status="created" if not workflow.auto_start else "starting"
        )
        
    except Exception as e:
        logger.error("legacy_workflow_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

async def execute_workflow_background(workflow_id: str, agent: Agent, timeout: Optional[float]):
    """Background task to execute workflow."""
    try:
        if engine and hasattr(engine, 'execute_workflow'):
            await engine.execute_workflow(workflow_id, agent, timeout)
        else:
            # Fallback: run agent directly
            await agent.run(timeout=timeout)
        logger.info("workflow_execution_completed", workflow_id=workflow_id)
    except Exception as e:
        logger.error("workflow_execution_failed", workflow_id=workflow_id, error=str(e))

@app.get(f"{settings.api_prefix}/workflows/{{workflow_id}}")
async def get_workflow(workflow_id: str):
    """Get a specific workflow."""
    try:
        if hasattr(app.state, 'workflows') and workflow_id in app.state.workflows:
            return app.state.workflows[workflow_id]
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_workflow_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

@app.post(
    f"{settings.api_prefix}/workflows/{{workflow_id}}/execute",
    response_model=Dict[str, str]
)
async def execute_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    timeout: Optional[float] = None
):
    """Execute a workflow."""
    if workflow_id not in agents:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    agent = agents[workflow_id]
    
    try:
        # Update status
        if hasattr(app.state, 'workflows') and workflow_id in app.state.workflows:
            app.state.workflows[workflow_id]['status'] = 'running'
            app.state.workflows[workflow_id]['updated_at'] = datetime.utcnow().isoformat()
        
        background_tasks.add_task(
            execute_workflow_background,
            workflow_id,
            agent,
            timeout
        )
        
        logger.info("workflow_execution_started", workflow_id=workflow_id)
        return {"status": "executing", "workflow_id": workflow_id}
        
    except Exception as e:
        logger.error("execute_workflow_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")

@app.delete(
    f"{settings.api_prefix}/workflows/{{workflow_id}}",
    response_model=Dict[str, str]
)
async def delete_workflow(workflow_id: str):
    """Delete a workflow."""
    try:
        # Cancel in engine if available
        if engine and hasattr(engine, 'cancel_workflow'):
            await engine.cancel_workflow(workflow_id)
        
        # Clean up agent
        agents.pop(workflow_id, None)
        
        # Remove from storage
        if hasattr(app.state, 'workflows') and workflow_id in app.state.workflows:
            del app.state.workflows[workflow_id]
        
        logger.info("workflow_deleted", workflow_id=workflow_id)
        return {"status": "deleted", "workflow_id": workflow_id}
        
    except Exception as e:
        logger.error("delete_workflow_failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

# Additional endpoint for testing workflow creation
@app.post(f"{settings.api_prefix}/workflows/simple")
async def create_simple_workflow(name: str = "Test Workflow", agent_name: str = "test_agent"):
    """Simplified workflow creation endpoint for testing."""
    try:
        workflow_data = WorkflowCreate(
            name=name,
            agent_name=agent_name,
            max_concurrent=5,
            auto_start=False
        )
        
        # Use the main create_workflow function
        background_tasks = BackgroundTasks()
        result = await create_workflow(workflow_data, background_tasks)
        
        return {
            "success": True,
            "workflow": result,
            "message": "Simple workflow created successfully"
        }
    except Exception as e:
        logger.error("simple_workflow_creation_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create simple workflow"
        }