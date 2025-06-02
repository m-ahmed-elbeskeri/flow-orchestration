# core/generator/engine.py
"""Main code generation engine - Updated for standalone workflows."""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2
from datetime import datetime
import json
import importlib.util
import inspect

from .parser import WorkflowSpec, parse_workflow_file, parse_workflow_string
from plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generate standalone workflow code from YAML specifications."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = templates_dir
        self._setup_jinja()
        
        # Load available plugins for code generation
        plugin_registry.load_plugins()
        self.available_plugins = {
            plugin['name']: plugin 
            for plugin in plugin_registry.list_plugins()
        }
    
    def _setup_jinja(self):
        """Setup Jinja2 environment."""
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            trim_blocks=False,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        self.jinja_env.filters['snake_case'] = self._snake_case
        self.jinja_env.filters['camel_case'] = self._camel_case
        self.jinja_env.filters['pascal_case'] = self._pascal_case
        self.jinja_env.filters['indent'] = self._indent
        self.jinja_env.filters['quote'] = self._quote
        self.jinja_env.filters['python_value'] = self._python_value
        
        # Add global functions
        self.jinja_env.globals['now'] = datetime.now
        self.jinja_env.globals['datetime'] = datetime
    
    def _python_value(self, value: Any) -> str:
        """Convert JSON-like values to Python syntax."""
        if isinstance(value, bool):
            return 'True' if value else 'False'
        elif value is None:
            return 'None'
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, (list, dict)):
            json_str = json.dumps(value)
            json_str = json_str.replace('true', 'True')
            json_str = json_str.replace('false', 'False')
            json_str = json_str.replace('null', 'None')
            return json_str
        else:
            return str(value)
    
    def generate_from_file(self, yaml_file: Path, output_dir: Path) -> None:
        """Generate code from YAML file."""
        spec = parse_workflow_file(yaml_file)
        self.generate_from_spec(spec, output_dir)
    
    def generate_from_string(self, yaml_content: str, output_dir: Path) -> None:
        """Generate code from YAML string."""
        spec = parse_workflow_string(yaml_content)
        self.generate_from_spec(spec, output_dir)
    
    def generate_from_spec(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate code from workflow specification."""
        logger.info(f"Generating standalone workflow code for '{spec.name}' in {output_dir}")
        
        # Create output directory
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        self._create_directory_structure(output_dir)
        
        # Extract and bundle plugins first (needed for other files)
        self._extract_and_bundle_plugins(spec, output_dir)
        
        # Generate core files
        self._generate_main_workflow(spec, output_dir)
        self._generate_requirements(spec, output_dir)
        self._generate_config_files(spec, output_dir)
        self._generate_utility_files(spec, output_dir)
        self._generate_docker_files(spec, output_dir)
        self._generate_documentation(spec, output_dir)
        self._generate_tests(spec, output_dir)
        
        logger.info(f"✅ Successfully generated standalone workflow project in {output_dir}")
    
    def _create_directory_structure(self, output_dir: Path) -> None:
        """Create project directory structure."""
        directories = [
            "workflow",
            "workflow/plugins",  # Local plugins directory
            "workflow/agent",    # Local agent system
            "config",
            "scripts",
            "tests",
            "docs",
            ".github/workflows"
        ]
        
        for dir_name in directories:
            (output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_dirs = ["workflow", "workflow/plugins", "workflow/agent", "tests"]
        for dir_name in init_dirs:
            init_file = output_dir / dir_name / "__init__.py"
            init_file.write_text('"""Generated workflow package."""\n')
    
    def _extract_and_bundle_plugins(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Extract plugin code and bundle it in the generated workflow."""
        plugins_dir = output_dir / "workflow" / "plugins"
        
        # Create plugin base classes in the local structure
        self._generate_plugin_base_classes(plugins_dir)
        
        # Extract each required plugin
        for integration in spec.integrations:
            plugin_name = integration.name
            plugin = plugin_registry.get_plugin(plugin_name)
            
            if not plugin:
                logger.warning(f"Plugin '{plugin_name}' not found, skipping...")
                continue
            
            logger.info(f"Extracting plugin: {plugin_name}")
            self._extract_plugin_code(plugin, plugins_dir, plugin_name)
    
    def _generate_plugin_base_classes(self, plugins_dir: Path) -> None:
        """Generate standalone plugin base classes."""
        
        # Create base.py with simplified plugin infrastructure
        base_content = '''"""Standalone plugin system base classes."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass


@dataclass
class PluginManifest:
    """Plugin manifest data."""
    name: str
    version: str
    description: str
    author: str
    states: Dict[str, Dict[str, Any]]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    resources: Dict[str, Any]
    dependencies: List[str]


class PluginState(ABC):
    """Base class for plugin states."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def execute(self, context) -> Any:
        """Execute the plugin state."""
        pass
    
    def validate_inputs(self, context) -> None:
        """Validate required inputs are present."""
        pass
    
    def validate_outputs(self, context) -> None:
        """Validate outputs were set correctly."""
        pass


class Plugin(ABC):
    """Base class for plugins."""
    
    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest
        self._states: Dict[str, Type[PluginState]] = {}
    
    @abstractmethod
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register all plugin states."""
        pass
    
    def get_state(self, state_name: str) -> Optional[Type[PluginState]]:
        """Get a specific state class."""
        if not self._states:
            self._states = self.register_states()
        return self._states.get(state_name)
    
    def get_state_function(self, state_name: str, config: Dict[str, Any]):
        """Get a state function for use with Agent.add_state()."""
        state_class = self.get_state(state_name)
        if not state_class:
            raise ValueError(f"State {state_name} not found in plugin {self.manifest.name}")
        
        async def state_function(context) -> Any:
            state = state_class(config)
            state.validate_inputs(context)
            result = await state.execute(context)
            state.validate_outputs(context)
            return result
        
        return state_function


# Simple context and state result classes for standalone operation
class Context:
    """Simplified context for standalone operation."""
    
    def __init__(self):
        self._outputs = {}
        self._state = {}
        self._constants = {}
        self._secrets = {}
    
    def get_output(self, key: str, default=None):
        return self._outputs.get(key, default)
    
    def set_output(self, key: str, value):
        self._outputs[key] = value
    
    def get_state(self, key: str, default=None):
        return self._state.get(key, default)
    
    def set_state(self, key: str, value):
        self._state[key] = value
    
    def get_constant(self, key: str, default=None):
        return self._constants.get(key, default)
    
    def set_constant(self, key: str, value):
        self._constants[key] = value
    
    def get_secret(self, key: str):
        return self._secrets.get(key)
    
    def set_secret(self, key: str, value):
        self._secrets[key] = value


StateResult = Any  # Simplified state result type
'''
        
        (plugins_dir / "base.py").write_text(base_content)
        
        # Create __init__.py for plugins
        init_content = '''"""Standalone plugins package."""

from .base import Plugin, PluginState, PluginManifest, Context, StateResult

__all__ = ['Plugin', 'PluginState', 'PluginManifest', 'Context', 'StateResult']
'''
        (plugins_dir / "__init__.py").write_text(init_content)
    
    def _extract_plugin_code(self, plugin, plugins_dir: Path, plugin_name: str) -> None:
        """Extract the code for a specific plugin."""
        plugin_dir = plugins_dir / plugin_name
        plugin_dir.mkdir(exist_ok=True)
        
        # Get the original plugin directory
        original_plugin_path = plugin.manifest_path.parent
        
        # Copy manifest (modified for standalone)
        self._generate_standalone_manifest(plugin, plugin_dir)
        
        # Extract and modify state classes
        self._extract_plugin_states(plugin, plugin_dir, original_plugin_path)
        
        # Generate standalone plugin class
        self._generate_standalone_plugin_class(plugin, plugin_dir, plugin_name)
    
    def _generate_standalone_manifest(self, plugin, plugin_dir: Path) -> None:
        """Generate a standalone manifest for the plugin."""
        manifest_content = f'''name: {plugin.manifest.name}
version: {plugin.manifest.version}
description: {plugin.manifest.description}
author: {plugin.manifest.author}

states:'''
        
        # Add state definitions
        try:
            states = plugin.register_states()
            for state_name in states.keys():
                manifest_content += f'''
  {state_name}:
    description: "{state_name} state"'''
        except Exception as e:
            logger.warning(f"Could not register states for {plugin.manifest.name}: {e}")
        
        manifest_content += f'''

resources:
  cpu_units: 0.1
  memory_mb: 100
  network_weight: 1.0

dependencies:
{self._format_dependencies_yaml(plugin.manifest.dependencies)}
'''
        
        (plugin_dir / "manifest.yaml").write_text(manifest_content)
    
    def _format_dependencies_yaml(self, dependencies: List[str]) -> str:
        """Format dependencies as YAML list."""
        if not dependencies:
            return "  []"
        
        yaml_deps = []
        for dep in dependencies:
            yaml_deps.append(f"  - {dep}")
        return "\n".join(yaml_deps)
    
    def _extract_plugin_states(self, plugin, plugin_dir: Path, original_plugin_path: Path) -> None:
        """Extract and modify plugin state classes."""
        try:
            states = plugin.register_states()
            
            # Create states.py with modified imports
            states_content = f'''"""
{plugin.manifest.description}

Extracted plugin states for standalone workflow.
Original plugin: {plugin.manifest.name} v{plugin.manifest.version}
"""

import base64
import time
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.encoders import encode_base64

# Import standalone plugin base
from ..base import PluginState, Context, StateResult

'''
            
            # Read original states file if it exists
            original_states_file = original_plugin_path / "states.py"
            if original_states_file.exists():
                original_content = original_states_file.read_text()
                
                # Modify imports in the original content
                modified_content = self._modify_plugin_imports(original_content)
                states_content += modified_content
            else:
                # Generate basic state classes if no states.py exists
                for state_name, state_class in states.items():
                    states_content += self._generate_state_class_code(state_name, state_class)
            
            (plugin_dir / "states.py").write_text(states_content)
            
        except Exception as e:
            logger.error(f"Error extracting states for plugin {plugin.manifest.name}: {e}")
            # Create empty states file
            (plugin_dir / "states.py").write_text(f'"""States for {plugin.manifest.name}"""\n')
    
    def _modify_plugin_imports(self, content: str) -> str:
        """Modify plugin imports to work in standalone mode."""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # Skip imports that reference the original project structure
            if 'from plugins.base import' in line:
                line = line.replace('from plugins.base import', 'from ..base import')
            elif 'from core.agent.context import' in line:
                continue  # Skip - we'll use our own Context
            elif 'from core.agent.state import' in line:
                continue  # Skip - we'll use our own StateResult
            elif 'from plugins.' in line:
                continue  # Skip other plugin imports
            elif 'from core.' in line:
                continue  # Skip core imports
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)
    
    def _generate_state_class_code(self, state_name: str, state_class: type) -> str:
        """Generate code for a state class."""
        class_name = state_class.__name__
        
        # Get the source code if possible
        try:
            import inspect
            source = inspect.getsource(state_class)
            # Modify the source to use standalone imports
            return self._modify_plugin_imports(source)
        except:
            # Fallback: generate a basic class structure
            return f'''

class {class_name}(PluginState):
    """Auto-generated state class for {state_name}."""
    
    async def execute(self, context: Context) -> StateResult:
        """Execute the {state_name} state."""
        # TODO: Implement {state_name} logic
        raise NotImplementedError("This state needs manual implementation")
'''
    
    def _generate_standalone_plugin_class(self, plugin, plugin_dir: Path, plugin_name: str) -> None:
        """Generate the main plugin class for standalone operation."""
        
        # Get state names
        try:
            states = plugin.register_states()
            state_names = list(states.keys())
        except:
            state_names = []
        
        plugin_content = f'''"""
{plugin.manifest.description}

Standalone plugin implementation.
Original: {plugin.manifest.name} v{plugin.manifest.version}
"""

from pathlib import Path
from typing import Dict, Type

from ..base import Plugin, PluginState, PluginManifest

# Import state classes
'''

        # Add state imports
        if state_names:
            plugin_content += "from .states import (\n"
            for name in state_names:
                class_name = self._get_state_class_name(name)
                plugin_content += f"    {class_name},\n"
            plugin_content += ")\n"
        
        plugin_content += f'''

class {self._pascal_case(plugin_name)}Plugin(Plugin):
    """{plugin.manifest.description}"""
    
    def __init__(self):
        # Create manifest
        manifest = PluginManifest(
            name="{plugin.manifest.name}",
            version="{plugin.manifest.version}",
            description="{plugin.manifest.description}",
            author="{plugin.manifest.author}",
            states={{}},
            inputs={{}},
            outputs={{}},
            resources={{}},
            dependencies={json.dumps(plugin.manifest.dependencies)}
        )
        super().__init__(manifest)
    
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register {plugin_name} states."""
        return {{
{self._generate_states_mapping(state_names)}
        }}
'''
        
        (plugin_dir / "__init__.py").write_text(plugin_content)
    
    def _get_state_class_name(self, state_name: str) -> str:
        """Convert state name to class name."""
        # Convert snake_case to PascalCase and add 'State' suffix
        words = state_name.split('_')
        class_name = ''.join(word.capitalize() for word in words)
        if not class_name.endswith('State'):
            class_name += 'State'
        return class_name
    
    def _generate_states_mapping(self, state_names: List[str]) -> str:
        """Generate the states mapping for register_states method."""
        mappings = []
        for state_name in state_names:
            class_name = self._get_state_class_name(state_name)
            mappings.append(f'            "{state_name}": {class_name}')
        return ',\n'.join(mappings)
    
    def _generate_main_workflow(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate main workflow file with standalone plugin imports."""
        template = self.jinja_env.get_template('workflow.py.j2')
        content = template.render(
            spec=spec, 
            plugins=self.available_plugins,
            extracted_plugins=self._get_extracted_plugin_info(spec)
        )
        
        workflow_file = output_dir / "workflow" / "main.py"
        workflow_file.write_text(content, encoding='utf-8')
        
        # Generate workflow runner
        template = self.jinja_env.get_template('run.py.j2')
        content = template.render(spec=spec)
        
        run_file = output_dir / "run.py"
        run_file.write_text(content)
        run_file.chmod(0o755)
    
    def _get_extracted_plugin_info(self, spec: WorkflowSpec) -> Dict[str, Any]:
        """Get information about extracted plugins."""
        extracted = {}
        for integration in spec.integrations:
            plugin_name = integration.name
            plugin = plugin_registry.get_plugin(plugin_name)
            
            if plugin:
                try:
                    states = plugin.register_states()
                    extracted[plugin_name] = {
                        'name': plugin_name,
                        'class_name': self._pascal_case(plugin_name) + 'Plugin',
                        'states': list(states.keys()),
                        'description': plugin.manifest.description
                    }
                except Exception as e:
                    logger.warning(f"Could not get states for {plugin_name}: {e}")
                    extracted[plugin_name] = {
                        'name': plugin_name,
                        'class_name': self._pascal_case(plugin_name) + 'Plugin',
                        'states': [],
                        'description': f"Plugin {plugin_name}"
                    }
        
        return extracted
    
    def _generate_requirements(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate requirements.txt with proper deduplication and version handling."""
        # Base requirements for standalone operation
        base_requirements = [
            "pyyaml>=6.0",
            "pydantic>=1.8.0",
            "structlog>=21.0.0",
            "click>=8.0.0",
            "python-dotenv>=0.19.0",
            "aiofiles>=0.8.0",
        ]
        
        # Collect all requirements
        all_requirements = set()
        
        # Add base requirements
        for req in base_requirements:
            all_requirements.add(req)
        
        # Add plugin dependencies
        for integration in spec.integrations:
            plugin = plugin_registry.get_plugin(integration.name)
            if not plugin:
                continue
            
            # Add manifest dependencies
            for dep in plugin.manifest.dependencies:
                if dep and not dep.startswith("#"):
                    all_requirements.add(dep.strip())
        
        # Sort requirements
        def get_package_name(requirement):
            """Extract package name from requirement string."""
            import re
            match = re.match(r'^([a-zA-Z0-9_\-\[\]]+)', requirement)
            return match.group(1).lower() if match else requirement.lower()
        
        sorted_requirements = sorted(all_requirements, key=get_package_name)
        
        # Write requirements.txt
        req_file = output_dir / "requirements.txt"
        with req_file.open('w', encoding='utf-8') as f:
            f.write("# Auto-generated requirements for standalone " + spec.name + "\n")
            f.write("# Generated on " + datetime.now().isoformat() + "\n\n")
            
            # Core dependencies
            f.write("# Core dependencies\n")
            for req in sorted_requirements:
                if any(req.startswith(core) for core in ['pyyaml', 'pydantic', 'structlog', 'click', 'python-dotenv', 'aiofiles']):
                    f.write(f"{req}\n")
            
            f.write("\n# Plugin dependencies\n")
            for req in sorted_requirements:
                if not any(req.startswith(core) for core in ['pyyaml', 'pydantic', 'structlog', 'click', 'python-dotenv', 'aiofiles']):
                    f.write(f"{req}\n")
    
    def _generate_config_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate configuration files."""
        # Main config.yaml
        template = self.jinja_env.get_template('config.yaml.j2')
        content = template.render(spec=spec)
        
        config_file = output_dir / "config" / "config.yaml"
        config_file.write_text(content)
        
        # Environment template
        template = self.jinja_env.get_template('env.template.j2')
        content = template.render(spec=spec)
        
        env_file = output_dir / ".env.template"
        env_file.write_text(content)
    
    def _generate_utility_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate utility and helper files."""
        # Generate agent system for standalone operation
        self._generate_standalone_agent_system(output_dir)
        
        # Generate logging configuration
        template = self.jinja_env.get_template('utils/logging.py.j2')
        content = template.render(spec=spec)
        
        utils_dir = output_dir / "workflow" / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        (utils_dir / "__init__.py").write_text("")
        (utils_dir / "logging.py").write_text(content)
        
        # Generate helpers
        template = self.jinja_env.get_template('utils/helpers.py.j2')
        content = template.render(spec=spec)
        (utils_dir / "helpers.py").write_text(content)
    
    def _generate_standalone_agent_system(self, output_dir: Path) -> None:
        """Generate a simplified agent system for standalone workflows."""
        agent_dir = output_dir / "workflow" / "agent"
        agent_dir.mkdir(exist_ok=True)
        
        # Create simplified agent classes
        agent_content = '''"""
Simplified agent system for standalone workflows.
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Optional, Set, List, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_retries: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ResourceRequirements:
    """Resource requirements for states."""
    cpu_units: float = 0.1
    memory_mb: int = 100
    network_weight: float = 1.0
    priority: int = 1
    timeout: Optional[float] = None


class Agent:
    """Simplified agent for standalone workflow execution."""
    
    def __init__(self, name: str, max_concurrent: int = 10, state_timeout: Optional[float] = None, retry_policy: Optional[RetryPolicy] = None):
        self.name = name
        self.max_concurrent = max_concurrent
        self.state_timeout = state_timeout
        self.retry_policy = retry_policy or RetryPolicy()
        
        self.states: Dict[str, Callable] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.completed_states: Set[str] = set()
        self._running_states: Set[str] = set()
        self.status = AgentStatus.IDLE
        
        # Shared context
        from ..plugins.base import Context
        self.context = Context()
    
    def add_state(self, name: str, func: Callable, 
                  dependencies: Optional[Union[List[str], Dict[str, str]]] = None, 
                  resources: Optional[ResourceRequirements] = None, 
                  retry_policy: Optional[RetryPolicy] = None, 
                  max_retries: int = 3):
        """Add a state to the agent."""
        self.states[name] = func
        
        # Handle both list and dict formats for dependencies
        if dependencies:
            if isinstance(dependencies, dict):
                # Extract keys from dict (ignore requirement types for now)
                self.dependencies[name] = list(dependencies.keys())
            elif isinstance(dependencies, list):
                self.dependencies[name] = dependencies
            else:
                self.dependencies[name] = []
        else:
            self.dependencies[name] = []
        
        logger.debug(f"Added state: {name} with dependencies: {self.dependencies[name]}")
    
    async def run(self, timeout: Optional[float] = None) -> None:
        """Run the workflow."""
        self.status = AgentStatus.RUNNING
        logger.info(f"Starting workflow execution: {self.name}")
        
        try:
            # Find start state
            start_states = [name for name, deps in self.dependencies.items() if not deps]
            if not start_states:
                raise ValueError("No start state found (state with no dependencies)")
            
            # Execute workflow
            await self._execute_workflow(start_states[0])
            
            self.status = AgentStatus.COMPLETED
            logger.info("Workflow execution completed successfully")
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    async def run_state(self, state_name: str) -> None:
        """Run workflow starting from a specific state."""
        self.status = AgentStatus.RUNNING
        logger.info(f"Starting workflow from state: {state_name}")
        
        try:
            await self._execute_workflow(state_name)
            self.status = AgentStatus.COMPLETED
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    async def _execute_workflow(self, start_state: str) -> None:
        """Execute workflow starting from a given state."""
        current_state = start_state
        
        while current_state:
            if current_state in self.completed_states:
                break
            
            logger.info(f"Executing state: {current_state}")
            
            # Check dependencies
            deps = self.dependencies.get(current_state, [])
            for dep in deps:
                if dep not in self.completed_states:
                    raise ValueError(f"State {current_state} depends on {dep} which hasn't completed")
            
            # Execute state
            state_func = self.states.get(current_state)
            if not state_func:
                raise ValueError(f"State function not found: {current_state}")
            
            try:
                self._running_states.add(current_state)
                result = await state_func(self.context)
                self._running_states.remove(current_state)
                self.completed_states.add(current_state)
                
                # Determine next state
                if isinstance(result, str):
                    current_state = result
                else:
                    current_state = None
                    
            except Exception as e:
                self._running_states.discard(current_state)
                logger.error(f"Error executing state {current_state}: {e}")
                raise
'''
        
        (agent_dir / "__init__.py").write_text('"""Simplified agent system."""\n')
        (agent_dir / "agent.py").write_text(agent_content)
    
    def _generate_docker_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate Docker configuration."""
        # Dockerfile
        template = self.jinja_env.get_template('Dockerfile.j2')
        content = template.render(spec=spec)
        (output_dir / "Dockerfile").write_text(content)
        
        # docker-compose.yml
        template = self.jinja_env.get_template('docker-compose.yml.j2')
        content = template.render(spec=spec)
        (output_dir / "docker-compose.yml").write_text(content)
        
        # .dockerignore
        dockerignore_content = """
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.env
"""
        (output_dir / ".dockerignore").write_text(dockerignore_content.strip())
    
    def _generate_documentation(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate documentation files."""
        # README.md
        template = self.jinja_env.get_template('README.md.j2')
        content = template.render(spec=spec, plugins=self.available_plugins)
        (output_dir / "README.md").write_text(content)
        
        # API documentation
        template = self.jinja_env.get_template('docs/api.md.j2')
        content = template.render(spec=spec)
        (output_dir / "docs" / "api.md").write_text(content)
        
        # Deployment guide
        template = self.jinja_env.get_template('docs/deployment.md.j2')
        content = template.render(spec=spec)
        (output_dir / "docs" / "deployment.md").write_text(content)
    
    def _generate_tests(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate test files."""
        # Main test file
        template = self.jinja_env.get_template('tests/test_workflow.py.j2')
        content = template.render(spec=spec)
        (output_dir / "tests" / "test_workflow.py").write_text(content)
        
        # pytest configuration
        pytest_ini_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
"""
        (output_dir / "pytest.ini").write_text(pytest_ini_content)
    
    # Utility methods for Jinja filters
    def _snake_case(self, text: str) -> str:
        """Convert to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _camel_case(self, text: str) -> str:
        """Convert to camelCase."""
        components = text.replace('-', '_').split('_')
        return components[0].lower() + ''.join(x.capitalize() for x in components[1:])
    
    def _pascal_case(self, text: str) -> str:
        """Convert to PascalCase."""
        components = text.replace('-', '_').split('_')
        return ''.join(x.capitalize() for x in components)
    
    def _indent(self, text: str, width: int = 4) -> str:
        """Indent text by specified width."""
        lines = text.splitlines()
        indent = ' ' * width
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    def _quote(self, text: str) -> str:
        """Quote text for Python strings."""
        return repr(text)