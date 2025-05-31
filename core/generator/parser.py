# core/generator/parser.py
"""YAML workflow parser and validator with plugin discovery."""

import yaml
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

# Import plugin registry to discover available integrations
from plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error details."""
    message: str
    path: str
    severity: str = "error"  # error, warning, info
    code: str = ""


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    timeout: float = 300.0
    max_concurrent: int = 10
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        'max_retries': 3,
        'initial_delay': 1.0,
        'exponential_base': 2.0,
        'jitter': True
    })
    cleanup_timeout: float = 10.0
    state_timeout: Optional[float] = 60.0


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)


@dataclass
class IntegrationConfig:
    """Integration configuration."""
    name: str
    version: str = "1.0.0"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    required_env: List[str] = field(default_factory=list)


@dataclass
class ResourceRequirements:
    """Resource requirements for a state."""
    cpu_units: float = 0.1
    memory_mb: int = 100
    network_weight: float = 1.0
    priority: int = 1
    timeout: Optional[float] = None


@dataclass
class StateTransition:
    """State transition definition."""
    condition: str  # on_success, on_failure, on_timeout, on_retry, etc.
    target: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDependency:
    """State dependency definition."""
    name: str
    type: str = "required"  # required, optional, conditional
    condition: Optional[str] = None
    timeout: Optional[float] = None


@dataclass
class StateDefinition:
    """Workflow state definition."""
    name: str
    type: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    function: Optional[str] = None  # For custom functions
    dependencies: List[StateDependency] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    enabled: bool = True


@dataclass
class ScheduleConfig:
    """Workflow schedule configuration."""
    cron: Optional[str] = None
    timezone: str = "UTC"
    enabled: bool = True
    max_instances: int = 1


@dataclass
class WorkflowSpec:
    """Complete workflow specification."""
    name: str
    version: str
    description: str
    author: str
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    integrations: List[IntegrationConfig] = field(default_factory=list)
    states: List[StateDefinition] = field(default_factory=list)
    schedule: Optional[ScheduleConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def snake_case_name(self) -> str:
        """Get snake_case version of workflow name."""
        return re.sub(r'[^a-zA-Z0-9]', '_', self.name.lower())
    
    @property
    def class_name(self) -> str:
        """Get CamelCase class name."""
        words = re.split(r'[^a-zA-Z0-9]', self.name)
        return ''.join(word.capitalize() for word in words if word)
    
    @property
    def required_integrations(self) -> List[str]:
        """Get list of required integration names from states."""
        integrations = set()
        for state in self.states:
            if '.' in state.type and not state.type.startswith('builtin.'):
                integration_name = state.type.split('.')[0]
                integrations.add(integration_name)
        return sorted(integrations)
    
    @property
    def all_secrets(self) -> List[str]:
        """Get all required secrets."""
        secrets = set(self.environment.secrets)
        
        # Add integration-specific secrets based on available plugins
        for integration_name in self.required_integrations:
            plugin = plugin_registry.get_plugin(integration_name)
            if plugin:
                # Get required secrets from plugin manifest
                for dep in plugin.manifest.dependencies:
                    if 'credentials' in dep.lower() or 'auth' in dep.lower():
                        secrets.add(f"{integration_name.upper()}_CREDENTIALS")
                        break
                else:
                    # Default credential pattern
                    secrets.add(f"{integration_name.upper()}_TOKEN")
        
        return sorted(secrets)
    
    @property
    def available_plugins(self) -> Dict[str, Any]:
        """Get information about available plugins."""
        return {
            plugin_info['name']: plugin_info 
            for plugin_info in plugin_registry.list_plugins()
        }

class WorkflowParser:
    """Parse and validate YAML workflow definitions."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        
        # Load available plugins with error handling
        try:
            plugin_registry.load_plugins()
            self.available_plugins = plugin_registry.list_plugins()
            self.available_integrations = {
                plugin['name']: plugin for plugin in self.available_plugins
            }
            logger.info(f"Loaded {len(self.available_plugins)} plugins for validation")
        except Exception as e:
            logger.warning(f"Could not load plugins: {e}")
            self.available_plugins = []
            self.available_integrations = {}
            
            # Add a warning but don't fail
            self.warnings.append(ValidationError(
                message=f"Plugin discovery failed: {e}. Only built-in states will be available.",
                path="plugins",
                severity="warning"
            ))
    
    def parse_file(self, yaml_file: Path) -> WorkflowSpec:
        """Parse workflow from YAML file."""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except FileNotFoundError:
            raise ValueError(f"YAML file not found: {yaml_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
    
    def parse_string(self, yaml_content: str) -> WorkflowSpec:
        """Parse workflow from YAML string."""
        self.errors.clear()
        self.warnings.clear()
        
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
        
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a dictionary")
        
        # Parse main workflow info
        spec = WorkflowSpec(
            name=self._get_required_string(data, 'name'),
            version=self._get_string(data, 'version', '1.0.0'),
            description=self._get_string(data, 'description', ''),
            author=self._get_string(data, 'author', '')
        )
        
        # Parse configuration
        if 'config' in data:
            spec.config = self._parse_config(data['config'])
        
        # Parse environment
        if 'environment' in data:
            spec.environment = self._parse_environment(data['environment'])
        
        # Parse integrations - auto-discover if not specified
        if 'integrations' in data:
            spec.integrations = self._parse_integrations(data['integrations'])
        else:
            spec.integrations = self._auto_discover_integrations(data.get('states', []))
        
        # Parse states
        if 'states' in data:
            spec.states = self._parse_states(data['states'])
        
        # Parse schedule
        if 'schedule' in data:
            spec.schedule = self._parse_schedule(data['schedule'])
        
        # Parse metadata
        spec.metadata = data.get('metadata', {})
        
        # Validate the complete spec
        self._validate_spec(spec)
        
        if self.errors:
            error_messages = [f"{e.path}: {e.message}" for e in self.errors]
            raise ValueError(f"Validation errors:\n" + "\n".join(error_messages))
        
        return spec
    
    def _auto_discover_integrations(self, states_data: List[Dict[str, Any]]) -> List[IntegrationConfig]:
        """Auto-discover required integrations from states."""
        required_integrations = set()
        
        for state_data in states_data:
            state_type = state_data.get('type', '')
            if '.' in state_type and not state_type.startswith('builtin.'):
                integration_name = state_type.split('.')[0]
                required_integrations.add(integration_name)
        
        integrations = []
        for integration_name in required_integrations:
            if integration_name in self.available_integrations:
                plugin_info = self.available_integrations[integration_name]
                integrations.append(IntegrationConfig(
                    name=integration_name,
                    version=plugin_info.get('version', '1.0.0'),
                    enabled=True
                ))
            else:
                self.warnings.append(ValidationError(
                    message=f"Integration '{integration_name}' not found in available plugins",
                    path=f"auto_discovery.{integration_name}",
                    severity="warning"
                ))
        
        return integrations
    
    def _get_required_string(self, data: Dict[str, Any], key: str) -> str:
        """Get required string value."""
        if key not in data:
            self.errors.append(ValidationError(
                message=f"Required field '{key}' is missing",
                path=key
            ))
            return ""
        
        value = data[key]
        if not isinstance(value, str):
            self.errors.append(ValidationError(
                message=f"Field '{key}' must be a string",
                path=key
            ))
            return ""
        
        return value
    
    def _get_string(self, data: Dict[str, Any], key: str, default: str = "") -> str:
        """Get optional string value."""
        value = data.get(key, default)
        if not isinstance(value, str):
            self.warnings.append(ValidationError(
                message=f"Field '{key}' should be a string, got {type(value).__name__}",
                path=key,
                severity="warning"
            ))
            return str(value)
        return value
    
    def _parse_config(self, config_data: Dict[str, Any]) -> WorkflowConfig:
        """Parse workflow configuration."""
        config = WorkflowConfig()
        
        if 'timeout' in config_data:
            config.timeout = float(config_data['timeout'])
        
        if 'max_concurrent' in config_data:
            config.max_concurrent = int(config_data['max_concurrent'])
        
        if 'retry_policy' in config_data:
            retry_data = config_data['retry_policy']
            config.retry_policy.update(retry_data)
        
        if 'cleanup_timeout' in config_data:
            config.cleanup_timeout = float(config_data['cleanup_timeout'])
        
        if 'state_timeout' in config_data:
            config.state_timeout = float(config_data['state_timeout'])
        
        return config
    
    def _parse_environment(self, env_data: Dict[str, Any]) -> EnvironmentConfig:
        """Parse environment configuration."""
        env = EnvironmentConfig()
        
        if 'variables' in env_data:
            env.variables = env_data['variables']
        
        if 'secrets' in env_data:
            env.secrets = env_data['secrets']
        
        return env
    
    def _parse_integrations(self, integrations_data: List[Dict[str, Any]]) -> List[IntegrationConfig]:
        """Parse integrations configuration."""
        integrations = []
        
        for integration_data in integrations_data:
            name = integration_data.get('name')
            if not name:
                self.errors.append(ValidationError(
                    message="Integration name is required",
                    path="integrations"
                ))
                continue
            
            # Check if integration is available
            if name not in self.available_integrations:
                self.warnings.append(ValidationError(
                    message=f"Integration '{name}' not found in available plugins",
                    path=f"integrations.{name}",
                    severity="warning"
                ))
            
            integration = IntegrationConfig(
                name=name,
                version=integration_data.get('version', '1.0.0'),
                config=integration_data.get('config', {}),
                enabled=integration_data.get('enabled', True)
            )
            
            integrations.append(integration)
        
        return integrations
    
    def _parse_states(self, states_data: List[Dict[str, Any]]) -> List[StateDefinition]:
        """Parse states configuration."""
        states = []
        
        for state_data in states_data:
            name = state_data.get('name')
            state_type = state_data.get('type')
            
            if not name:
                self.errors.append(ValidationError(
                    message="State name is required",
                    path="states"
                ))
                continue
            
            if not state_type:
                self.errors.append(ValidationError(
                    message=f"State '{name}' is missing type",
                    path=f"states.{name}.type"
                ))
                continue
            
            # Validate integration state types
            if '.' in state_type and not state_type.startswith('builtin.'):
                integration_name, action_name = state_type.split('.', 1)
                if integration_name in self.available_integrations:
                    plugin_info = self.available_integrations[integration_name]
                    available_states = plugin_info.get('states', [])
                    if action_name not in available_states:
                        self.warnings.append(ValidationError(
                            message=f"Action '{action_name}' not found in plugin '{integration_name}'. Available: {available_states}",
                            path=f"states.{name}.type",
                            severity="warning"
                        ))
            
            # Parse dependencies
            dependencies = []
            for dep in state_data.get('dependencies', []):
                if isinstance(dep, str):
                    dependencies.append(StateDependency(name=dep))
                elif isinstance(dep, dict):
                    dependencies.append(StateDependency(
                        name=dep['name'],
                        type=dep.get('type', 'required'),
                        condition=dep.get('condition'),
                        timeout=dep.get('timeout')
                    ))
            
            # Parse transitions
            transitions = []
            for trans in state_data.get('transitions', []):
                if isinstance(trans, dict):
                    for condition, target in trans.items():
                        transitions.append(StateTransition(
                            condition=condition,
                            target=target
                        ))
            
            # Parse resources
            resources = ResourceRequirements()
            if 'resources' in state_data:
                res_data = state_data['resources']
                if 'cpu_units' in res_data:
                    resources.cpu_units = float(res_data['cpu_units'])
                if 'memory_mb' in res_data:
                    resources.memory_mb = int(res_data['memory_mb'])
                if 'network_weight' in res_data:
                    resources.network_weight = float(res_data['network_weight'])
                if 'priority' in res_data:
                    resources.priority = int(res_data['priority'])
                if 'timeout' in res_data:
                    resources.timeout = float(res_data['timeout'])
            
            state = StateDefinition(
                name=name,
                type=state_type,
                description=state_data.get('description', ''),
                config=state_data.get('config', {}),
                function=state_data.get('function'),
                dependencies=dependencies,
                transitions=transitions,
                resources=resources,
                retry_policy=state_data.get('retry_policy'),
                timeout=state_data.get('timeout'),
                enabled=state_data.get('enabled', True)
            )
            
            states.append(state)
        
        return states
    
    def _parse_schedule(self, schedule_data: Dict[str, Any]) -> ScheduleConfig:
        """Parse schedule configuration."""
        return ScheduleConfig(
            cron=schedule_data.get('cron'),
            timezone=schedule_data.get('timezone', 'UTC'),
            enabled=schedule_data.get('enabled', True),
            max_instances=schedule_data.get('max_instances', 1)
        )
    
    def _validate_spec(self, spec: WorkflowSpec) -> None:
        """Validate the complete workflow specification."""
        # Check for duplicate state names
        state_names = [state.name for state in spec.states]
        duplicates = set([name for name in state_names if state_names.count(name) > 1])
        for duplicate in duplicates:
            self.errors.append(ValidationError(
                message=f"Duplicate state name: {duplicate}",
                path=f"states.{duplicate}"
            ))
        
        # Check dependencies reference valid states
        valid_states = set(state_names)
        for state in spec.states:
            for dep in state.dependencies:
                if dep.name not in valid_states:
                    self.errors.append(ValidationError(
                        message=f"State '{state.name}' depends on unknown state '{dep.name}'",
                        path=f"states.{state.name}.dependencies"
                    ))
        
        # Check transitions reference valid states
        for state in spec.states:
            for trans in state.transitions:
                if trans.target not in valid_states:
                    self.errors.append(ValidationError(
                        message=f"State '{state.name}' transitions to unknown state '{trans.target}'",
                        path=f"states.{state.name}.transitions"
                    ))
        
        # Check for start and end states
        has_start = any(state.type == 'builtin.start' for state in spec.states)
        has_end = any(state.type == 'builtin.end' for state in spec.states)
        
        if not has_start:
            self.warnings.append(ValidationError(
                message="Workflow should have a start state (builtin.start)",
                path="states",
                severity="warning"
            ))
        
        if not has_end:
            self.warnings.append(ValidationError(
                message="Workflow should have an end state (builtin.end)",
                path="states",
                severity="warning"
            ))
        
        # Check for unreachable states (simple check)
        if has_start:
            start_states = [state for state in spec.states if state.type == 'builtin.start']
            if start_states:
                reachable = self._find_reachable_states(start_states[0], spec.states)
                unreachable = set(state_names) - reachable
                for state_name in unreachable:
                    self.warnings.append(ValidationError(
                        message=f"State '{state_name}' may be unreachable",
                        path=f"states.{state_name}",
                        severity="warning"
                    ))
    
    def _find_reachable_states(self, start_state: StateDefinition, all_states: List[StateDefinition]) -> set:
        """Find all states reachable from start state."""
        state_map = {state.name: state for state in all_states}
        reachable = set()
        to_visit = [start_state.name]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            
            reachable.add(current)
            
            if current in state_map:
                state = state_map[current]
                for trans in state.transitions:
                    if trans.target not in reachable:
                        to_visit.append(trans.target)
        
        return reachable


def parse_workflow_file(yaml_file: Path) -> WorkflowSpec:
    """Convenience function to parse a workflow file."""
    parser = WorkflowParser()
    return parser.parse_file(yaml_file)


def parse_workflow_string(yaml_content: str) -> WorkflowSpec:
    """Convenience function to parse a workflow string."""
    parser = WorkflowParser()
    return parser.parse_string(yaml_content)