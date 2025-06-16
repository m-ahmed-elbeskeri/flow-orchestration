import yaml
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging

from plugins.registry import plugin_registry

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    message: str
    path: str
    severity: str = "error"
    code: str = ""

@dataclass
class WorkflowConfig:
    timeout: float = 300.0
    max_concurrent: int = 10
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "initial_delay": 2.0,
        "exponential_base": 2.0,
        "jitter": True
    })
    cleanup_timeout: float = 10.0
    state_timeout: Optional[float] = 60.0

@dataclass
class EnvironmentConfig:
    variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)

@dataclass
class IntegrationConfig:
    name: str
    version: str = "1.0.0"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    required_env: List[str] = field(default_factory=list)

@dataclass
class ResourceRequirements:
    cpu_units: float = 0.1
    memory_mb: int = 100
    network_weight: float = 1.0
    priority: int = 1
    timeout: Optional[float] = None

@dataclass
class StateTransition:
    condition: str
    target: str
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateDependency:
    name: str
    type: str = "required"
    condition: Optional[str] = None

@dataclass
class StateDefinition:
    name: str
    type: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    function: Optional[str] = None
    dependencies: List[StateDependency] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    retry_policy: Optional[Dict[str, Any]] = None

@dataclass
class ScheduleConfig:
    cron: Optional[str] = None
    timezone: str = "UTC"
    enabled: bool = True
    max_instances: int = 1

@dataclass
class WorkflowSpec:
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
        """Convert name to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', self.name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @property
    def class_name(self) -> str:
        """Convert name to PascalCase for class names"""
        words = re.split(r'[^a-zA-Z0-9]', self.name)
        return ''.join(word.capitalize() for word in words if word)

    @property
    def required_integrations(self) -> List[str]:
        """Get list of required integrations from states"""
        integrations = set()
        for state in self.states:
            if '.' in state.type and not state.type.startswith('builtin.'):
                integration_name = state.type.split('.')[0]
                integrations.add(integration_name)
        return list(integrations)

    @property
    def all_secrets(self) -> List[str]:
        """Get all secrets from environment and integrations"""
        secrets = set(self.environment.secrets)
        
        # Add secrets from integration states
        for state in self.states:
            if '.' in state.type and not state.type.startswith('builtin.'):
                integration_name = state.type.split('.')[0]
                try:
                    plugin = plugin_registry.get_plugin(integration_name)
                    if plugin and hasattr(plugin, 'manifest'):
                        # Extract secrets from plugin manifest
                        pass  # Implementation depends on plugin structure
                except Exception:
                    pass
        
        return list(secrets)

    @property
    def available_plugins(self) -> Dict[str, Any]:
        return plugin_registry.list_plugins()

class WorkflowParser:
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.available_integrations = self._get_available_integrations()

    def _get_available_integrations(self) -> Dict[str, Dict[str, Any]]:
        """Get available integrations from plugin registry"""
        integrations = {}
        try:
            plugins = plugin_registry.list_plugins()
            for plugin_info in plugins:
                integrations[plugin_info['name']] = plugin_info
        except Exception as e:
            logger.warning(f"Failed to load available integrations: {e}")
        return integrations

    def parse_file(self, yaml_file: Path) -> WorkflowSpec:
        """Parse workflow from YAML file"""
        with open(yaml_file, 'r') as f:
            content = f.read()
        return self.parse_string(content)

    def parse_string(self, yaml_content: str) -> WorkflowSpec:
        """Parse workflow from YAML string"""
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

        # Clear previous errors
        self.errors.clear()

        # Parse main workflow info
        spec = WorkflowSpec(
            name=self._get_required_string(data, 'name'),
            version=self._get_string(data, 'version', '1.0.0'),
            description=self._get_string(data, 'description'),
            author=self._get_string(data, 'author')
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
            # Auto-discover from states
            states_data = data.get('states', [])
            spec.integrations = self._auto_discover_integrations(states_data)

        # Parse states (no requirement for explicit start/end)
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
        """Auto-discover required integrations from states"""
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
                    config=plugin_info.get('config', {}),
                    enabled=True
                ))
            else:
                self.errors.append(ValidationError(
                    message=f"Unknown integration: {integration_name}",
                    path=f"states.{integration_name}",
                    severity="error"
                ))

        return integrations

    def _get_required_string(self, data: Dict[str, Any], key: str) -> str:
        if key not in data:
            raise ValueError(f"Required field '{key}' is missing")
        value = data[key]
        if not isinstance(value, str):
            raise ValueError(f"Field '{key}' must be a string")
        return value

    def _get_string(self, data: Dict[str, Any], key: str, default: str = "") -> str:
        value = data.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"Field '{key}' must be a string")
        return value

    def _parse_config(self, config_data: Dict[str, Any]) -> WorkflowConfig:
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
        env = EnvironmentConfig()
        
        if 'variables' in env_data:
            env.variables = dict(env_data['variables'])
        if 'secrets' in env_data:
            env.secrets = list(env_data['secrets'])
            
        return env

    def _parse_integrations(self, integrations_data: List[Dict[str, Any]]) -> List[IntegrationConfig]:
        integrations = []
        
        for integration_data in integrations_data:
            name = integration_data.get('name')
            if not name:
                self.errors.append(ValidationError(
                    message="Integration name is required",
                    path="integrations",
                    severity="error"
                ))
                continue

            # Check if integration is available
            if name not in self.available_integrations:
                self.errors.append(ValidationError(
                    message=f"Unknown integration: {name}",
                    path=f"integrations.{name}",
                    severity="warning"
                ))

            integration = IntegrationConfig(
                name=name,
                version=integration_data.get('version', '1.0.0'),
                config=integration_data.get('config', {}),
                enabled=integration_data.get('enabled', True),
                required_env=integration_data.get('required_env', [])
            )
            integrations.append(integration)
            
        return integrations

    def _parse_states(self, states_data: List[Dict[str, Any]]) -> List[StateDefinition]:
        states = []
        
        for state_data in states_data:
            name = state_data.get('name')
            state_type = state_data.get('type')
            
            if not name:
                self.errors.append(ValidationError(
                    message="State name is required",
                    path="states",
                    severity="error"
                ))
                continue
                
            if not state_type:
                self.errors.append(ValidationError(
                    message=f"State type is required for state '{name}'",
                    path=f"states.{name}",
                    severity="error"
                ))
                continue

            # Validate integration state types
            if '.' in state_type and not state_type.startswith('builtin.'):
                integration_name = state_type.split('.')[0]
                if integration_name in self.available_integrations:
                    plugin_info = self.available_integrations[integration_name]
                    available_states = plugin_info.get('states', [])
                    state_name_part = state_type.split('.')[1]
                    if state_name_part not in available_states:
                        self.errors.append(ValidationError(
                            message=f"Unknown state type '{state_type}' for integration '{integration_name}'",
                            path=f"states.{name}.type",
                            severity="error"
                        ))

            # Parse dependencies
            dependencies = []
            if 'dependencies' in state_data:
                for dep_data in state_data['dependencies']:
                    if isinstance(dep_data, str):
                        dependencies.append(StateDependency(name=dep_data))
                    elif isinstance(dep_data, dict):
                        dependencies.append(StateDependency(
                            name=dep_data.get('name', ''),
                            type=dep_data.get('type', 'required'),
                            condition=dep_data.get('condition')
                        ))

            # Parse transitions
            transitions = []
            if 'transitions' in state_data:
                for trans_data in state_data['transitions']:
                    if isinstance(trans_data, dict):
                        for condition, target in trans_data.items():
                            if condition.startswith('on_'):
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
                retry_policy=state_data.get('retry_policy')
            )
            states.append(state)
            
        return states

    def _parse_schedule(self, schedule_data: Dict[str, Any]) -> ScheduleConfig:
        return ScheduleConfig(
            cron=schedule_data.get('cron'),
            timezone=schedule_data.get('timezone', 'UTC'),
            enabled=schedule_data.get('enabled', True),
            max_instances=schedule_data.get('max_instances', 1)
        )

    def _validate_spec(self, spec: WorkflowSpec) -> None:
        """Validate the complete workflow specification"""
        
        # Check for duplicate state names
        state_names = [state.name for state in spec.states]
        duplicates = set([name for name in state_names if state_names.count(name) > 1])
        for duplicate in duplicates:
            self.errors.append(ValidationError(
                message=f"Duplicate state name: {duplicate}",
                path=f"states.{duplicate}",
                severity="error"
            ))

        # Check dependencies reference valid states
        valid_states = set(state_names)
        for state in spec.states:
            for dep in state.dependencies:
                if dep.name not in valid_states:
                    self.errors.append(ValidationError(
                        message=f"State '{state.name}' depends on unknown state '{dep.name}'",
                        path=f"states.{state.name}.dependencies",
                        severity="error"
                    ))

        # Check transitions reference valid states
        for state in spec.states:
            for transition in state.transitions:
                if transition.target not in valid_states:
                    self.errors.append(ValidationError(
                        message=f"State '{state.name}' transitions to unknown state '{transition.target}'",
                        path=f"states.{state.name}.transitions",
                        severity="error"
                    ))

        # Workflow can start without explicit start state - just needs states without dependencies
        entry_states = [state for state in spec.states if not state.dependencies]
        if not entry_states and spec.states:
            self.errors.append(ValidationError(
                message="No entry point states found (states without dependencies). At least one state should have no dependencies.",
                path="states",
                severity="warning"
            ))

        # Check for unreachable states (simple check)
        if entry_states and len(spec.states) > 1:
            reachable = self._find_reachable_states(entry_states[0], spec.states)
            unreachable = set(state_names) - reachable
            for state_name in unreachable:
                self.errors.append(ValidationError(
                    message=f"State '{state_name}' may be unreachable",
                    path=f"states.{state_name}",
                    severity="warning"
                ))

    def _find_reachable_states(self, start_state: StateDefinition, all_states: List[StateDefinition]) -> set:
        """Find all states reachable from start state"""
        state_map = {state.name: state for state in all_states}
        reachable = set()
        to_visit = [start_state.name]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            reachable.add(current)
            
            state = state_map[current]
            for transition in state.transitions:
                if transition.target not in reachable:
                    to_visit.append(transition.target)
        
        return reachable

def parse_workflow_file(yaml_file: Path) -> WorkflowSpec:
    """Convenience function to parse workflow from file"""
    parser = WorkflowParser()
    return parser.parse_file(yaml_file)

def parse_workflow_string(yaml_content: str) -> WorkflowSpec:
    """Convenience function to parse workflow from string"""
    parser = WorkflowParser()
    return parser.parse_string(yaml_content)