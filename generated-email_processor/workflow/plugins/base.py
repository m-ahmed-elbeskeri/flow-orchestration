"""Standalone plugin system base classes."""

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
