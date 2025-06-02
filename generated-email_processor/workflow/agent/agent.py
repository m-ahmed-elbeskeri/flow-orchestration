"""
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
