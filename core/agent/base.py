import asyncio
import heapq
import time
import logging
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import uuid

from .state import StateStatus, AgentStatus, StateMetadata, PrioritizedState, Priority
from .context import Context
from .dependencies import DependencyConfig, DependencyLifecycle
from ..resources.requirements import ResourceRequirements

logger = logging.getLogger(__name__)

@dataclass
class RetryPolicy:
    max_retries: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    jitter: bool = True

    async def wait(self, attempt: int) -> None:
        import random
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            60.0  # Max 60 seconds
        )
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        await asyncio.sleep(delay)

class Agent:
    def __init__(
        self,
        name: str,
        max_concurrent: int = 5,
        retry_policy: Optional[RetryPolicy] = None,
        state_timeout: Optional[float] = None,
        resource_pool=None
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_timeout = state_timeout
        self.resource_pool = resource_pool
        
        # Core state management
        self.states: Dict[str, Callable] = {}
        self.state_metadata: Dict[str, StateMetadata] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.priority_queue: List[PrioritizedState] = []
        self.shared_state: Dict[str, Any] = {}
        
        # Execution tracking
        self._running_states: Set[str] = set()
        self.completed_states: Set[str] = set()
        self.completed_once: Set[str] = set()
        self.status = AgentStatus.IDLE
        self.session_start: Optional[float] = None
        
        # Context for state execution
        self.context = Context(self.shared_state)

    def add_state(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        resources: Optional[ResourceRequirements] = None,
        max_retries: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        priority: Priority = Priority.NORMAL
    ) -> None:
        """Add a state to the agent"""
        self.states[name] = func
        
        metadata = StateMetadata(
            status=StateStatus.PENDING,
            max_retries=max_retries or self.retry_policy.max_retries,
            resources=resources or ResourceRequirements(),
            retry_policy=retry_policy or self.retry_policy,
            priority=priority
        )
        self.state_metadata[name] = metadata
        
        # Set up dependencies
        self.dependencies[name] = dependencies or []

    def create_checkpoint(self):
        """Create a checkpoint of current agent state"""
        from .checkpoint import AgentCheckpoint
        return AgentCheckpoint.create_from_agent(self)

    async def restore_from_checkpoint(self, checkpoint) -> None:
        """Restore agent from checkpoint"""
        self.status = checkpoint.agent_status
        self.priority_queue = checkpoint.priority_queue.copy()
        self.state_metadata = checkpoint.state_metadata.copy()
        self._running_states = checkpoint.running_states.copy()
        self.completed_states = checkpoint.completed_states.copy()
        self.completed_once = checkpoint.completed_once.copy()
        self.shared_state = checkpoint.shared_state.copy()
        self.session_start = checkpoint.session_start

    async def pause(self):
        """Pause the agent and return checkpoint"""
        self.status = AgentStatus.PAUSED
        return self.create_checkpoint()

    async def resume(self) -> None:
        """Resume the agent"""
        self.status = AgentStatus.RUNNING

    async def run(self, timeout: Optional[float] = None) -> None:
        """Run workflow starting from entry point states"""
        if not self.states:
            logger.info("No states defined, nothing to run")
            return
        
        self.status = AgentStatus.RUNNING
        self.session_start = time.time()
        
        # Find entry point states (states without dependencies)
        entry_states = self._find_entry_states()
        if not entry_states:
            logger.warning("No entry point states found, using first state")
            entry_states = [next(iter(self.states.keys()))]
        
        logger.info(f"Starting workflow with entry states: {entry_states}")
        
        # Add entry states to queue
        for state_name in entry_states:
            await self._add_to_queue(state_name)
        
        try:
            # Main execution loop
            start_time = time.time()
            while (self.priority_queue and 
                   self.status == AgentStatus.RUNNING and
                   (not timeout or time.time() - start_time < timeout)):
                
                ready_states = await self._get_ready_states()
                if not ready_states:
                    # No ready states, check if we're waiting for running states
                    if self._running_states:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        # No running states and no ready states - we're done
                        break
                
                # Process ready states (up to max_concurrent)
                tasks = []
                for state_name in ready_states[:self.max_concurrent]:
                    task = asyncio.create_task(self.run_state(state_name))
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
            
            # Mark as completed if we finished normally
            if self.status == AgentStatus.RUNNING:
                self.status = AgentStatus.COMPLETED
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.status = AgentStatus.FAILED
            raise

    def _find_entry_states(self) -> List[str]:
        """Find states that can be entry points (no dependencies)"""
        entry_states = []
        for state_name in self.states.keys():
            deps = self.dependencies.get(state_name, [])
            if not deps:
                entry_states.append(state_name)
        return entry_states

    async def _add_to_queue(self, state_name: str, priority_boost: int = 0) -> None:
        """Add state to execution queue"""
        if state_name not in self.state_metadata:
            logger.error(f"State {state_name} not found in metadata")
            return
        
        metadata = self.state_metadata[state_name]
        priority_value = metadata.priority.value + priority_boost
        
        heapq.heappush(
            self.priority_queue,
            PrioritizedState(
                priority=-priority_value,  # Negative for max heap behavior
                timestamp=time.time(),
                state_name=state_name,
                metadata=metadata
            )
        )

    async def _get_ready_states(self) -> List[str]:
        """Get states that are ready to run"""
        ready_states = []
        temp_queue = []
        
        while self.priority_queue:
            state = heapq.heappop(self.priority_queue)
            
            if await self._can_run(state.state_name):
                ready_states.append(state.state_name)
            else:
                temp_queue.append(state)
        
        # Put non-ready states back in queue
        for state in temp_queue:
            heapq.heappush(self.priority_queue, state)
        
        return ready_states

    async def run_state(self, state_name: str) -> None:
        """Execute a single state"""
        if state_name in self._running_states:
            return
        
        self._running_states.add(state_name)
        metadata = self.state_metadata[state_name]
        metadata.status = StateStatus.RUNNING
        
        logger.info(f"Starting state: {state_name}")
        
        try:
            # Resolve dependencies
            await self._resolve_dependencies(state_name)
            
            # Execute the state function
            context = Context(self.shared_state)
            start_time = time.time()
            
            result = await self.states[state_name](context)
            
            duration = time.time() - start_time
            logger.info(f"State {state_name} completed in {duration:.2f}s")
            
            # Update metadata
            metadata.status = StateStatus.COMPLETED
            metadata.last_execution = time.time()
            metadata.last_success = time.time()
            
            # Track completion
            self.completed_states.add(state_name)
            self.completed_once.add(state_name)
            
            # Handle transitions/next states
            await self._handle_state_result(state_name, result)
            
        except Exception as error:
            logger.error(f"State {state_name} failed: {error}")
            await self._handle_failure(state_name, error)
        finally:
            self._running_states.discard(state_name)

    async def _handle_state_result(self, state_name: str, result: Any) -> None:
        """Handle the result of state execution"""
        if result is None:
            return
        
        if isinstance(result, str):
            # Result is a next state name
            if result in self.states:
                await self._add_to_queue(result)
        elif isinstance(result, list):
            # Result is a list of next states
            for next_state in result:
                if isinstance(next_state, str) and next_state in self.states:
                    await self._add_to_queue(next_state)

    async def _handle_failure(self, state_name: str, error: Exception) -> None:
        """Handle state execution failure"""
        metadata = self.state_metadata[state_name]
        metadata.attempts += 1
        
        if metadata.attempts < metadata.max_retries:
            logger.info(f"Retrying state {state_name} (attempt {metadata.attempts + 1})")
            metadata.status = StateStatus.PENDING
            await metadata.retry_policy.wait(metadata.attempts)
            await self._add_to_queue(state_name)
        else:
            logger.error(f"State {state_name} failed after {metadata.attempts} attempts")
            metadata.status = StateStatus.FAILED
            
            # Try compensation if available
            compensation_state = f"{state_name}_compensation"
            if compensation_state in self.states:
                await self._add_to_queue(compensation_state)

    async def _resolve_dependencies(self, state_name: str) -> None:
        """Resolve state dependencies"""
        deps = self.dependencies.get(state_name, [])
        for dep_name in deps:
            if dep_name not in self.completed_states:
                # This shouldn't happen if _can_run works correctly
                logger.warning(f"Dependency {dep_name} not satisfied for {state_name}")

    async def _can_run(self, state_name: str) -> bool:
        """Check if state can run (dependencies satisfied)"""
        if state_name in self._running_states:
            return False
        
        if state_name in self.completed_once:
            # Check if this is a repeatable state
            return False
        
        # Check dependencies
        deps = self.dependencies.get(state_name, [])
        for dep_name in deps:
            if dep_name not in self.completed_states:
                return False
        
        return True

    def cancel_state(self, state_name: str) -> None:
        """Cancel a specific state"""
        if state_name in self.state_metadata:
            self.state_metadata[state_name].status = StateStatus.CANCELLED
        self._running_states.discard(state_name)

    async def cancel_all(self) -> None:
        """Cancel all running and pending states"""
        self.status = AgentStatus.CANCELLED
        for state_name in self._running_states.copy():
            self.cancel_state(state_name)
        self.priority_queue.clear() 