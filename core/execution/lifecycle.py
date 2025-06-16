"""
Comprehensive lifecycle management system for workflow execution.
Provides hooks, monitoring, validation, and state management throughout the execution lifecycle.
"""

import asyncio
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union, Type
from collections import defaultdict
import inspect
import functools

import structlog

from core.agent.base import Agent
from core.agent.state import StateFunction, StateMetadata, StateStatus
from core.agent.context import Context
from core.agent.checkpoint import AgentCheckpoint
from core.resources.requirements import ResourceRequirements

logger = structlog.get_logger(__name__)

class HookType(Enum):
    """Types of lifecycle hooks available"""
    # Workflow level hooks
    BEFORE_WORKFLOW = "before_workflow"
    AFTER_WORKFLOW = "after_workflow"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    
    # State level hooks
    BEFORE_STATE = "before_state"
    AFTER_STATE = "after_state"
    STATE_SCHEDULED = "state_scheduled"
    STATE_STARTED = "state_started"
    STATE_COMPLETED = "state_completed"
    STATE_FAILED = "state_failed"
    STATE_CANCELLED = "state_cancelled"
    STATE_SKIPPED = "state_skipped"
    STATE_TIMEOUT = "state_timeout"
    
    # Retry and error handling
    BEFORE_RETRY = "before_retry"
    AFTER_RETRY = "after_retry"
    BEFORE_COMPENSATION = "before_compensation"
    AFTER_COMPENSATION = "after_compensation"
    RETRY_EXHAUSTED = "retry_exhausted"
    
    # Transition hooks
    BEFORE_TRANSITION = "before_transition"
    AFTER_TRANSITION = "after_transition"
    TRANSITION_FAILED = "transition_failed"
    
    # Success/failure hooks
    AFTER_SUCCESS = "after_success"
    AFTER_FAILURE = "after_failure"
    
    # Resource management hooks
    BEFORE_RESOURCE_ACQUIRE = "before_resource_acquire"
    AFTER_RESOURCE_ACQUIRE = "after_resource_acquire"
    RESOURCE_ACQUIRE_FAILED = "resource_acquire_failed"
    BEFORE_RESOURCE_RELEASE = "before_resource_release"
    AFTER_RESOURCE_RELEASE = "after_resource_release"
    RESOURCE_PREEMPTED = "resource_preempted"
    RESOURCE_QUOTA_EXCEEDED = "resource_quota_exceeded"
    
    # Dependency hooks
    DEPENDENCY_SATISFIED = "dependency_satisfied"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    DEPENDENCY_TIMEOUT = "dependency_timeout"
    
    # Checkpoint and persistence hooks
    BEFORE_CHECKPOINT = "before_checkpoint"
    AFTER_CHECKPOINT = "after_checkpoint"
    CHECKPOINT_FAILED = "checkpoint_failed"
    BEFORE_RESTORE = "before_restore"
    AFTER_RESTORE = "after_restore"
    RESTORE_FAILED = "restore_failed"
    
    # Monitoring and observability
    METRICS_COLLECTED = "metrics_collected"
    ALERT_TRIGGERED = "alert_triggered"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_THRESHOLD = "performance_threshold"

class HookPriority(Enum):
    """Priority levels for hook execution"""
    CRITICAL = 0    # Must execute first (e.g., security, validation)
    HIGH = 1        # Important hooks (e.g., logging, metrics)
    NORMAL = 2      # Standard hooks (e.g., notifications)
    LOW = 3         # Optional hooks (e.g., debugging, analytics)

@dataclass
class HookContext:
    """Complete context information for lifecycle hooks"""
    hook_type: HookType
    state_name: str
    agent_name: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # State and agent information
    metadata: Optional[StateMetadata] = None
    context: Optional[Context] = None
    agent: Optional[Agent] = None
    
    # Execution information
    error: Optional[Exception] = None
    result: Any = None
    retry_count: int = 0
    attempt_count: int = 0
    duration: Optional[float] = None
    
    # Resource information
    resources_requested: Optional[ResourceRequirements] = None
    resources_allocated: Optional[Dict[str, float]] = None
    resource_pool_status: Optional[Dict[str, Any]] = None
    
    # Dependency information
    dependencies: List[str] = field(default_factory=list)
    satisfied_dependencies: List[str] = field(default_factory=list)
    blocked_dependencies: List[str] = field(default_factory=list)
    
    # Transition information
    previous_state: Optional[str] = None
    next_state: Optional[str] = None
    transition_condition: Optional[str] = None
    
    # Checkpoint information
    checkpoint: Optional[AgentCheckpoint] = None
    checkpoint_id: Optional[str] = None
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    network_io: Optional[float] = None
    
    # Custom data
    extra: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'execution_id': self.execution_id,
            'hook_type': self.hook_type.value,
            'state_name': self.state_name,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'attempt_count': self.attempt_count,
            'duration': self.duration,
            'has_error': self.error is not None,
            'error_type': type(self.error).__name__ if self.error else None,
            'error_message': str(self.error) if self.error else None,
            'previous_state': self.previous_state,
            'next_state': self.next_state,
            'transition_condition': self.transition_condition,
            'dependencies': self.dependencies,
            'satisfied_dependencies': self.satisfied_dependencies,
            'blocked_dependencies': self.blocked_dependencies,
            'checkpoint_id': self.checkpoint_id,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_io': self.network_io,
            'extra': self.extra,
            'tags': list(self.tags)
        }
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the context"""
        self.tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if context has a specific tag"""
        return tag in self.tags
    
    def set_error(self, error: Exception) -> None:
        """Set error information"""
        self.error = error
        self.add_tag("error")
    
    def set_performance_metrics(self, cpu: float, memory: float, network: float) -> None:
        """Set performance metrics"""
        self.cpu_usage = cpu
        self.memory_usage = memory
        self.network_io = network

# Context variables for hook execution
_current_hook_context: contextvars.ContextVar[Optional[HookContext]] = \
    contextvars.ContextVar('hook_context', default=None)
_hook_execution_stack: contextvars.ContextVar[Optional[List[str]]] = \
    contextvars.ContextVar('hook_stack', default=None)

class LifecycleHook:
    """Base class for all lifecycle hooks"""
    
    def __init__(
        self, 
        name: str, 
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        tags: Optional[Set[str]] = None
    ):
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.timeout = timeout or 30.0  # Default 30 second timeout
        self.retry_count = retry_count
        self.tags = tags or set()
        
        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_executed: Optional[datetime] = None
        self.last_error: Optional[Exception] = None
        
        # Hook filters
        self._hook_types: Set[HookType] = set()
        self._state_names: Set[str] = set()
        self._agent_names: Set[str] = set()
        self._condition: Optional[Callable[[HookContext], bool]] = None

    def set_filters(
        self,
        hook_types: Optional[Union[HookType, List[HookType]]] = None,
        state_names: Optional[Union[str, List[str]]] = None,
        agent_names: Optional[Union[str, List[str]]] = None,
        condition: Optional[Callable[[HookContext], bool]] = None
    ) -> None:
        """Set filters for when this hook should execute"""
        if hook_types:
            if isinstance(hook_types, HookType):
                self._hook_types = {hook_types}
            else:
                self._hook_types = set(hook_types)
        
        if state_names:
            if isinstance(state_names, str):
                self._state_names = {state_names}
            else:
                self._state_names = set(state_names)
        
        if agent_names:
            if isinstance(agent_names, str):
                self._agent_names = {agent_names}
            else:
                self._agent_names = set(agent_names)
        
        self._condition = condition

    def should_execute(self, context: HookContext) -> bool:
        """Check if this hook should execute for the given context"""
        if not self.enabled:
            return False
        
        # Check hook type filter
        if self._hook_types and context.hook_type not in self._hook_types:
            return False
        
        # Check state name filter
        if self._state_names and context.state_name not in self._state_names:
            return False
        
        # Check agent name filter
        if self._agent_names and context.agent_name not in self._agent_names:
            return False
        
        # Check custom condition
        if self._condition and not self._condition(context):
            return False
        
        return True

    async def execute(self, context: HookContext) -> None:
        """Execute the hook with full error handling and metrics"""
        if not self.should_execute(context):
            return
        
        start_time = time.time()
        self.execution_count += 1
        self.last_executed = datetime.utcnow()
        
        # Check for circular execution
        stack = _hook_execution_stack.get()
        if stack is None:
            stack = []
            
        if self.name in stack:
            logger.warning(f"Circular hook execution detected: {self.name}")
            return

        # Add to execution stack
        stack = stack.copy()  # Create a copy to avoid mutating shared state
        stack.append(self.name)
        _hook_execution_stack.set(stack)
        
        try:
            # Execute with timeout
            await asyncio.wait_for(self._execute_with_retry(context), timeout=self.timeout)
            self.success_count += 1
            
        except asyncio.TimeoutError:
            error = TimeoutError(f"Hook {self.name} timed out after {self.timeout}s")
            self.last_error = error
            self.failure_count += 1
            logger.error(f"Hook {self.name} timed out", timeout=self.timeout)
            
        except Exception as e:
            self.last_error = e
            self.failure_count += 1
            logger.error(f"Hook {self.name} failed: {e}", exc_info=True)
            
        finally:
            # Remove from execution stack
            stack = _hook_execution_stack.get()
            if stack is not None and self.name in stack:
                stack = stack.copy()  # Create a copy
                stack.remove(self.name)
                _hook_execution_stack.set(stack)
            
            # Update timing statistics
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

    async def _execute_with_retry(self, context: HookContext) -> None:
        """Execute with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_count + 1):
            try:
                await self._execute(context)
                return  # Success
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Hook {self.name} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise last_error

    async def _execute(self, context: HookContext) -> None:
        """Override this method in subclasses"""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for this hook"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        success_rate = (
            self.success_count / self.execution_count * 100 
            if self.execution_count > 0 else 0
        )
        
        return {
            'name': self.name,
            'priority': self.priority.name,
            'enabled': self.enabled,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': avg_execution_time,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'last_error': str(self.last_error) if self.last_error else None,
            'tags': list(self.tags)
        }

class LoggingHook(LifecycleHook):
    """Comprehensive logging hook with structured logging"""
    
    def __init__(
        self, 
        name: str = "logging_hook", 
        level: int = logging.INFO,
        include_context: bool = True,
        include_metrics: bool = True,
        log_format: str = "structured"
    ):
        super().__init__(name, HookPriority.HIGH)
        self.level = level
        self.include_context = include_context
        self.include_metrics = include_metrics
        self.log_format = log_format

    async def _execute(self, context: HookContext) -> None:
        """Log lifecycle events with full context"""
        base_message = f"Lifecycle event: {context.hook_type.value}"
        
        log_data = {
            'event_type': 'lifecycle',
            'hook_type': context.hook_type.value,
            'agent': context.agent_name,
            'state': context.state_name,
            'execution_id': context.execution_id,
            'timestamp': context.timestamp.isoformat(),
        }
        
        # Add timing information
        if context.duration is not None:
            log_data['duration_ms'] = round(context.duration * 1000, 2)
        
        # Add retry information
        if context.retry_count > 0:
            log_data['retry_count'] = context.retry_count
            log_data['attempt_count'] = context.attempt_count
        
        # Add error information
        if context.error:
            log_data.update({
                'error_type': type(context.error).__name__,
                'error_message': str(context.error),
                'has_error': True
            })
        
        # Add transition information
        if context.previous_state:
            log_data['previous_state'] = context.previous_state
        if context.next_state:
            log_data['next_state'] = context.next_state
        if context.transition_condition:
            log_data['transition_condition'] = context.transition_condition
        
        # Add dependency information
        if context.dependencies:
            log_data['dependencies'] = context.dependencies
        if context.blocked_dependencies:
            log_data['blocked_dependencies'] = context.blocked_dependencies
        
        # Add performance metrics
        if self.include_metrics:
            if context.cpu_usage is not None:
                log_data['cpu_usage'] = context.cpu_usage
            if context.memory_usage is not None:
                log_data['memory_usage'] = context.memory_usage
            if context.network_io is not None:
                log_data['network_io'] = context.network_io
        
        # Add context information
        if self.include_context and context.context:
            log_data['context_keys'] = list(context.context.get_keys())
        
        # Add tags
        if context.tags:
            log_data['tags'] = list(context.tags)
        
        # Add extra data
        if context.extra:
            log_data['extra'] = context.extra
        
        # Log with appropriate level
        if context.error:
            logger.error(base_message, **log_data)
        elif context.hook_type in [HookType.STATE_FAILED, HookType.WORKFLOW_CANCELLED]:
            logger.warning(base_message, **log_data)
        else:
            logger.log(self.level, base_message, **log_data)

class MetricsHook(LifecycleHook):
    """Comprehensive metrics collection hook"""
    
    def __init__(
        self, 
        name: str = "metrics_hook", 
        metrics_collector=None,
        collect_performance: bool = True,
        collect_resources: bool = True,
        collect_timing: bool = True
    ):
        super().__init__(name, HookPriority.HIGH)
        self.metrics_collector = metrics_collector
        self.collect_performance = collect_performance
        self.collect_resources = collect_resources
        self.collect_timing = collect_timing
        
        # Internal metrics storage
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._state_timings: Dict[str, float] = {}

    async def _execute(self, context: HookContext) -> None:
        """Collect comprehensive metrics"""
        if not self.metrics_collector:
            return
        
        metric_data = {
            'timestamp': context.timestamp,
            'agent': context.agent_name,
            'state': context.state_name,
            'hook_type': context.hook_type.value,
            'execution_id': context.execution_id
        }
        
        # Timing metrics
        if self.collect_timing and context.duration is not None:
            self.metrics_collector.record_state_duration(
                context.agent_name,
                context.state_name,
                context.duration
            )
            metric_data['duration'] = context.duration
        
        # Performance metrics
        if self.collect_performance:
            if context.cpu_usage is not None:
                self.metrics_collector.record_cpu_usage(
                    context.agent_name,
                    context.state_name,
                    context.cpu_usage
                )
                metric_data['cpu_usage'] = context.cpu_usage
            
            if context.memory_usage is not None:
                self.metrics_collector.record_memory_usage(
                    context.agent_name,
                    context.state_name,
                    context.memory_usage
                )
                metric_data['memory_usage'] = context.memory_usage
        
        # Resource metrics
        if self.collect_resources and context.resources_allocated:
            for resource_type, amount in context.resources_allocated.items():
                self.metrics_collector.record_resource_allocation(
                    context.agent_name,
                    context.state_name,
                    resource_type,
                    amount
                )
        
        # Error metrics
        if context.error:
            self.metrics_collector.record_state_error(
                context.agent_name,
                context.state_name,
                type(context.error).__name__
            )
            metric_data['error_type'] = type(context.error).__name__
        
        # Success metrics
        elif context.hook_type == HookType.AFTER_SUCCESS:
            self.metrics_collector.record_state_success(
                context.agent_name,
                context.state_name
            )
        
        # Retry metrics
        if context.retry_count > 0:
            self.metrics_collector.record_state_retry(
                context.agent_name,
                context.state_name,
                context.retry_count
            )
            metric_data['retry_count'] = context.retry_count
        
        # Store in buffer for batch processing
        self._metrics_buffer.append(metric_data)
        
        # Flush buffer if it gets too large
        if len(self._metrics_buffer) > 1000:
            await self._flush_metrics()

    async def _flush_metrics(self) -> None:
        """Flush metrics buffer"""
        if self.metrics_collector and hasattr(self.metrics_collector, 'flush_batch'):
            await self.metrics_collector.flush_batch(self._metrics_buffer)
        self._metrics_buffer.clear()

class ValidationHook(LifecycleHook):
    """Comprehensive validation hook for inputs, outputs, and state"""
    
    def __init__(self, name: str = "validation_hook"):
        super().__init__(name, HookPriority.CRITICAL)
        self.validators: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.validation_failures: List[Dict[str, Any]] = []

    def add_input_validator(self, state_name: str, validator: Callable[[Context], bool]) -> None:
        """Add input validator for a state"""
        self.validators[state_name]['input'] = validator

    def add_output_validator(self, state_name: str, validator: Callable[[Context, Any], bool]) -> None:
        """Add output validator for a state"""
        self.validators[state_name]['output'] = validator

    def add_context_validator(self, state_name: str, validator: Callable[[Context], bool]) -> None:
        """Add context validator for a state"""
        self.validators[state_name]['context'] = validator

    async def _execute(self, context: HookContext) -> None:
        """Execute validation based on hook type"""
        if context.hook_type == HookType.BEFORE_STATE:
            await self._validate_inputs(context)
            await self._validate_context(context)
        elif context.hook_type == HookType.AFTER_STATE and not context.error:
            await self._validate_outputs(context)

    async def _validate_inputs(self, context: HookContext) -> None:
        """Validate state inputs"""
        validator = self.validators.get(context.state_name, {}).get('input')
        if validator and context.context:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(context.context)
                else:
                    result = validator(context.context)
                
                if not result:
                    error = ValidationError(f"Input validation failed for state {context.state_name}")
                    self._record_validation_failure(context, "input", error)
                    raise error
                    
            except Exception as e:
                self._record_validation_failure(context, "input", e)
                raise

    async def _validate_outputs(self, context: HookContext) -> None:
        """Validate state outputs"""
        validator = self.validators.get(context.state_name, {}).get('output')
        if validator and context.context:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(context.context, context.result)
                else:
                    result = validator(context.context, context.result)
                
                if not result:
                    error = ValidationError(f"Output validation failed for state {context.state_name}")
                    self._record_validation_failure(context, "output", error)
                    raise error
                    
            except Exception as e:
                self._record_validation_failure(context, "output", e)
                raise

    async def _validate_context(self, context: HookContext) -> None:
        """Validate context state"""
        validator = self.validators.get(context.state_name, {}).get('context')
        if validator and context.context:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(context.context)
                else:
                    result = validator(context.context)
                
                if not result:
                    error = ValidationError(f"Context validation failed for state {context.state_name}")
                    self._record_validation_failure(context, "context", error)
                    raise error
                    
            except Exception as e:
                self._record_validation_failure(context, "context", e)
                raise

    def _record_validation_failure(self, context: HookContext, validation_type: str, error: Exception) -> None:
        """Record validation failure for analysis"""
        failure_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'state_name': context.state_name,
            'agent_name': context.agent_name,
            'execution_id': context.execution_id,
            'validation_type': validation_type,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
        self.validation_failures.append(failure_record)

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class ResourceHook(LifecycleHook):
    """Hook for managing resource allocation and monitoring"""
    
    def __init__(self, name: str = "resource_hook", resource_pool=None):
        super().__init__(name, HookPriority.HIGH)
        self.resource_pool = resource_pool
        self.resource_allocations: Dict[str, Dict[str, float]] = defaultdict(dict)

    async def _execute(self, context: HookContext) -> None:
        """Handle resource-related lifecycle events"""
        if context.hook_type == HookType.BEFORE_RESOURCE_ACQUIRE:
            await self._log_resource_request(context)
        elif context.hook_type == HookType.AFTER_RESOURCE_ACQUIRE:
            await self._track_resource_allocation(context)
        elif context.hook_type == HookType.RESOURCE_ACQUIRE_FAILED:
            await self._handle_resource_failure(context)
        elif context.hook_type == HookType.AFTER_RESOURCE_RELEASE:
            await self._track_resource_release(context)

    async def _log_resource_request(self, context: HookContext) -> None:
        """Log resource requests"""
        if context.resources_requested:
            logger.info(
                f"Resource request for state {context.state_name}",
                cpu=context.resources_requested.cpu_units,
                memory=context.resources_requested.memory_mb,
                network=context.resources_requested.network_weight
            )

    async def _track_resource_allocation(self, context: HookContext) -> None:
        """Track successful resource allocations"""
        if context.resources_allocated:
            self.resource_allocations[context.state_name] = context.resources_allocated
            logger.debug(f"Resources allocated for {context.state_name}: {context.resources_allocated}")

    async def _handle_resource_failure(self, context: HookContext) -> None:
        """Handle resource allocation failures"""
        logger.warning(
            f"Resource allocation failed for state {context.state_name}",
            error=str(context.error) if context.error else "Unknown error"
        )

    async def _track_resource_release(self, context: HookContext) -> None:
        """Track resource releases"""
        if context.state_name in self.resource_allocations:
            released = self.resource_allocations.pop(context.state_name)
            logger.debug(f"Resources released for {context.state_name}: {released}")

class CheckpointHook(LifecycleHook):
    """Hook for managing checkpoints and state persistence"""
    
    def __init__(self, name: str = "checkpoint_hook", storage_backend=None):
        super().__init__(name, HookPriority.NORMAL)
        self.storage_backend = storage_backend
        self.checkpoint_frequency = timedelta(minutes=5)  # Checkpoint every 5 minutes
        self.last_checkpoint: Dict[str, datetime] = {}

    async def _execute(self, context: HookContext) -> None:
        """Handle checkpoint-related events"""
        if context.hook_type == HookType.BEFORE_CHECKPOINT:
            await self._prepare_checkpoint(context)
        elif context.hook_type == HookType.AFTER_CHECKPOINT:
            await self._finalize_checkpoint(context)
        elif context.hook_type == HookType.AFTER_STATE and self._should_checkpoint(context):
            await self._auto_checkpoint(context)

    def _should_checkpoint(self, context: HookContext) -> bool:
        """Determine if an automatic checkpoint should be created"""
        last_checkpoint = self.last_checkpoint.get(context.agent_name)
        if not last_checkpoint:
            return True
        
        return datetime.utcnow() - last_checkpoint >= self.checkpoint_frequency

    async def _prepare_checkpoint(self, context: HookContext) -> None:
        """Prepare for checkpoint creation"""
        logger.debug(f"Preparing checkpoint for agent {context.agent_name}")

    async def _finalize_checkpoint(self, context: HookContext) -> None:
        """Finalize checkpoint creation"""
        if context.checkpoint_id:
            self.last_checkpoint[context.agent_name] = datetime.utcnow()
            logger.info(f"Checkpoint created: {context.checkpoint_id}")

    async def _auto_checkpoint(self, context: HookContext) -> None:
        """Create automatic checkpoint"""
        if context.agent and self.storage_backend:
            try:
                checkpoint = context.agent.create_checkpoint()
                checkpoint_id = await self.storage_backend.save_checkpoint(
                    context.agent_name, 
                    checkpoint
                )
                self.last_checkpoint[context.agent_name] = datetime.utcnow()
                logger.debug(f"Auto-checkpoint created: {checkpoint_id}")
            except Exception as e:
                logger.error(f"Auto-checkpoint failed: {e}")

class StateLifecycle:
    """Comprehensive lifecycle management for workflow states"""
    
    def __init__(self):
        self.hooks: Dict[HookPriority, List[LifecycleHook]] = {
            priority: [] for priority in HookPriority
        }
        self.global_hooks: List[LifecycleHook] = []
        self.hook_statistics: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[HookContext] = []
        self.max_history_size = 1000

    def add_hook(
        self, 
        hook: LifecycleHook, 
        hook_types: Optional[Union[HookType, List[HookType]]] = None,
        state_names: Optional[Union[str, List[str]]] = None,
        agent_names: Optional[Union[str, List[str]]] = None,
        condition: Optional[Callable[[HookContext], bool]] = None,
        global_hook: bool = False
    ) -> None:
        """Add a hook with comprehensive filtering options"""
        
        # Set filters on the hook
        hook.set_filters(hook_types, state_names, agent_names, condition)
        
        if global_hook:
            self.global_hooks.append(hook)
        else:
            self.hooks[hook.priority].append(hook)
        
        logger.debug(f"Added hook: {hook.name} with priority {hook.priority.name}")

    def remove_hook(self, hook: LifecycleHook) -> None:
        """Remove a hook from all priority levels"""
        removed = False
        
        for priority_hooks in self.hooks.values():
            if hook in priority_hooks:
                priority_hooks.remove(hook)
                removed = True
        
        if hook in self.global_hooks:
            self.global_hooks.remove(hook)
            removed = True
        
        if removed:
            logger.debug(f"Removed hook: {hook.name}")

    def create_hook_context(
        self,
        hook_type: HookType,
        state_name: str,
        agent_name: str,
        **kwargs
    ) -> HookContext:
        """Create a comprehensive hook context"""
        return HookContext(
            hook_type=hook_type,
            state_name=state_name,
            agent_name=agent_name,
            **kwargs
        )

    async def execute_hooks(self, context: HookContext) -> None:
        """Execute all applicable hooks in priority order"""
        # Set current context
        token = _current_hook_context.set(context)
        
        try:
            # Execute hooks by priority (CRITICAL first, LOW last)
            for priority in HookPriority:
                priority_hooks = self.hooks[priority]
                if priority_hooks:
                    # Execute hooks of same priority in parallel
                    tasks = [
                        hook.execute(context) 
                        for hook in priority_hooks 
                        if hook.should_execute(context)
                    ]
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Log any hook execution failures
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                hook_name = priority_hooks[i].name
                                logger.error(f"Hook {hook_name} failed: {result}")
            
            # Execute global hooks
            if self.global_hooks:
                global_tasks = [
                    hook.execute(context)
                    for hook in self.global_hooks
                    if hook.should_execute(context)
                ]
                
                if global_tasks:
                    await asyncio.gather(*global_tasks, return_exceptions=True)
            
            # Store execution history
            self._store_execution_history(context)
            
        finally:
            _current_hook_context.set(None)

    def _store_execution_history(self, context: HookContext) -> None:
        """Store execution history for analysis"""
        self.execution_history.append(context)
        
        # Trim history if it gets too large
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size // 2:]

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about hook execution"""
        stats = {}
        
        for priority, priority_hooks in self.hooks.items():
            priority_stats = []
            for hook in priority_hooks:
                priority_stats.append(hook.get_statistics())
            stats[priority.name] = priority_stats
        
        # Global hooks statistics
        global_stats = []
        for hook in self.global_hooks:
            global_stats.append(hook.get_statistics())
        stats['GLOBAL'] = global_stats
        
        return stats

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of recent executions"""
        if not self.execution_history:
            return {}
        
        recent_executions = self.execution_history[-100:]  # Last 100 executions
        
        hook_type_counts = defaultdict(int)
        state_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        error_counts = defaultdict(int)
        
        total_duration = 0
        duration_count = 0
        
        for context in recent_executions:
            hook_type_counts[context.hook_type.value] += 1
            state_counts[context.state_name] += 1
            agent_counts[context.agent_name] += 1
            
            if context.error:
                error_counts[type(context.error).__name__] += 1
            
            if context.duration:
                total_duration += context.duration
                duration_count += 1
        
        avg_duration = total_duration / duration_count if duration_count > 0 else 0
        
        return {
            'total_executions': len(recent_executions),
            'hook_type_distribution': dict(hook_type_counts),
            'state_distribution': dict(state_counts),
            'agent_distribution': dict(agent_counts),
            'error_distribution': dict(error_counts),
            'average_duration': avg_duration,
            'error_rate': sum(error_counts.values()) / len(recent_executions) * 100
        }

def get_current_hook_context() -> Optional[HookContext]:
    """Get the current hook execution context"""
    return _current_hook_context.get()

def lifecycle_hook(
    hook_types: Union[HookType, List[HookType]],
    state_names: Optional[Union[str, List[str]]] = None,
    agent_names: Optional[Union[str, List[str]]] = None,
    priority: HookPriority = HookPriority.NORMAL,
    condition: Optional[Callable[[HookContext], bool]] = None
):
    """Decorator to create a callback hook from a function"""
    def decorator(func: Callable[[HookContext], None]) -> Callable:
        # Create a callback hook
        hook = CallbackHook(
            name=func.__name__,
            callback=func,
            hook_types=hook_types if isinstance(hook_types, list) else [hook_types],
            priority=priority
        )
        
        # Set additional filters
        hook.set_filters(hook_types, state_names, agent_names, condition)
        
        # Mark function as a lifecycle hook
        func._lifecycle_hook = hook
        
        return func
    
    return decorator

class CallbackHook(LifecycleHook):
    """Hook that executes a callback function"""
    
    def __init__(
        self, 
        name: str, 
        callback: Callable[[HookContext], None], 
        hook_types: List[HookType],
        priority: HookPriority = HookPriority.NORMAL
    ):
        super().__init__(name, priority)
        self.callback = callback
        self.set_filters(hook_types=hook_types)

    async def _execute(self, context: HookContext) -> None:
        """Execute the callback function"""
        if asyncio.iscoroutinefunction(self.callback):
            await self.callback(context)
        else:
            self.callback(context)

class StateLifecycleManager:
    """Comprehensive lifecycle manager for workflow agents"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.lifecycle = StateLifecycle()
        self._original_methods: Dict[str, Any] = {}
        self._performance_monitor = PerformanceMonitor()
        self._active_contexts: Dict[str, HookContext] = {}
        
        # Add default hooks
        self._setup_default_hooks()
        self._setup_lifecycle()

    def _setup_default_hooks(self) -> None:
        """Setup default hooks for comprehensive monitoring"""
        # Add logging hook
        logging_hook = LoggingHook("default_logging")
        self.lifecycle.add_hook(logging_hook, global_hook=True)
        
        # Add metrics hook if metrics collector is available
        if hasattr(self.agent, 'metrics_collector'):
            metrics_hook = MetricsHook("default_metrics", self.agent.metrics_collector)
            self.lifecycle.add_hook(metrics_hook, global_hook=True)
        
        # Add validation hook
        validation_hook = ValidationHook("default_validation")
        self.lifecycle.add_hook(validation_hook, global_hook=True)
        
        # Add resource hook if resource pool is available
        if hasattr(self.agent, 'resource_pool'):
            resource_hook = ResourceHook("default_resource", self.agent.resource_pool)
            self.lifecycle.add_hook(resource_hook, global_hook=True)

    def _setup_lifecycle(self) -> None:
        """Setup comprehensive lifecycle hooks for the agent"""
        self._wrap_agent_methods()

    def _wrap_agent_methods(self) -> None:
        """Wrap all relevant agent methods with lifecycle hooks"""
        # Wrap run_state method
        self._wrap_run_state()
        
        # Wrap run method
        self._wrap_run()
        
        # Wrap pause/resume methods
        self._wrap_pause_resume()
        
        # Wrap checkpoint methods
        self._wrap_checkpoint_methods()

    def _wrap_run_state(self) -> None:
        """Wrap the agent's run_state method with comprehensive hooks"""
        original_run_state = self.agent.run_state
        lifecycle = self.lifecycle
        performance_monitor = self._performance_monitor
        
        async def wrapped_run_state(state_name: str) -> None:
            execution_id = str(uuid.uuid4())
            start_time = time.time()
            context = None
            error = None
            result = None
            
            # Get state metadata
            metadata = self.agent.state_metadata.get(state_name)
            
            try:
                # Start performance monitoring
                performance_monitor.start_monitoring(state_name)
                
                # Create initial context
                hook_context = lifecycle.create_hook_context(
                    HookType.STATE_SCHEDULED,
                    state_name,
                    self.agent.name,
                    execution_id=execution_id,
                    metadata=metadata,
                    agent=self.agent
                )
                self._active_contexts[execution_id] = hook_context
                
                # State scheduled hook
                await lifecycle.execute_hooks(hook_context)
                
                # Before state hook
                hook_context.hook_type = HookType.BEFORE_STATE
                hook_context.context = Context(self.agent.shared_state)
                await lifecycle.execute_hooks(hook_context)
                
                # State started hook
                hook_context.hook_type = HookType.STATE_STARTED
                await lifecycle.execute_hooks(hook_context)
                
                # Execute original method
                result = await original_run_state(state_name)
                
                # Get performance metrics
                perf_metrics = performance_monitor.get_metrics(state_name)
                hook_context.set_performance_metrics(
                    perf_metrics.get('cpu', 0),
                    perf_metrics.get('memory', 0),
                    perf_metrics.get('network', 0)
                )
                
                duration = time.time() - start_time
                hook_context.duration = duration
                hook_context.result = result
                
                # State completed hook
                hook_context.hook_type = HookType.STATE_COMPLETED
                await lifecycle.execute_hooks(hook_context)
                
                # After state hook
                hook_context.hook_type = HookType.AFTER_STATE
                await lifecycle.execute_hooks(hook_context)
                
                # Success hook
                hook_context.hook_type = HookType.AFTER_SUCCESS
                await lifecycle.execute_hooks(hook_context)
                
            except Exception as e:
                error = e
                duration = time.time() - start_time
                
                # Update context with error information
                if execution_id in self._active_contexts:
                    hook_context = self._active_contexts[execution_id]
                    hook_context.set_error(error)
                    hook_context.duration = duration
                    
                    # Get performance metrics even on failure
                    try:
                        perf_metrics = performance_monitor.get_metrics(state_name)
                        hook_context.set_performance_metrics(
                            perf_metrics.get('cpu', 0),
                            perf_metrics.get('memory', 0),
                            perf_metrics.get('network', 0)
                        )
                    except:
                        pass
                    
                    # State failed hook
                    hook_context.hook_type = HookType.STATE_FAILED
                    await lifecycle.execute_hooks(hook_context)
                    
                    # After state hook (with error)
                    hook_context.hook_type = HookType.AFTER_STATE
                    await lifecycle.execute_hooks(hook_context)
                    
                    # Failure hook
                    hook_context.hook_type = HookType.AFTER_FAILURE
                    await lifecycle.execute_hooks(hook_context)
                
                raise
            
            finally:
                # Stop performance monitoring
                performance_monitor.stop_monitoring(state_name)
                
                # Clean up context
                if execution_id in self._active_contexts:
                    del self._active_contexts[execution_id]
        
        # Replace the method
        self.agent.run_state = wrapped_run_state
        self._original_methods['run_state'] = original_run_state

    def _wrap_run(self) -> None:
        """Wrap the agent's run method"""
        original_run = self.agent.run
        lifecycle = self.lifecycle
        
        async def wrapped_run(timeout: Optional[float] = None) -> None:
            execution_id = str(uuid.uuid4())
            
            try:
                # Before workflow hook
                hook_context = lifecycle.create_hook_context(
                    HookType.BEFORE_WORKFLOW,
                    "workflow",
                    self.agent.name,
                    execution_id=execution_id,
                    agent=self.agent
                )
                await lifecycle.execute_hooks(hook_context)
                
                # Execute original method
                result = await original_run(timeout)
                
                # After workflow hook
                hook_context.hook_type = HookType.AFTER_WORKFLOW
                hook_context.result = result
                await lifecycle.execute_hooks(hook_context)
                
            except Exception as e:
                # Workflow failed
                hook_context = lifecycle.create_hook_context(
                    HookType.AFTER_WORKFLOW,
                    "workflow",
                    self.agent.name,
                    execution_id=execution_id,
                    error=e,
                    agent=self.agent
                )
                await lifecycle.execute_hooks(hook_context)
                raise
        
        self.agent.run = wrapped_run
        self._original_methods['run'] = original_run

    def _wrap_pause_resume(self) -> None:
        """Wrap pause and resume methods"""
        original_pause = self.agent.pause
        original_resume = self.agent.resume
        lifecycle = self.lifecycle
        
        async def wrapped_pause():
            result = await original_pause()
            
            hook_context = lifecycle.create_hook_context(
                HookType.WORKFLOW_PAUSED,
                "workflow",
                self.agent.name,
                agent=self.agent,
                result=result
            )
            await lifecycle.execute_hooks(hook_context)
            
            return result
        
        async def wrapped_resume():
            result = await original_resume()
            
            hook_context = lifecycle.create_hook_context(
                HookType.WORKFLOW_RESUMED,
                "workflow",
                self.agent.name,
                agent=self.agent,
                result=result
            )
            await lifecycle.execute_hooks(hook_context)
            
            return result
        
        self.agent.pause = wrapped_pause
        self.agent.resume = wrapped_resume
        self._original_methods['pause'] = original_pause
        self._original_methods['resume'] = original_resume

    def _wrap_checkpoint_methods(self) -> None:
        """Wrap checkpoint-related methods"""
        original_create_checkpoint = self.agent.create_checkpoint
        lifecycle = self.lifecycle
        
        def wrapped_create_checkpoint():
            # Before checkpoint hook
            hook_context = lifecycle.create_hook_context(
                HookType.BEFORE_CHECKPOINT,
                "workflow",
                self.agent.name,
                agent=self.agent
            )
            
            # Note: This is sync, so we can't await
            # In a real implementation, you might want to make this async
            
            try:
                result = original_create_checkpoint()
                
                # After checkpoint hook
                hook_context.hook_type = HookType.AFTER_CHECKPOINT
                hook_context.checkpoint = result
                hook_context.checkpoint_id = getattr(result, 'checkpoint_id', str(uuid.uuid4()))
                
                return result
            except Exception as e:
                # Checkpoint failed hook
                hook_context.hook_type = HookType.CHECKPOINT_FAILED
                hook_context.set_error(e)
                raise
        
        self.agent.create_checkpoint = wrapped_create_checkpoint
        self._original_methods['create_checkpoint'] = original_create_checkpoint

    def add_hook(self, hook: LifecycleHook, **kwargs) -> None:
        """Add a hook to the lifecycle"""
        self.lifecycle.add_hook(hook, **kwargs)

    def remove_hook(self, hook: LifecycleHook) -> None:
        """Remove a hook from the lifecycle"""
        self.lifecycle.remove_hook(hook)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'agent_name': self.agent.name,
            'hook_statistics': self.lifecycle.get_hook_statistics(),
            'execution_summary': self.lifecycle.get_execution_summary(),
            'performance_summary': self._performance_monitor.get_summary(),
            'active_contexts': len(self._active_contexts)
        }

    def restore_original_methods(self) -> None:
        """Restore original agent methods"""
        for method_name, original_method in self._original_methods.items():
            setattr(self.agent, method_name, original_method)
        self._original_methods.clear()

class PerformanceMonitor:
    """Monitor performance metrics during state execution"""
    
    def __init__(self):
        self._monitoring: Dict[str, Dict[str, Any]] = {}

    def start_monitoring(self, state_name: str) -> None:
        """Start monitoring for a state"""
        self._monitoring[state_name] = {
            'start_time': time.time(),
            'cpu_start': self._get_cpu_usage(),
            'memory_start': self._get_memory_usage()
        }

    def stop_monitoring(self, state_name: str) -> None:
        """Stop monitoring for a state"""
        if state_name in self._monitoring:
            monitoring_data = self._monitoring[state_name]
            monitoring_data['end_time'] = time.time()
            monitoring_data['cpu_end'] = self._get_cpu_usage()
            monitoring_data['memory_end'] = self._get_memory_usage()

    def get_metrics(self, state_name: str) -> Dict[str, float]:
        """Get performance metrics for a state"""
        if state_name not in self._monitoring:
            return {}
        
        data = self._monitoring[state_name]
        
        duration = data.get('end_time', time.time()) - data['start_time']
        cpu_delta = data.get('cpu_end', self._get_cpu_usage()) - data['cpu_start']
        memory_delta = data.get('memory_end', self._get_memory_usage()) - data['memory_start']
        
        return {
            'duration': duration,
            'cpu': cpu_delta,
            'memory': memory_delta,
            'network': 0.0  # Placeholder - would need actual network monitoring
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring data"""
        return {
            'monitored_states': len(self._monitoring),
            'active_monitoring': sum(1 for data in self._monitoring.values() if 'end_time' not in data)
        }

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (placeholder implementation)"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage (placeholder implementation)"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

# Additional hook implementations

class TimingHook(LifecycleHook):
    """Hook that tracks detailed timing information"""
    
    def __init__(self, name: str = "timing_hook"):
        super().__init__(name, HookPriority.LOW)
        self._start_times: Dict[str, float] = {}
        self._state_timings: Dict[str, List[float]] = defaultdict(list)

    async def _execute(self, context: HookContext) -> None:
        if context.hook_type == HookType.BEFORE_STATE:
            self._start_times[f"{context.agent_name}:{context.state_name}"] = time.time()
        elif context.hook_type == HookType.AFTER_STATE:
            key = f"{context.agent_name}:{context.state_name}"
            if key in self._start_times:
                duration = time.time() - self._start_times[key]
                self._state_timings[context.state_name].append(duration)
                
                logger.info(
                    f"State timing: {context.state_name}",
                    duration_ms=round(duration * 1000, 2),
                    agent=context.agent_name
                )
                
                del self._start_times[key]

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary for all states"""
        summary = {}
        
        for state_name, timings in self._state_timings.items():
            if timings:
                summary[state_name] = {
                    'count': len(timings),
                    'total': sum(timings),
                    'average': sum(timings) / len(timings),
                    'min': min(timings),
                    'max': max(timings)
                }
        
        return summary

class RetryHook(LifecycleHook):
    """Hook that implements sophisticated retry logic"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_multiplier: float = 2.0):
        super().__init__("retry_hook", HookPriority.HIGH)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self._retry_counts: Dict[str, int] = {}
        self._retry_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def _execute(self, context: HookContext) -> None:
        if context.hook_type == HookType.AFTER_FAILURE:
            await self._handle_failure(context)
        elif context.hook_type == HookType.AFTER_SUCCESS:
            await self._handle_success(context)

    async def _handle_failure(self, context: HookContext) -> None:
        """Handle state failure and determine if retry is needed"""
        state_key = f"{context.agent_name}:{context.state_name}"
        retry_count = self._retry_counts.get(state_key, 0)
        
        # Record retry attempt
        retry_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'attempt': retry_count + 1,
            'error': str(context.error) if context.error else "Unknown error",
            'error_type': type(context.error).__name__ if context.error else "Unknown"
        }
        self._retry_history[state_key].append(retry_record)
        
        if retry_count < self.max_retries:
            self._retry_counts[state_key] = retry_count + 1
            
            # Calculate delay with exponential backoff
            delay = self.base_delay * (self.backoff_multiplier ** retry_count)
            
            logger.info(
                f"Scheduling retry for state {context.state_name}",
                attempt=retry_count + 1,
                max_retries=self.max_retries,
                delay_seconds=delay,
                error=str(context.error) if context.error else "Unknown"
            )
            
            # Before retry hook
            retry_context = context
            retry_context.hook_type = HookType.BEFORE_RETRY
            retry_context.retry_count = retry_count + 1
            await asyncio.sleep(delay)
            
            # This would trigger the actual retry - implementation depends on agent architecture
            # For now, we just log the retry intention
            
        else:
            logger.error(
                f"Retry exhausted for state {context.state_name}",
                total_attempts=retry_count + 1,
                max_retries=self.max_retries
            )
            
            # Retry exhausted hook
            context.hook_type = HookType.RETRY_EXHAUSTED
            # This would be handled by the lifecycle manager

    async def _handle_success(self, context: HookContext) -> None:
        """Handle successful state execution"""
        state_key = f"{context.agent_name}:{context.state_name}"
        
        # Reset retry count on success
        if state_key in self._retry_counts:
            retry_count = self._retry_counts.pop(state_key)
            logger.debug(f"Reset retry count for {context.state_name} after {retry_count} failures")

    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry statistics"""
        stats = {
            'active_retries': len(self._retry_counts),
            'retry_history': dict(self._retry_history),
            'total_retries': sum(len(history) for history in self._retry_history.values())
        }
        
        # Calculate success rate after retries
        successful_after_retry = 0
        total_with_retries = 0
        
        for state_key, history in self._retry_history.items():
            if history and state_key not in self._retry_counts:  # No longer retrying = success
                successful_after_retry += 1
            if history:
                total_with_retries += 1
        
        if total_with_retries > 0:
            stats['success_rate_after_retry'] = successful_after_retry / total_with_retries * 100
        
        return stats
    
# Convenience functions for common hook patterns

def timing_hook(name: str = "timing") -> TimingHook:
    """Create a timing hook"""
    return TimingHook(name)

def retry_hook(max_retries: int = 3, base_delay: float = 1.0) -> RetryHook:
    """Create a retry hook"""
    return RetryHook(max_retries, base_delay)

def logging_hook(level: int = logging.INFO, name: str = "logging") -> LoggingHook:
    """Create a logging hook"""
    return LoggingHook(name, level)

def metrics_hook(metrics_collector=None, name: str = "metrics") -> MetricsHook:
    """Create a metrics hook"""
    return MetricsHook(name, metrics_collector)

def validation_hook(name: str = "validation") -> ValidationHook:
    """Create a validation hook"""
    return ValidationHook(name)