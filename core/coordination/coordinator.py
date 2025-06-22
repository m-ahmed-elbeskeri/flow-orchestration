"""Coordination system with comprehensive monitoring and control."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Set, Optional, List, Any, Union, Callable, Protocol
import asyncio
import time
import contextlib
import weakref
import logging
from datetime import datetime, timedelta
import uuid
from functools import wraps
import inspect

from core.coordination.primitives import CoordinationPrimitive, PrimitiveType, ResourceState
from core.coordination.rate_limiter import RateLimiter, RateLimitStrategy
from core.coordination.deadlock import DeadlockDetector

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agent objects that can be coordinated."""
    name: str
    state_metadata: Dict[str, Any]

    def _add_to_queue(self, state_name: str, metadata: Any, priority_boost: int = 0) -> None: ...
    async def run_state(self, state_name: str) -> None: ...


@dataclass
class CoordinationConfig:
    """Configuration for coordination system."""
    detection_interval: float = 1.0
    cleanup_interval: float = 60.0
    max_coordination_timeout: float = 30.0
    enable_metrics: bool = True
    enable_deadlock_detection: bool = True
    max_retry_attempts: int = 3
    backoff_multiplier: float = 1.5


class CoordinationError(Exception):
    """Base exception for coordination errors."""
    pass


class CoordinationTimeout(CoordinationError):
    """Raised when coordination times out."""
    pass


class AgentCoordinator:
    """Enhanced agent coordination system with comprehensive monitoring and control."""

    def __init__(
        self,
        agent: AgentProtocol,
        config: Optional[CoordinationConfig] = None
    ):
        """Initialize the coordination system.

        Args:
            agent: The agent to coordinate
            config: Configuration for the coordination system
        """
        self.agent = weakref.proxy(agent)
        self.config = config or CoordinationConfig()
        self.instance_id = str(uuid.uuid4())

        # Components
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.primitives: Dict[str, CoordinationPrimitive] = {}

        # Initialize deadlock detector if enabled
        self.deadlock_detector: Optional[DeadlockDetector] = None
        if self.config.enable_deadlock_detection:
            self.deadlock_detector = DeadlockDetector(
                agent,
                detection_interval=self.config.detection_interval
            )

        # State management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._start_time: Optional[float] = None
        self._coordination_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'timeout_requests': 0
        }

        # Thread safety
        self._state_lock = asyncio.Lock()

        logger.info(
            f"coordinator_initialized: instance_id={self.instance_id}, "
            f"agent_name={agent.name}, detection_interval={self.config.detection_interval}, "
            f"cleanup_interval={self.config.cleanup_interval}, "
            f"deadlock_detection={self.config.enable_deadlock_detection}"
        )

    async def start(self) -> None:
        """Start the coordination system."""
        async with self._state_lock:
            if self._cleanup_task is not None:
                logger.warning(f"coordinator_already_started: instance_id={self.instance_id}")
                return

            self._start_time = time.time()
            self._shutting_down = False

            try:
                # Start deadlock detector
                if self.deadlock_detector:
                    await self.deadlock_detector.start()

                # Start cleanup task
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

                logger.info(f"coordinator_started: instance_id={self.instance_id}")

            except Exception as e:
                logger.error(f"coordinator_start_failed: instance_id={self.instance_id}, error={str(e)}")
                await self._emergency_cleanup()
                raise CoordinationError(f"Failed to start coordinator: {e}") from e

    async def stop(self) -> None:
        """Stop the coordination system gracefully."""
        async with self._state_lock:
            if self._shutting_down:
                return

            self._shutting_down = True
            logger.info(f"coordinator_stopping: instance_id={self.instance_id}")

            try:
                # Stop deadlock detector
                if self.deadlock_detector:
                    await self.deadlock_detector.stop()

                # Cancel and wait for cleanup task
                if self._cleanup_task and not self._cleanup_task.done():
                    self._cleanup_task.cancel()
                    try:
                        await asyncio.wait_for(self._cleanup_task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.warning(f"cleanup_task_forced_termination: instance_id={self.instance_id}")

                # Release all coordination resources
                await self._release_all_resources()

                # Log final statistics
                uptime = time.time() - (self._start_time or 0)
                logger.info(
                    f"coordinator_stopped: instance_id={self.instance_id}, "
                    f"uptime={uptime:.2f}, total_requests={self._coordination_stats['total_requests']}, "
                    f"successful_requests={self._coordination_stats['successful_requests']}, "
                    f"failed_requests={self._coordination_stats['failed_requests']}"
                )

            except Exception as e:
                logger.error(f"coordinator_stop_error: instance_id={self.instance_id}, error={str(e)}")

    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup in case of startup failure."""
        try:
            if self.deadlock_detector:
                await self.deadlock_detector.stop()
        except Exception as e:
            logger.error(f"emergency_cleanup_failed: instance_id={self.instance_id}, error={str(e)}")

    async def _release_all_resources(self) -> None:
        """Release all coordination resources."""
        released_count = 0
        for primitive in self.primitives.values():
            try:
                # Release all acquisitions for this coordinator instance
                caller_prefix = f"{self.instance_id}:"
                for owner in list(primitive._owners):
                    if owner.startswith(caller_prefix):
                        await primitive.release(owner)
                        released_count += 1
            except Exception as e:
                logger.error(
                    f"resource_release_error: primitive={primitive.name}, error={str(e)}"
                )

        if released_count > 0:
            logger.info(f"released_all_resources: instance_id={self.instance_id}, count={released_count}")

    def add_rate_limiter(
        self,
        name: str,
        max_rate: float,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        **kwargs
    ) -> None:
        """Add a rate limiter.

        Args:
            name: Name of the rate limiter
            max_rate: Maximum rate (requests per second)
            strategy: Rate limiting strategy
            **kwargs: Additional arguments for the rate limiter
        """
        if name in self.rate_limiters:
            logger.warning(f"rate_limiter_already_exists: name={name}")
            return

        self.rate_limiters[name] = RateLimiter(
            max_rate=max_rate,
            strategy=strategy,
            **kwargs
        )

        logger.info(
            f"rate_limiter_added: name={name}, max_rate={max_rate}, strategy={strategy.name}"
        )

    def create_primitive(
        self,
        name: str,
        primitive_type: PrimitiveType,
        **kwargs
    ) -> None:
        """Create a coordination primitive.

        Args:
            name: Name of the primitive
            primitive_type: Type of coordination primitive
            **kwargs: Additional arguments for the primitive
        """
        if name in self.primitives:
            logger.warning(f"primitive_already_exists: name={name}")
            return

        self.primitives[name] = CoordinationPrimitive(
            name=name,
            type=primitive_type,
            **kwargs
        )

        logger.info(
            f"primitive_created: name={name}, type={primitive_type.name}, "
            f"config={kwargs}"
        )

    async def coordinate_state_execution(
        self,
        state_name: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Coordinate state execution with rate limiting and resource management.

        Args:
            state_name: Name of the state to coordinate
            timeout: Optional timeout for coordination

        Returns:
            True if coordination successful, False otherwise
        """
        coordination_id = str(uuid.uuid4())
        start_time = time.time()
        timeout = timeout or self.config.max_coordination_timeout

        self._coordination_stats['total_requests'] += 1

        logger.debug(
            f"coordination_request: state={state_name}, "
            f"coordination_id={coordination_id}, timeout={timeout}"
        )

        try:
            # Check rate limits
            if state_name in self.rate_limiters:
                if not await asyncio.wait_for(
                    self.rate_limiters[state_name].acquire(),
                    timeout=timeout
                ):
                    self._coordination_stats['rate_limited_requests'] += 1
                    await self._log_coordination_failure(
                        state_name, coordination_id, "rate_limit_exceeded"
                    )
                    return False

            # Check coordination primitives
            caller_id = f"{self.instance_id}:{state_name}:{coordination_id}"
            acquired_primitives = []

            try:
                for primitive_name, primitive in self.primitives.items():
                    remaining_timeout = timeout - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        raise asyncio.TimeoutError("Coordination timeout")

                    if not await asyncio.wait_for(
                        primitive.acquire(caller_id, timeout=remaining_timeout),
                        timeout=remaining_timeout
                    ):
                        await self._log_coordination_failure(
                            state_name, coordination_id, f"primitive_blocked:{primitive_name}"
                        )
                        return False

                    acquired_primitives.append((primitive_name, primitive))

                # All coordination successful
                self._coordination_stats['successful_requests'] += 1
                duration = time.time() - start_time

                logger.debug(
                    f"coordination_successful: state={state_name}, "
                    f"coordination_id={coordination_id}, duration={duration:.3f}, "
                    f"acquired_primitives={[name for name, _ in acquired_primitives]}"
                )

                return True

            except asyncio.TimeoutError:
                self._coordination_stats['timeout_requests'] += 1
                # Release any acquired primitives
                for primitive_name, primitive in acquired_primitives:
                    try:
                        await primitive.release(caller_id)
                    except Exception as release_error:
                        logger.error(
                            f"primitive_release_error: primitive={primitive_name}, "
                            f"caller_id={caller_id}, error={str(release_error)}"
                        )

                await self._log_coordination_failure(
                    state_name, coordination_id, "timeout"
                )
                return False

        except Exception as e:
            self._coordination_stats['failed_requests'] += 1
            await self._log_coordination_failure(
                state_name, coordination_id, f"exception:{str(e)}"
            )
            return False

    async def _log_coordination_failure(
        self,
        state_name: str,
        coordination_id: str,
        reason: str
    ) -> None:
        """Log coordination failure with monitoring integration."""
        logger.warning(
            f"coordination_failed: state={state_name}, "
            f"coordination_id={coordination_id}, reason={reason}"
        )

        if hasattr(self.agent, '_monitor'):
            try:
                self.agent._monitor.logger.warning(
                    f"coordination_failed: state={state_name}, "
                    f"coordination_id={coordination_id}, reason={reason}"
                )
            except Exception as e:
                logger.error(f"monitor_logging_error: {str(e)}")

    async def release_coordination(self, state_name: str, coordination_id: Optional[str] = None) -> None:
        """Release coordination resources for a state.

        Args:
            state_name: Name of the state
            coordination_id: Optional specific coordination ID
        """
        if coordination_id:
            caller_id = f"{self.instance_id}:{state_name}:{coordination_id}"
        else:
            # Release all coordinations for this state
            caller_prefix = f"{self.instance_id}:{state_name}:"

        released_count = 0

        for primitive_name, primitive in self.primitives.items():
            try:
                if coordination_id:
                    await primitive.release(caller_id)
                    released_count += 1
                else:
                    # Release all matching coordination IDs
                    for owner in list(primitive._owners):
                        if owner.startswith(caller_prefix):
                            await primitive.release(owner)
                            released_count += 1
            except Exception as e:
                logger.error(
                    f"coordination_release_error: primitive={primitive_name}, "
                    f"state={state_name}, error={str(e)}"
                )

        logger.debug(
            f"coordination_released: state={state_name}, "
            f"coordination_id={coordination_id}, released_count={released_count}"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status."""
        uptime = time.time() - (self._start_time or 0) if self._start_time else 0

        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent.name,
            "uptime": uptime,
            "shutting_down": self._shutting_down,
            "config": asdict(self.config),
            "stats": self._coordination_stats.copy(),
            "rate_limiters": {
                name: limiter.get_stats()
                for name, limiter in self.rate_limiters.items()
            },
            "primitives": {
                name: primitive.get_state()
                for name, primitive in self.primitives.items()
            },
            "deadlock_detector": (
                self.deadlock_detector.get_status()
                if self.deadlock_detector else None
            )
        }

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for maintenance tasks."""
        logger.info(f"cleanup_loop_started: instance_id={self.instance_id}")

        while not self._shutting_down:
            try:
                cleanup_start = time.time()

                # Clean up expired primitive acquisitions
                cleanup_count = 0
                for primitive in self.primitives.values():
                    try:
                        async with primitive._lock:
                            before_count = len(primitive._owners)
                            primitive._cleanup_expired()
                            after_count = len(primitive._owners)
                            cleanup_count += before_count - after_count
                    except Exception as e:
                        logger.error(
                            f"primitive_cleanup_error: primitive={primitive.name}, error={str(e)}"
                        )

                cleanup_duration = time.time() - cleanup_start

                if cleanup_count > 0 or cleanup_duration > 1.0:
                    logger.debug(
                        f"cleanup_cycle_completed: instance_id={self.instance_id}, "
                        f"cleaned_acquisitions={cleanup_count}, duration={cleanup_duration:.3f}"
                    )

                await asyncio.sleep(self.config.cleanup_interval)

            except asyncio.CancelledError:
                logger.info(f"cleanup_loop_cancelled: instance_id={self.instance_id}")
                break
            except Exception as e:
                logger.error(
                    f"cleanup_loop_error: instance_id={self.instance_id}, error={str(e)}"
                )
                # Continue the loop even on errors
                await asyncio.sleep(1.0)

        logger.info(f"cleanup_loop_stopped: instance_id={self.instance_id}")


def enhance_agent(agent: AgentProtocol, config: Optional[CoordinationConfig] = None) -> AgentProtocol:
    """Add production coordination to an agent with proper method binding.

    Args:
        agent: The agent to enhance
        config: Optional coordination configuration

    Returns:
        The enhanced agent
    """
    # Add coordinator
    coordinator = AgentCoordinator(agent, config)
    agent._coordinator = coordinator

    # Store original methods
    original_run_state = agent.run_state
    original_cleanup = getattr(agent, '_cleanup', None)

    # Handle startup coordination
    async def start_coordinator():
        await coordinator.start()

    if hasattr(agent, '_startup_tasks'):
        agent._startup_tasks.append(start_coordinator())
    else:
        # Only create task if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(start_coordinator())
        except RuntimeError:
            # No running event loop, coordinator will be started manually
            logger.info(f"no_event_loop_for_auto_start: agent={agent.name}")

    # Enhanced run_state with proper binding
    async def enhanced_run_state(state_name: str) -> None:
        """Enhanced state execution with coordination and monitoring."""
        attempt_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check coordination and rate limits
            if not await agent._coordinator.coordinate_state_execution(state_name):
                if hasattr(agent, '_monitor'):
                    agent._monitor.logger.warning(
                        f"coordination_failed: state={state_name}, attempt={attempt_id}"
                    )

                # Requeue with backoff if agent supports it
                if hasattr(agent, '_add_to_queue') and hasattr(agent, 'state_metadata'):
                    if state_name in agent.state_metadata:
                        agent._add_to_queue(
                            state_name,
                            agent.state_metadata[state_name],
                            priority_boost=-1
                        )
                return

            # Log execution start
            if hasattr(agent, '_monitor'):
                metadata = {}
                if hasattr(agent, 'state_metadata') and state_name in agent.state_metadata:
                    state_meta = agent.state_metadata[state_name]
                    metadata = {
                        "resources": asdict(state_meta.resources) if hasattr(state_meta, 'resources') else {},
                        "dependencies": len(getattr(state_meta, 'dependencies', [])),
                        "attempts": getattr(state_meta, 'attempts', 0)
                    }

                agent._monitor.logger.info(
                    f"state_execution_started: state={state_name}, "
                    f"attempt={attempt_id}, metadata={metadata}"
                )

            # Execute original state with monitoring span
            async with agent._execution_span(state_name, attempt_id):
                await original_run_state(state_name)

            # Record success metrics
            if hasattr(agent, '_monitor'):
                duration = time.time() - start_time
                await agent._monitor.record_metric(
                    'state_duration',
                    duration,
                    {'state': state_name, 'status': 'success'}
                )
                await agent._monitor.record_metric(
                    'state_success',
                    1,
                    {'state': state_name}
                )

        except Exception as e:
            # Handle failure with monitoring
            if hasattr(agent, '_monitor'):
                duration = time.time() - start_time
                agent._monitor.logger.error(
                    f"state_execution_failed: state={state_name}, "
                    f"attempt={attempt_id}, error={str(e)}, duration={duration:.3f}"
                )
                await agent._monitor.record_metric(
                    'state_duration',
                    duration,
                    {'state': state_name, 'status': 'error'}
                )
                await agent._monitor.record_metric(
                    'state_error',
                    1,
                    {'state': state_name, 'error_type': type(e).__name__}
                )
            raise

        finally:
            # Always release coordination
            await agent._coordinator.release_coordination(state_name, attempt_id)

    # Bind the enhanced method to the agent
    agent.run_state = enhanced_run_state

    # Add execution span context manager
    @contextlib.asynccontextmanager
    async def _execution_span(state_name: str, attempt_id: str):
        """Create execution span for monitoring."""
        if hasattr(agent, '_monitor'):
            try:
                async with agent._monitor.monitor_operation(
                    "state_execution",
                    {
                        "state": state_name,
                        "attempt": attempt_id,
                        "agent": agent.name
                    }
                ) as span:
                    yield span
            except Exception as e:
                logger.error(f"monitor_span_error: {str(e)}")
                yield None
        else:
            yield None

    agent._execution_span = _execution_span

    # Enhanced cleanup
    async def enhanced_cleanup():
        """Enhanced cleanup with coordination system shutdown."""
        try:
            # Stop coordinator first
            await agent._coordinator.stop()

            # Run original cleanup if it exists
            if original_cleanup:
                if inspect.iscoroutinefunction(original_cleanup):
                    await original_cleanup()
                else:
                    original_cleanup()

        except Exception as e:
            if hasattr(agent, '_monitor'):
                agent._monitor.logger.error(f"cleanup_error: error={str(e)}")
            logger.error(f"enhanced_cleanup_error: agent={agent.name}, error={str(e)}")
            raise

    agent._cleanup = enhanced_cleanup

    # Add utility methods with proper binding
    def add_utility_methods():
        async def get_coordination_status() -> Dict[str, Any]:
            """Get coordination system status."""
            return agent._coordinator.get_status()

        async def reset_coordination():
            """Reset coordination system."""
            old_config = agent._coordinator.config
            await agent._coordinator.stop()
            agent._coordinator = AgentCoordinator(agent, old_config)
            await agent._coordinator.start()

        def add_state_rate_limit(
            state_name: str,
            max_rate: float,
            strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
            **kwargs
        ):
            """Add rate limit for specific state."""
            agent._coordinator.add_rate_limiter(
                state_name,
                max_rate,
                strategy,
                **kwargs
            )

        def add_state_coordination(
            state_name: str,
            primitive_type: PrimitiveType,
            **kwargs
        ):
            """Add coordination primitive for specific state."""
            agent._coordinator.create_primitive(
                f"state_{state_name}",
                primitive_type,
                **kwargs
            )

        # Bind methods to agent
        agent.get_coordination_status = get_coordination_status
        agent.reset_coordination = reset_coordination
        agent.add_state_rate_limit = add_state_rate_limit
        agent.add_state_coordination = add_state_coordination

    add_utility_methods()

    logger.info(
        f"agent_enhanced: agent_name={agent.name}, "
        f"coordinator_id={coordinator.instance_id}"
    )

    return agent


def create_coordinated_agent(
    name: str,
    config: Optional[CoordinationConfig] = None,
    **agent_kwargs
) -> Any:
    """Create an agent with coordination enabled.

    Args:
        name: Name of the agent
        config: Optional coordination configuration
        **agent_kwargs: Additional arguments for agent creation

    Returns:
        Enhanced agent with coordination
    """
    from core.agent.base import Agent

    agent = Agent(name, **agent_kwargs)
    return enhance_agent(agent, config)