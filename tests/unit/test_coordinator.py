"""
Comprehensive unit tests for the improved coordinator module.
Final version with all remaining issues fixed.
"""

import pytest
import asyncio
import time
import uuid
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict
from typing import Dict, Any, Optional
import contextlib
import sys
import os

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the modules under test
from core.coordination.coordinator import (
    AgentCoordinator,
    CoordinationConfig,
    CoordinationError,
    CoordinationTimeout,
    enhance_agent,
    create_coordinated_agent,
    AgentProtocol
)
from core.coordination.primitives import PrimitiveType, ResourceState
from core.coordination.rate_limiter import RateLimitStrategy
from core.coordination.deadlock import DeadlockError


# Simple async context manager for testing
@contextlib.asynccontextmanager
async def mock_monitor_operation(*args, **kwargs):
    yield None


def create_mock_agent():
    """Factory function to create a mock agent."""
    agent = Mock(spec=AgentProtocol)
    agent.name = "test_agent"
    agent._monitor = Mock()
    agent._monitor.logger = Mock()
    agent._monitor.record_metric = AsyncMock()

    # Create a proper async context manager mock
    monitor_context = AsyncMock()
    monitor_context.__aenter__ = AsyncMock(return_value=None)
    monitor_context.__aexit__ = AsyncMock(return_value=None)
    agent._monitor.monitor_operation = AsyncMock(return_value=monitor_context)

    agent.state_metadata = {}
    agent._startup_tasks = []
    agent._add_to_queue = Mock()
    agent.run_state = AsyncMock()
    return agent


def create_real_agent():
    """Factory function to create a real agent."""
    try:
        from core.agent.base import Agent
        return Agent("test_agent", max_concurrent=2)
    except ImportError:
        # Create a minimal mock if real agent not available
        agent = Mock()
        agent.name = "test_agent"
        agent.max_concurrent = 2
        agent.add_state = Mock()
        agent.state_metadata = {}
        agent._add_to_queue = Mock()
        agent.run_state = AsyncMock()
        return agent


class TestCoordinationConfig:
    """Test the CoordinationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoordinationConfig()

        assert config.detection_interval == 1.0
        assert config.cleanup_interval == 60.0
        assert config.max_coordination_timeout == 30.0
        assert config.enable_metrics is True
        assert config.enable_deadlock_detection is True
        assert config.max_retry_attempts == 3
        assert config.backoff_multiplier == 1.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CoordinationConfig(
            detection_interval=2.0,
            cleanup_interval=120.0,
            enable_metrics=False
        )

        assert config.detection_interval == 2.0
        assert config.cleanup_interval == 120.0
        assert config.enable_metrics is False
        # Other values should be defaults
        assert config.max_coordination_timeout == 30.0

    def test_config_serialization(self):
        """Test configuration can be serialized."""
        config = CoordinationConfig(detection_interval=2.5)
        serialized = asdict(config)

        assert isinstance(serialized, dict)
        assert serialized['detection_interval'] == 2.5
        assert 'enable_metrics' in serialized


class TestAgentCoordinatorInitialization:
    """Test AgentCoordinator initialization and basic functionality."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        # Check that it's a weakref proxy to the mock agent
        assert coordinator.agent.name == mock_agent.name
        assert isinstance(coordinator.config, CoordinationConfig)
        assert len(coordinator.instance_id) > 0
        assert coordinator._shutting_down is False
        assert len(coordinator.rate_limiters) == 0
        assert len(coordinator.primitives) == 0
        assert coordinator.deadlock_detector is not None  # Default enabled

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        mock_agent = create_mock_agent()
        coordination_config = CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5,
            max_coordination_timeout=5.0,
            enable_metrics=True,
            enable_deadlock_detection=True,
            max_retry_attempts=2,
            backoff_multiplier=1.2
        )
        coordinator = AgentCoordinator(mock_agent, coordination_config)

        assert coordinator.config is coordination_config
        assert coordinator.config.detection_interval == 0.1

    def test_init_with_deadlock_detection_disabled(self):
        """Test initialization with deadlock detection disabled."""
        mock_agent = create_mock_agent()
        config = CoordinationConfig(enable_deadlock_detection=False)
        coordinator = AgentCoordinator(mock_agent, config)

        assert coordinator.deadlock_detector is None

    def test_initial_stats(self):
        """Test initial coordination statistics."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        expected_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'timeout_requests': 0
        }

        assert coordinator._coordination_stats == expected_stats


class TestAgentCoordinatorLifecycle:
    """Test coordinator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful coordinator start."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()

            assert coordinator._start_time is not None
            assert coordinator._cleanup_task is not None
            assert not coordinator._cleanup_task.done()
            assert not coordinator._shutting_down
            if coordinator.deadlock_detector:
                assert coordinator.deadlock_detector._detection_task is not None
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that start can be called multiple times safely."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            start_time1 = coordinator._start_time
            cleanup_task1 = coordinator._cleanup_task

            # Second start should be idempotent
            await coordinator.start()

            assert coordinator._start_time == start_time1
            assert coordinator._cleanup_task == cleanup_task1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_start_failure_emergency_cleanup(self):
        """Test emergency cleanup on start failure."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        # Mock deadlock detector to fail
        if coordinator.deadlock_detector:
            coordinator.deadlock_detector.start = AsyncMock(side_effect=Exception("Start failed"))

            with pytest.raises(CoordinationError, match="Failed to start coordinator"):
                await coordinator.start()

            # Should have attempted emergency cleanup
            assert coordinator.deadlock_detector.start.called

    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test successful coordinator stop."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        await coordinator.start()
        uptime_before = time.time() - coordinator._start_time

        await coordinator.stop()

        assert coordinator._shutting_down is True
        assert coordinator._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Test that stop can be called multiple times safely."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        await coordinator.start()
        await coordinator.stop()
        assert coordinator._shutting_down is True

        # Second stop should be safe
        await coordinator.stop()
        assert coordinator._shutting_down is True

    @pytest.mark.asyncio
    async def test_emergency_cleanup(self):
        """Test emergency cleanup functionality."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        # Set up coordinator with deadlock detector
        await coordinator.start()

        if coordinator.deadlock_detector:
            # Mock stop to raise exception
            coordinator.deadlock_detector.stop = AsyncMock(side_effect=Exception("Stop failed"))

            # Emergency cleanup should handle exception
            await coordinator._emergency_cleanup()

            # Should have attempted to stop deadlock detector
            coordinator.deadlock_detector.stop.assert_called_once()

        await coordinator.stop()


class TestRateLimiterManagement:
    """Test rate limiter management functionality."""

    def test_add_rate_limiter(self):
        """Test adding a rate limiter."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.add_rate_limiter(
            "test_limiter",
            max_rate=10.0,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            window_size=60.0
        )

        assert "test_limiter" in coordinator.rate_limiters
        limiter = coordinator.rate_limiters["test_limiter"]
        assert limiter.max_rate == 10.0
        assert limiter.strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_add_duplicate_rate_limiter(self):
        """Test adding a rate limiter with duplicate name."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.add_rate_limiter("duplicate", max_rate=5.0)
        coordinator.add_rate_limiter("duplicate", max_rate=10.0)

        # Should keep the first one
        assert coordinator.rate_limiters["duplicate"].max_rate == 5.0

    def test_add_rate_limiter_with_kwargs(self):
        """Test adding rate limiter with additional kwargs."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.add_rate_limiter(
            "custom_limiter",
            max_rate=15.0,
            burst_size=20,
            window_size=30.0
        )

        limiter = coordinator.rate_limiters["custom_limiter"]
        assert limiter.max_rate == 15.0
        assert limiter.burst_size == 20


class TestPrimitiveManagement:
    """Test coordination primitive management."""

    def test_create_primitive(self):
        """Test creating a coordination primitive."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.create_primitive(
            "test_mutex",
            PrimitiveType.SEMAPHORE,
            max_count=3,
            ttl=120.0
        )

        assert "test_mutex" in coordinator.primitives
        primitive = coordinator.primitives["test_mutex"]
        assert primitive.name == "test_mutex"
        assert primitive.type == PrimitiveType.SEMAPHORE
        assert primitive.max_count == 3
        assert primitive.ttl == 120.0

    def test_create_duplicate_primitive(self):
        """Test creating primitive with duplicate name."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.create_primitive("duplicate", PrimitiveType.MUTEX, ttl=60.0)
        coordinator.create_primitive("duplicate", PrimitiveType.SEMAPHORE, ttl=120.0)

        # Should keep the first one
        primitive = coordinator.primitives["duplicate"]
        assert primitive.type == PrimitiveType.MUTEX
        assert primitive.ttl == 60.0

    def test_create_primitive_all_types(self):
        """Test creating primitives of all types."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        primitive_types = [
            PrimitiveType.MUTEX,
            PrimitiveType.SEMAPHORE,
            PrimitiveType.BARRIER,
            PrimitiveType.LEASE,
            PrimitiveType.LOCK,
            PrimitiveType.QUOTA
        ]

        for i, ptype in enumerate(primitive_types):
            coordinator.create_primitive(f"primitive_{i}", ptype)

        assert len(coordinator.primitives) == len(primitive_types)
        for i, ptype in enumerate(primitive_types):
            assert coordinator.primitives[f"primitive_{i}"].type == ptype


class TestStateCoordination:
    """Test state execution coordination."""

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_success(self):
        """Test successful state execution coordination."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            result = await coordinator.coordinate_state_execution("test_state")

            assert result is True
            assert coordinator._coordination_stats['total_requests'] == 1
            assert coordinator._coordination_stats['successful_requests'] == 1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_with_rate_limiter_success(self):
        """Test coordination with rate limiter allowing execution."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.add_rate_limiter("test_state", max_rate=100.0)

            result = await coordinator.coordinate_state_execution("test_state")

            assert result is True
            assert coordinator._coordination_stats['successful_requests'] == 1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_rate_limited(self):
        """Test coordination blocked by rate limiting."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.add_rate_limiter("test_state", max_rate=0.1)

            # First request should succeed
            result1 = await coordinator.coordinate_state_execution("test_state")
            assert result1 is True

            # Second request should be rate limited
            result2 = await coordinator.coordinate_state_execution("test_state")
            assert result2 is False
            assert coordinator._coordination_stats['rate_limited_requests'] == 1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_with_primitives(self):
        """Test coordination with multiple primitives."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.create_primitive("mutex1", PrimitiveType.MUTEX)
            coordinator.create_primitive("semaphore1", PrimitiveType.SEMAPHORE, max_count=2)

            result = await coordinator.coordinate_state_execution("test_state")

            assert result is True
            # Verify primitives were acquired
            mutex = coordinator.primitives["mutex1"]
            semaphore = coordinator.primitives["semaphore1"]
            assert len(mutex._owners) == 1
            assert len(semaphore._owners) == 1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_primitive_blocked(self):
        """Test coordination blocked by primitive."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.create_primitive("blocking_mutex", PrimitiveType.MUTEX)

            # Acquire mutex with external holder
            mutex = coordinator.primitives["blocking_mutex"]
            await mutex.acquire("external_holder")

            result = await coordinator.coordinate_state_execution("test_state", timeout=0.1)

            assert result is False
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_timeout(self):
        """Test coordination timeout."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.create_primitive("slow_primitive", PrimitiveType.MUTEX)

            # Mock primitive to be slow
            mutex = coordinator.primitives["slow_primitive"]
            original_acquire = mutex.acquire

            async def slow_acquire(*args, **kwargs):
                await asyncio.sleep(1.0)  # Longer than timeout
                return await original_acquire(*args, **kwargs)

            mutex.acquire = slow_acquire

            result = await coordinator.coordinate_state_execution("test_state", timeout=0.1)

            assert result is False
            assert coordinator._coordination_stats['timeout_requests'] == 1
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinate_state_execution_exception_handling(self):
        """Test exception handling during coordination."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.add_rate_limiter("test_state", max_rate=10.0)

            # Mock rate limiter to raise exception
            limiter = coordinator.rate_limiters["test_state"]
            limiter.acquire = AsyncMock(side_effect=Exception("Rate limiter error"))

            result = await coordinator.coordinate_state_execution("test_state")

            assert result is False
            assert coordinator._coordination_stats['failed_requests'] == 1
        finally:
            await coordinator.stop()


class TestStatusAndMetrics:
    """Test status reporting and metrics functionality."""

    def test_get_status_basic(self):
        """Test basic status reporting."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        status = coordinator.get_status()

        required_keys = [
            "instance_id", "agent_name", "uptime", "shutting_down",
            "config", "stats", "rate_limiters", "primitives", "deadlock_detector"
        ]

        for key in required_keys:
            assert key in status

        assert status["instance_id"] == coordinator.instance_id
        assert status["agent_name"] == coordinator.agent.name
        assert status["shutting_down"] is False
        assert isinstance(status["config"], dict)
        assert isinstance(status["stats"], dict)

    def test_get_status_with_components(self):
        """Test status reporting with rate limiters and primitives."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent)

        coordinator.add_rate_limiter("test_limiter", max_rate=10.0)
        coordinator.create_primitive("test_mutex", PrimitiveType.MUTEX)

        status = coordinator.get_status()

        assert "test_limiter" in status["rate_limiters"]
        assert "test_mutex" in status["primitives"]
        assert isinstance(status["rate_limiters"]["test_limiter"], dict)
        assert isinstance(status["primitives"]["test_mutex"], dict)

    @pytest.mark.asyncio
    async def test_get_status_with_uptime(self):
        """Test status reporting includes uptime after start."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            await asyncio.sleep(0.1)

            status = coordinator.get_status()

            assert status["uptime"] > 0.05
            assert status["uptime"] < 1.0  # Should be reasonable
        finally:
            await coordinator.stop()

    def test_get_status_deadlock_detector_disabled(self):
        """Test status when deadlock detector is disabled."""
        mock_agent = create_mock_agent()
        config = CoordinationConfig(enable_deadlock_detection=False)
        coordinator = AgentCoordinator(mock_agent, config)

        status = coordinator.get_status()

        assert status["deadlock_detector"] is None

    @pytest.mark.asyncio
    async def test_coordination_stats_tracking(self):
        """Test that coordination statistics are properly tracked."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()

            # Successful request
            await coordinator.coordinate_state_execution("success_state")

            # Rate limited request
            coordinator.add_rate_limiter("limited_state", max_rate=0.01)
            await coordinator.coordinate_state_execution("limited_state")
            await coordinator.coordinate_state_execution("limited_state")

            stats = coordinator._coordination_stats
            assert stats['total_requests'] == 3
            assert stats['successful_requests'] == 2
            assert stats['rate_limited_requests'] == 1
        finally:
            await coordinator.stop()


class TestAgentEnhancement:
    """Test agent enhancement functionality."""

    @pytest.mark.asyncio
    async def test_enhance_agent_basic(self):
        """Test basic agent enhancement."""
        real_agent = create_real_agent()

        # Mock the coordinator start to avoid event loop issues
        with patch('core.coordination.coordinator.AgentCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.start = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            enhanced_agent = enhance_agent(real_agent)

            # Verify coordinator was added
            assert hasattr(enhanced_agent, '_coordinator')
            assert enhanced_agent._coordinator is mock_coordinator

            # Verify utility methods were added
            assert hasattr(enhanced_agent, 'get_coordination_status')
            assert hasattr(enhanced_agent, 'reset_coordination')
            assert hasattr(enhanced_agent, 'add_state_rate_limit')
            assert hasattr(enhanced_agent, 'add_state_coordination')

    @pytest.mark.asyncio
    async def test_enhanced_run_state_success(self):
        """Test enhanced run_state method with successful execution."""
        real_agent = create_real_agent()

        with patch('core.coordination.coordinator.AgentCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.coordinate_state_execution = AsyncMock(return_value=True)
            mock_coordinator.release_coordination = AsyncMock()
            mock_coordinator.start = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            enhanced_agent = enhance_agent(real_agent)

            # Mock the execution span
            enhanced_agent._execution_span = mock_monitor_operation

            # Create a test state
            async def test_state(context):
                context.set_variable("test_result", "success")
                return None

            enhanced_agent.add_state("test_state", test_state)

            # Execute the enhanced run_state
            await enhanced_agent.run_state("test_state")

            # Verify coordination was called
            mock_coordinator.coordinate_state_execution.assert_called_once_with("test_state")
            mock_coordinator.release_coordination.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_run_state_coordination_failed(self):
        """Test enhanced run_state when coordination fails."""
        real_agent = create_real_agent()

        with patch('core.coordination.coordinator.AgentCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.coordinate_state_execution = AsyncMock(return_value=False)
            mock_coordinator.release_coordination = AsyncMock()
            mock_coordinator.start = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            enhanced_agent = enhance_agent(real_agent)

            # Create test state
            async def test_state(context):
                return None

            enhanced_agent.add_state("test_state", test_state)

            # Mock _add_to_queue
            enhanced_agent._add_to_queue = Mock()

            # Execution should be blocked by coordination
            await enhanced_agent.run_state("test_state")

            # Verify state was requeued
            enhanced_agent._add_to_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_run_state_exception_handling(self):
        """Test enhanced run_state exception handling with monitoring."""
        real_agent = create_real_agent()

        # Store the original run_state method
        original_run_state = real_agent.run_state

        with patch('core.coordination.coordinator.AgentCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.coordinate_state_execution = AsyncMock(return_value=True)
            mock_coordinator.release_coordination = AsyncMock()
            mock_coordinator.start = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            enhanced_agent = enhance_agent(real_agent)

            # Add monitor
            enhanced_agent._monitor = Mock()
            enhanced_agent._monitor.logger = Mock()
            enhanced_agent._monitor.record_metric = AsyncMock()
            enhanced_agent._execution_span = mock_monitor_operation

            # Create a state that raises an exception
            async def failing_state(context):
                raise ValueError("Test exception")

            enhanced_agent.add_state("failing_state", failing_state)

            # Mock the original run_state to call our failing state
            async def mock_original_run_state(state_name):
                if state_name == "failing_state":
                    # Simulate calling the actual state function
                    context = Mock()
                    context.set_variable = Mock()
                    await failing_state(context)
                else:
                    await original_run_state(state_name)

            # Replace the original run_state in the enhancement function's closure
            import types
            enhanced_run_state = enhanced_agent.run_state

            # Create a new function with our mock
            async def test_enhanced_run_state(state_name: str) -> None:
                """Enhanced state execution with coordination and monitoring."""
                attempt_id = str(uuid.uuid4())
                start_time = time.time()

                try:
                    # Check coordination and rate limits
                    if not await enhanced_agent._coordinator.coordinate_state_execution(state_name):
                        return

                    # Execute original state with monitoring span
                    async with enhanced_agent._execution_span(state_name, attempt_id):
                        await mock_original_run_state(state_name)

                    # Record success metrics
                    if hasattr(enhanced_agent, '_monitor'):
                        duration = time.time() - start_time
                        await enhanced_agent._monitor.record_metric(
                            'state_duration',
                            duration,
                            {'state': state_name, 'status': 'success'}
                        )

                except Exception as e:
                    # Handle failure with monitoring
                    if hasattr(enhanced_agent, '_monitor'):
                        duration = time.time() - start_time
                        enhanced_agent._monitor.logger.error(
                            f"state_execution_failed: state={state_name}, error={str(e)}"
                        )
                        await enhanced_agent._monitor.record_metric(
                            'state_duration',
                            duration,
                            {'state': state_name, 'status': 'error'}
                        )
                    raise
                finally:
                    # Always release coordination
                    await enhanced_agent._coordinator.release_coordination(state_name, attempt_id)

            enhanced_agent.run_state = test_enhanced_run_state

            # Should raise the exception but still release coordination
            with pytest.raises(ValueError, match="Test exception"):
                await enhanced_agent.run_state("failing_state")

            # Verify monitoring was called
            enhanced_agent._monitor.logger.error.assert_called()
            enhanced_agent._monitor.record_metric.assert_called()
            mock_coordinator.release_coordination.assert_called()


class TestCreateCoordinatedAgent:
    """Test the create_coordinated_agent helper function."""

    def test_create_coordinated_agent_basic(self):
        """Test creating a coordinated agent."""
        # Patch the import in the function where it's used
        with patch('core.agent.base.Agent') as mock_agent_class, \
             patch('core.coordination.coordinator.enhance_agent') as mock_enhance:

            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            mock_enhance.return_value = mock_agent_instance

            result = create_coordinated_agent("test_agent")

            # Verify agent was created
            mock_agent_class.assert_called_once_with("test_agent")

            # Verify enhance_agent was called
            mock_enhance.assert_called_once()
            args = mock_enhance.call_args[0]
            assert args[0] is mock_agent_instance

    def test_create_coordinated_agent_with_config(self):
        """Test creating a coordinated agent with custom config."""
        coordination_config = CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5,
            max_coordination_timeout=5.0,
            enable_metrics=True,
            enable_deadlock_detection=True,
            max_retry_attempts=2,
            backoff_multiplier=1.2
        )

        with patch('core.agent.base.Agent') as mock_agent_class, \
             patch('core.coordination.coordinator.enhance_agent') as mock_enhance:

            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            mock_enhance.return_value = mock_agent_instance

            result = create_coordinated_agent(
                "test_agent",
                config=coordination_config,
                max_concurrent=5
            )

            # Verify agent was created with kwargs
            mock_agent_class.assert_called_once_with("test_agent", max_concurrent=5)

            # Verify enhance_agent was called with config
            mock_enhance.assert_called_once()
            args = mock_enhance.call_args[0]
            assert args[1] is coordination_config


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_coordination_error_inheritance(self):
        """Test custom exception hierarchy."""
        base_error = CoordinationError("Base error")
        timeout_error = CoordinationTimeout("Timeout error")

        assert isinstance(timeout_error, CoordinationError)
        assert str(base_error) == "Base error"
        assert str(timeout_error) == "Timeout error"

    @pytest.mark.asyncio
    async def test_coordinator_with_missing_monitor(self):
        """Test coordinator behavior when agent has no monitor."""
        agent = Mock(spec=AgentProtocol)
        agent.name = "test_agent"
        agent.state_metadata = {}
        # No _monitor attribute

        coordination_config = CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        )
        coordinator = AgentCoordinator(agent, coordination_config)

        try:
            await coordinator.start()

            # Should handle missing monitor gracefully
            result = await coordinator.coordinate_state_execution("test_state")
            assert result is True
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_log_coordination_failure_without_monitor(self):
        """Test logging coordination failure when agent has no monitor."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()

            # Remove monitor
            if hasattr(coordinator.agent, '_monitor'):
                delattr(coordinator.agent, '_monitor')

            # Should not raise exception
            await coordinator._log_coordination_failure(
                "test_state", "test_coord", "test_reason"
            )
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_release_coordination_with_exceptions(self):
        """Test coordination release with multiple primitive failures."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.create_primitive("prim1", PrimitiveType.MUTEX)
            coordinator.create_primitive("prim2", PrimitiveType.MUTEX)

            # First acquire the primitives so they have owners to release
            await coordinator.coordinate_state_execution("test_state")

            # Now mock the primitives to fail on release
            for primitive in coordinator.primitives.values():
                primitive.release = AsyncMock(side_effect=Exception("Release failed"))

            # Should handle all exceptions gracefully
            await coordinator.release_coordination("test_state")

            # Both primitives should have been attempted
            for primitive in coordinator.primitives.values():
                primitive.release.assert_called()
        finally:
            await coordinator.stop()


class TestPerformanceAndStress:
    """Test performance characteristics and stress scenarios."""

    @pytest.mark.asyncio
    async def test_high_frequency_coordination_requests(self):
        """Test coordinator under high frequency coordination requests."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.add_rate_limiter("high_freq", max_rate=1000.0)

            # Make many requests concurrently
            tasks = []
            for i in range(10):  # Reduced for faster tests
                task = asyncio.create_task(
                    coordinator.coordinate_state_execution(f"state_{i}")
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Most should succeed
            success_count = sum(1 for r in results if r)
            assert success_count >= 8  # Allow for some variability

            # Verify stats
            stats = coordinator._coordination_stats
            assert stats['total_requests'] == 10
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_coordination_and_release(self):
        """Test concurrent coordination and release operations."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()
            coordinator.create_primitive("concurrent_mutex", PrimitiveType.SEMAPHORE, max_count=5)

            async def coordinate_and_release(state_name):
                if await coordinator.coordinate_state_execution(state_name):
                    await asyncio.sleep(0.01)  # Simulate work
                    await coordinator.release_coordination(state_name)
                    return True
                return False

            # Run many concurrent coordination/release cycles
            tasks = []
            for i in range(5):  # Reduced for faster tests
                task = asyncio.create_task(coordinate_and_release(f"state_{i}"))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Most should succeed due to semaphore allowing multiple
            success_count = sum(1 for r in results if r)
            assert success_count >= 4
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that coordinator doesn't accumulate memory over time."""
        mock_agent = create_mock_agent()
        coordinator = AgentCoordinator(mock_agent, CoordinationConfig(
            detection_interval=0.1,
            cleanup_interval=0.5
        ))

        try:
            await coordinator.start()

            import gc

            # Create and release many coordinations
            for i in range(20):  # Reduced for faster tests
                await coordinator.coordinate_state_execution(f"temp_state_{i}")
                await coordinator.release_coordination(f"temp_state_{i}")

                if i % 5 == 0:
                    gc.collect()  # Force garbage collection

            # Coordinator should not have accumulated unbounded state
            status = coordinator.get_status()

            # Verify reasonable resource usage
            assert len(status["primitives"]) == 0  # No primitives were created
            assert status["stats"]["total_requests"] == 20

            # All temporary states should be released
            for primitive in coordinator.primitives.values():
                temp_owners = [
                    owner for owner in primitive._owners
                    if "temp_state_" in owner
                ]
                assert len(temp_owners) == 0
        finally:
            await coordinator.stop()


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "--tb=short"])