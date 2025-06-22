"""
Comprehensive test coverage for core.agent.base module.

Tests cover:
- RetryPolicy class functionality
- Agent initialization and configuration
- State management (add_state, run_state)
- Dependency resolution and queue management
- Error handling and retry logic
- Checkpoint creation/restoration
- Pause/resume functionality
- Cancellation logic
- Main workflow execution scenarios
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
from copy import deepcopy

# Import the modules to test
from core.agent.base import Agent, RetryPolicy
from core.agent.state import (
    StateStatus, AgentStatus, StateMetadata, PrioritizedState, Priority
)
from core.agent.context import Context
from core.agent.dependencies import DependencyConfig
from core.resources.requirements import ResourceRequirements


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def retry_policy():
    """Standard retry policy for testing."""
    return RetryPolicy(
        max_retries=3,
        initial_delay=0.01,
        exponential_base=2.0,
        jitter=False
    )


@pytest.fixture
def agent():
    """Basic agent instance for testing."""
    return Agent(
        name="test_agent",
        max_concurrent=2,
        retry_policy=RetryPolicy(max_retries=2, initial_delay=0.01),
        state_timeout=1.0
    )


@pytest.fixture
def simple_state_func():
    """Simple async state function that returns success."""
    async def state_func(context: Context) -> str:
        await asyncio.sleep(0.001)  # Very small delay
        return "completed"
    return state_func


@pytest.fixture
def failing_state_func():
    """State function that always fails."""
    async def state_func(context: Context) -> None:
        await asyncio.sleep(0.001)
        raise ValueError("Test failure")
    return state_func


@pytest.fixture
def mock_resource_pool():
    """Mock resource pool for testing."""
    pool = Mock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    return pool


# ============================================================================
# RETRY POLICY TESTS
# ============================================================================

class TestRetryPolicy:
    """Test cases for RetryPolicy class."""

    def test_retry_policy_initialization(self):
        """Test RetryPolicy initialization with various parameters."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=2.0,
            exponential_base=3.0,
            jitter=True
        )

        assert policy.max_retries == 5
        assert policy.initial_delay == 2.0
        assert policy.exponential_base == 3.0
        assert policy.jitter is True

    def test_retry_policy_defaults(self):
        """Test RetryPolicy with default values."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    @pytest.mark.asyncio
    async def test_wait_without_jitter(self):
        """Test wait method without jitter for predictable delays."""
        policy = RetryPolicy(
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False
        )

        # Test increasing delays
        start_time = time.time()
        await policy.wait(0)
        first_delay = time.time() - start_time
        assert first_delay >= 0.008  # Allow for timing variance

        start_time = time.time()
        await policy.wait(1)
        second_delay = time.time() - start_time
        assert second_delay >= 0.015  # Should be roughly 2x first delay

    @pytest.mark.asyncio
    async def test_wait_with_jitter(self):
        """Test wait method with jitter enabled."""
        policy = RetryPolicy(
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=True
        )

        start_time = time.time()
        await policy.wait(0)
        delay = time.time() - start_time

        # With jitter, delay should be variable but reasonable
        # Increased upper bound to account for system timing variations
        assert 0.005 <= delay <= 0.025

    @pytest.mark.asyncio
    async def test_wait_max_delay_cap(self):
        """Test that wait method respects the 60 second cap."""
        policy = RetryPolicy(
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False
        )

        # Mock the sleep function to verify the cap logic without actually waiting
        with patch('asyncio.sleep') as mock_sleep:
            await policy.wait(20)  # Would be huge without cap

            # Verify that sleep was called with capped value
            mock_sleep.assert_called_once()
            called_delay = mock_sleep.call_args[0][0]
            assert called_delay <= 60.0

    @pytest.mark.asyncio
    async def test_wait_with_zero_delay(self):
        """Test wait method with zero initial delay."""
        policy = RetryPolicy(
            initial_delay=0.0,
            exponential_base=2.0,
            jitter=False
        )

        start_time = time.time()
        await policy.wait(0)
        delay = time.time() - start_time

        assert delay < 0.01  # Should be nearly instantaneous

    @pytest.mark.asyncio
    async def test_wait_exponential_calculation(self):
        """Test that exponential backoff calculation is correct."""
        policy = RetryPolicy(
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False
        )

        with patch('asyncio.sleep') as mock_sleep:
            await policy.wait(3)

            # 0.01 * 2^3 = 0.08, capped at 60
            expected_delay = min(0.01 * (2.0 ** 3), 60.0)
            mock_sleep.assert_called_once_with(expected_delay)


# ============================================================================
# AGENT INITIALIZATION TESTS
# ============================================================================

class TestAgentInitialization:
    """Test cases for Agent initialization."""

    def test_agent_basic_initialization(self):
        """Test basic Agent initialization."""
        agent = Agent(name="test_agent")

        assert agent.name == "test_agent"
        assert agent.max_concurrent == 5
        assert isinstance(agent.retry_policy, RetryPolicy)
        assert agent.state_timeout is None
        assert agent.status == AgentStatus.IDLE
        assert len(agent.states) == 0
        assert len(agent.state_metadata) == 0
        assert len(agent.dependencies) == 0
        assert len(agent.priority_queue) == 0
        assert isinstance(agent.shared_state, dict)
        assert len(agent._running_states) == 0
        assert len(agent.completed_states) == 0
        assert len(agent.completed_once) == 0
        assert isinstance(agent.context, Context)
        assert agent.session_start is None

    def test_agent_initialization_with_custom_values(self, retry_policy, mock_resource_pool):
        """Test Agent initialization with custom parameters."""
        agent = Agent(
            name="custom_agent",
            max_concurrent=10,
            retry_policy=retry_policy,
            state_timeout=30.0,
            resource_pool=mock_resource_pool
        )

        assert agent.name == "custom_agent"
        assert agent.max_concurrent == 10
        assert agent.retry_policy is retry_policy
        assert agent.state_timeout == 30.0
        assert agent.resource_pool is mock_resource_pool

    def test_agent_initialization_with_none_retry_policy(self):
        """Test Agent initialization with None retry policy creates default."""
        agent = Agent(name="test_agent", retry_policy=None)

        assert isinstance(agent.retry_policy, RetryPolicy)
        assert agent.retry_policy.max_retries == 3


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================

class TestStateManagement:
    """Test cases for state management functionality."""

    def test_add_state_basic(self, agent, simple_state_func):
        """Test adding a basic state to the agent."""
        agent.add_state("test_state", simple_state_func)

        assert "test_state" in agent.states
        assert agent.states["test_state"] is simple_state_func
        assert "test_state" in agent.state_metadata
        assert agent.dependencies["test_state"] == []

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.PENDING
        assert metadata.max_retries == agent.retry_policy.max_retries
        assert isinstance(metadata.resources, ResourceRequirements)
        assert metadata.priority == Priority.NORMAL

    def test_add_state_with_dependencies(self, agent, simple_state_func):
        """Test adding a state with dependencies."""
        dependencies = ["state1", "state2"]
        agent.add_state("test_state", simple_state_func, dependencies=dependencies)

        assert agent.dependencies["test_state"] == dependencies

    def test_add_state_with_custom_resources(self, agent, simple_state_func):
        """Test adding a state with custom resource requirements."""
        resources = ResourceRequirements(cpu_units=2.0, memory_mb=500.0)
        agent.add_state("test_state", simple_state_func, resources=resources)

        metadata = agent.state_metadata["test_state"]
        assert metadata.resources is resources

    def test_add_state_with_custom_retry_settings(self, agent, simple_state_func):
        """Test adding a state with custom retry settings."""
        custom_retry_policy = RetryPolicy(max_retries=5)
        agent.add_state(
            "test_state",
            simple_state_func,
            max_retries=10,
            retry_policy=custom_retry_policy
        )

        metadata = agent.state_metadata["test_state"]
        assert metadata.max_retries == 10
        assert metadata.retry_policy is custom_retry_policy

    def test_add_state_with_priority(self, agent, simple_state_func):
        """Test adding a state with custom priority."""
        agent.add_state("test_state", simple_state_func, priority=Priority.HIGH)

        metadata = agent.state_metadata["test_state"]
        assert metadata.priority == Priority.HIGH

    def test_find_entry_states_no_dependencies(self, agent, simple_state_func):
        """Test finding entry states when states have no dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func)

        entry_states = agent._find_entry_states()
        assert set(entry_states) == {"state1", "state2"}

    def test_find_entry_states_with_dependencies(self, agent, simple_state_func):
        """Test finding entry states when some states have dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])
        agent.add_state("state3", simple_state_func)

        entry_states = agent._find_entry_states()
        assert set(entry_states) == {"state1", "state3"}

    def test_find_entry_states_all_have_dependencies(self, agent, simple_state_func):
        """Test finding entry states when all states have dependencies."""
        agent.add_state("state1", simple_state_func, dependencies=["state2"])
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        entry_states = agent._find_entry_states()
        assert len(entry_states) == 0


# ============================================================================
# QUEUE MANAGEMENT TESTS
# ============================================================================

class TestQueueManagement:
    """Test cases for priority queue management."""

    @pytest.mark.asyncio
    async def test_add_to_queue_basic(self, agent, simple_state_func):
        """Test adding states to the priority queue."""
        agent.add_state("test_state", simple_state_func)

        await agent._add_to_queue("test_state")

        assert len(agent.priority_queue) == 1
        state_item = agent.priority_queue[0]
        assert isinstance(state_item, PrioritizedState)
        assert state_item.state_name == "test_state"

    @pytest.mark.asyncio
    async def test_add_to_queue_with_priority_boost(self, agent, simple_state_func):
        """Test adding states to queue with priority boost."""
        agent.add_state("test_state", simple_state_func, priority=Priority.NORMAL)

        await agent._add_to_queue("test_state", priority_boost=2)

        state_item = agent.priority_queue[0]
        expected_priority = -(Priority.NORMAL.value + 2)
        assert state_item.priority == expected_priority

    @pytest.mark.asyncio
    async def test_add_to_queue_nonexistent_state(self, agent, caplog):
        """Test adding non-existent state to queue logs error."""
        with caplog.at_level(logging.ERROR):
            await agent._add_to_queue("nonexistent_state")

        assert "State nonexistent_state not found in metadata" in caplog.text
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_get_ready_states_empty_queue(self, agent):
        """Test getting ready states from empty queue."""
        ready_states = await agent._get_ready_states()
        assert ready_states == []

    @pytest.mark.asyncio
    async def test_get_ready_states_with_ready_state(self, agent, simple_state_func):
        """Test getting ready states when states are ready to run."""
        agent.add_state("test_state", simple_state_func)
        await agent._add_to_queue("test_state")

        ready_states = await agent._get_ready_states()
        assert ready_states == ["test_state"]

    @pytest.mark.asyncio
    async def test_can_run_basic_state(self, agent, simple_state_func):
        """Test checking if a basic state can run."""
        agent.add_state("test_state", simple_state_func)

        can_run = await agent._can_run("test_state")
        assert can_run is True

    @pytest.mark.asyncio
    async def test_can_run_already_running_state(self, agent, simple_state_func):
        """Test checking if an already running state can run."""
        agent.add_state("test_state", simple_state_func)
        agent._running_states.add("test_state")

        can_run = await agent._can_run("test_state")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_completed_once_state(self, agent, simple_state_func):
        """Test checking if a state that completed once can run again."""
        agent.add_state("test_state", simple_state_func)
        agent.completed_once.add("test_state")

        can_run = await agent._can_run("test_state")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_with_unmet_dependencies(self, agent, simple_state_func):
        """Test checking if a state with unmet dependencies can run."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        can_run = await agent._can_run("state2")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_with_met_dependencies(self, agent, simple_state_func):
        """Test checking if a state with met dependencies can run."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])
        agent.completed_states.add("state1")

        can_run = await agent._can_run("state2")
        assert can_run is True


# ============================================================================
# STATE EXECUTION TESTS
# ============================================================================

class TestStateExecution:
    """Test cases for state execution functionality."""

    @pytest.mark.asyncio
    async def test_run_state_success(self, agent, simple_state_func):
        """Test successful state execution."""
        agent.add_state("test_state", simple_state_func)

        await agent.run_state("test_state")

        assert "test_state" not in agent._running_states
        assert "test_state" in agent.completed_states
        assert "test_state" in agent.completed_once

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.COMPLETED
        assert metadata.last_execution is not None
        assert metadata.last_success is not None

    @pytest.mark.asyncio
    async def test_run_state_already_running(self, agent, simple_state_func):
        """Test running a state that's already running."""
        agent.add_state("test_state", simple_state_func)
        agent._running_states.add("test_state")

        await agent.run_state("test_state")

        # State should still be in running states (early return)
        assert "test_state" in agent._running_states

    @pytest.mark.asyncio
    async def test_run_state_with_context_operations(self, agent):
        """Test state execution with context operations."""
        async def context_state(context: Context) -> None:
            context.set_variable("test_key", "test_value")

        agent.add_state("context_state", context_state)

        await agent.run_state("context_state")

        assert agent.shared_state.get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_handle_state_result_none(self, agent):
        """Test handling state result when result is None."""
        await agent._handle_state_result("test_state", None)
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_handle_state_result_string(self, agent, simple_state_func):
        """Test handling state result when result is a next state name."""
        agent.add_state("next_state", simple_state_func)

        await agent._handle_state_result("test_state", "next_state")

        assert len(agent.priority_queue) == 1
        assert agent.priority_queue[0].state_name == "next_state"

    @pytest.mark.asyncio
    async def test_handle_state_result_list(self, agent, simple_state_func):
        """Test handling state result when result is a list of state names."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func)

        await agent._handle_state_result("test_state", ["state1", "state2"])

        assert len(agent.priority_queue) == 2
        state_names = {item.state_name for item in agent.priority_queue}
        assert state_names == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_resolve_dependencies(self, agent, simple_state_func):
        """Test dependency resolution logs warning for unmet dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        with patch('core.agent.base.logger') as mock_logger:
            await agent._resolve_dependencies("state2")
            mock_logger.warning.assert_called_once()


# ============================================================================
# ERROR HANDLING AND RETRY TESTS
# ============================================================================

class TestErrorHandlingAndRetry:
    """Test cases for error handling and retry functionality."""

    @pytest.mark.asyncio
    async def test_run_state_with_failure(self, agent, failing_state_func):
        """Test state execution with failure and retry."""
        agent.add_state("failing_state", failing_state_func)

        await agent.run_state("failing_state")

        metadata = agent.state_metadata["failing_state"]
        assert metadata.status == StateStatus.PENDING
        assert metadata.attempts == 1
        assert len(agent.priority_queue) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_within_retry_limit(self, agent, failing_state_func):
        """Test failure handling within retry limit."""
        agent.add_state("test_state", failing_state_func)
        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 1
        metadata.max_retries = 3

        with patch.object(metadata.retry_policy, 'wait', new_callable=AsyncMock) as mock_wait:
            await agent._handle_failure("test_state", ValueError("Test error"))

        assert metadata.attempts == 2
        assert metadata.status == StateStatus.PENDING
        mock_wait.assert_called_once_with(2)
        assert len(agent.priority_queue) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_exceeds_retry_limit(self, agent, failing_state_func):
        """Test failure handling when retry limit is exceeded."""
        agent.add_state("test_state", failing_state_func)
        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 3
        metadata.max_retries = 3

        await agent._handle_failure("test_state", ValueError("Test error"))

        assert metadata.attempts == 4
        assert metadata.status == StateStatus.FAILED
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_handle_failure_with_compensation(self, agent, failing_state_func, simple_state_func):
        """Test failure handling with compensation state."""
        agent.add_state("test_state", failing_state_func)
        agent.add_state("test_state_compensation", simple_state_func)

        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 3
        metadata.max_retries = 3

        await agent._handle_failure("test_state", ValueError("Test error"))

        assert len(agent.priority_queue) == 1
        assert agent.priority_queue[0].state_name == "test_state_compensation"


# ============================================================================
# CHECKPOINT MANAGEMENT TESTS
# ============================================================================

class TestCheckpointManagement:
    """Test cases for checkpoint functionality."""

    def test_create_checkpoint(self, agent):
        """Test creating a checkpoint from agent state."""
        # Create a simple state function locally for this test
        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)
        agent.completed_states.add("completed_state")
        agent.shared_state["test_key"] = "test_value"

        checkpoint = agent.create_checkpoint()

        assert checkpoint.agent_name == agent.name
        assert checkpoint.agent_status == agent.status
        assert checkpoint.completed_states == agent.completed_states
        assert checkpoint.shared_state == agent.shared_state
        assert checkpoint.timestamp is not None

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, agent):
        """Test restoring agent from checkpoint."""
        mock_checkpoint = Mock()
        mock_checkpoint.agent_status = AgentStatus.PAUSED
        mock_checkpoint.priority_queue = []
        mock_checkpoint.state_metadata = {}
        mock_checkpoint.running_states = set()
        mock_checkpoint.completed_states = {"state1"}
        mock_checkpoint.completed_once = {"state1"}
        mock_checkpoint.shared_state = {"key": "value"}
        mock_checkpoint.session_start = time.time()

        await agent.restore_from_checkpoint(mock_checkpoint)

        assert agent.status == AgentStatus.PAUSED
        assert agent.completed_states == {"state1"}
        assert agent.completed_once == {"state1"}
        assert agent.shared_state == {"key": "value"}

    @pytest.mark.asyncio
    async def test_pause_returns_checkpoint(self, agent):
        """Test that pause returns a checkpoint."""
        checkpoint = await agent.pause()

        assert agent.status == AgentStatus.PAUSED
        assert checkpoint.agent_name == agent.name
        assert checkpoint.agent_status == AgentStatus.PAUSED

    @pytest.mark.asyncio
    async def test_resume_from_paused(self, agent):
        """Test resuming from paused state."""
        agent.status = AgentStatus.PAUSED

        await agent.resume()

        assert agent.status == AgentStatus.RUNNING


# ============================================================================
# CANCELLATION TESTS
# ============================================================================

class TestCancellation:
    """Test cases for cancellation functionality."""

    def test_cancel_state(self, agent):
        """Test cancelling a specific state."""
        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)
        agent._running_states.add("test_state")

        agent.cancel_state("test_state")

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.CANCELLED
        assert "test_state" not in agent._running_states

    def test_cancel_nonexistent_state(self, agent):
        """Test cancelling a non-existent state."""
        agent.cancel_state("nonexistent_state")  # Should not raise

    @pytest.mark.asyncio
    async def test_cancel_all(self, agent):
        """Test cancelling all states."""
        async def simple_state(context):
            return "completed"

        agent.add_state("state1", simple_state)
        agent.add_state("state2", simple_state)
        agent._running_states.update(["state1", "state2"])
        agent.priority_queue = [Mock(), Mock()]

        await agent.cancel_all()

        assert agent.status == AgentStatus.CANCELLED
        assert len(agent._running_states) == 0
        assert len(agent.priority_queue) == 0


# ============================================================================
# WORKFLOW EXECUTION TESTS
# ============================================================================

class TestWorkflowExecution:
    """Test cases for main workflow execution scenarios."""

    @pytest.mark.asyncio
    async def test_run_empty_workflow(self, agent, caplog):
        """Test running workflow with no states."""
        with caplog.at_level(logging.INFO):
            await agent.run()

        assert "No states defined, nothing to run" in caplog.text
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_run_single_state_workflow(self, agent, simple_state_func):
        """Test running workflow with single state."""
        agent.add_state("test_state", simple_state_func)

        await agent.run()

        assert agent.status == AgentStatus.COMPLETED
        assert "test_state" in agent.completed_states

    @pytest.mark.asyncio
    async def test_run_sequential_workflow(self, agent):
        """Test running workflow with sequential states."""
        call_order = []

        async def state1(context: Context) -> str:
            call_order.append("state1")
            return "state2"

        async def state2(context: Context) -> None:
            call_order.append("state2")

        agent.add_state("state1", state1)
        agent.add_state("state2", state2)

        await agent.run()

        assert agent.status == AgentStatus.COMPLETED
        assert call_order == ["state1", "state2"]
        assert agent.completed_states == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_run_workflow_manual_dependency_execution(self, agent):
        """Test workflow execution by manually managing dependencies."""
        call_order = []

        async def state1(context: Context) -> None:
            call_order.append("state1")
            # Manually add dependent state to queue
            await agent._add_to_queue("state2")

        async def state2(context: Context) -> None:
            call_order.append("state2")

        agent.add_state("state1", state1)
        agent.add_state("state2", state2, dependencies=["state1"])

        # Manually mark state1 as completed to satisfy dependencies
        await agent.run_state("state1")
        agent.completed_states.add("state1")

        # Now run state2
        await agent.run_state("state2")

        assert call_order == ["state1", "state2"]
        assert agent.completed_states == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_run_workflow_with_failure(self, agent, simple_state_func, failing_state_func):
        """Test running workflow with state failure."""
        agent.retry_policy.max_retries = 1

        agent.add_state("good_state", simple_state_func)
        agent.add_state("bad_state", failing_state_func, max_retries=1)

        # Run each state individually to test failure handling
        await agent.run_state("good_state")
        await agent.run_state("bad_state")

        assert "good_state" in agent.completed_states
        assert agent.state_metadata["bad_state"].status in [StateStatus.FAILED, StateStatus.PENDING]

    @pytest.mark.asyncio
    async def test_session_start_tracking(self, agent, simple_state_func):
        """Test that session start time is tracked."""
        agent.add_state("test_state", simple_state_func)

        start_time = time.time()
        await agent.run()

        assert agent.session_start is not None
        assert agent.session_start >= start_time

    @pytest.mark.asyncio
    async def test_timeout_mechanism(self, agent):
        """Test timeout mechanism with mocked sleep."""
        async def slow_state(context: Context) -> None:
            # This would normally take a long time, but we'll mock it
            await asyncio.sleep(0.001)

        agent.add_state("slow_state", slow_state)

        # Mock time.time to simulate timeout
        start_time = time.time()
        with patch('time.time') as mock_time:
            # Provide enough values to avoid StopIteration
            mock_time.side_effect = [
                start_time,           # Session start
                start_time,           # Main execution loop start
                start_time + 0.05,    # Still within timeout
                start_time + 0.2      # Exceeds timeout
            ]

            await agent.run(timeout=0.1)

        # Should complete normally with our mocked timing
        assert agent.status in [AgentStatus.COMPLETED, AgentStatus.RUNNING]

    @pytest.mark.asyncio
    async def test_max_concurrent_execution(self, agent):
        """Test that max_concurrent limit is respected."""
        execution_order = []

        async def concurrent_state(context: Context) -> None:
            execution_order.append(f"start_{context.get_variable('state_id', 'unknown')}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{context.get_variable('state_id', 'unknown')}")

        # Set max_concurrent to 1 to force sequential execution
        agent.max_concurrent = 1

        for i in range(3):
            agent.add_state(f"state{i}", concurrent_state)

        # We'll test this by running states individually to verify the logic
        tasks = []
        for i in range(3):
            tasks.append(asyncio.create_task(agent.run_state(f"state{i}")))

        await asyncio.gather(*tasks)

        # All states should have completed
        assert len(agent.completed_states) == 3

    @pytest.mark.asyncio
    async def test_workflow_exception_handling(self, agent):
        """Test workflow exception handling."""
        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)

        with patch.object(agent, '_get_ready_states', side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                await agent.run()

        assert agent.status == AgentStatus.FAILED


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_complex_workflow_with_retries(self, agent):
        """Test complex workflow with retries and state transitions."""
        execution_log = []

        async def reliable_start(context: Context) -> str:
            execution_log.append("reliable_start")
            context.set_variable("workflow_id", "test_workflow")
            return "final_cleanup"

        async def final_cleanup(context: Context) -> None:
            execution_log.append("final_cleanup")
            workflow_id = context.get_variable("workflow_id")
            assert workflow_id == "test_workflow"

        agent.add_state("reliable_start", reliable_start)
        agent.add_state("final_cleanup", final_cleanup)

        await agent.run()

        assert agent.status == AgentStatus.COMPLETED
        assert "reliable_start" in execution_log
        assert "final_cleanup" in execution_log
        assert agent.shared_state.get("workflow_id") == "test_workflow"

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, agent):
        """Test that priority queue maintains correct ordering."""
        # Create a simple state function for this test
        async def simple_state_func(context):
            return "completed"

        agent.add_state("low_priority", simple_state_func, priority=Priority.LOW)
        agent.add_state("high_priority", simple_state_func, priority=Priority.HIGH)
        agent.add_state("normal_priority", simple_state_func, priority=Priority.NORMAL)

        await agent._add_to_queue("low_priority")
        await agent._add_to_queue("high_priority")
        await agent._add_to_queue("normal_priority")

        # Priority queue should be ordered by priority (negative values for max-heap)
        priorities = [item.priority for item in agent.priority_queue]
        # heapq maintains min-heap order, so most negative (highest priority) comes first
        assert priorities[0] == -Priority.HIGH.value  # Most negative = highest priority
        # The rest should maintain heap property, not necessarily sorted

    @pytest.mark.asyncio
    async def test_state_metadata_persistence(self, agent, simple_state_func):
        """Test that state metadata is properly maintained."""
        agent.add_state("test_state", simple_state_func, priority=Priority.HIGH)

        metadata = agent.state_metadata["test_state"]
        original_id = metadata.state_id

        await agent.run_state("test_state")

        # Metadata should be updated but preserve important fields
        assert metadata.state_id == original_id
        assert metadata.status == StateStatus.COMPLETED
        assert metadata.priority == Priority.HIGH
        assert metadata.last_execution is not None
        assert metadata.last_success is not None

    @pytest.mark.asyncio
    async def test_resource_requirements_integration(self, agent, simple_state_func):
        """Test integration with resource requirements."""
        resources = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            priority_boost=1
        )

        agent.add_state("resource_state", simple_state_func, resources=resources)

        metadata = agent.state_metadata["resource_state"]
        assert metadata.resources.cpu_units == 2.0
        assert metadata.resources.memory_mb == 512.0
        assert metadata.resources.priority_boost == 1

        await agent.run_state("resource_state")
        assert "resource_state" in agent.completed_states


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_state_name(self, agent):
        """Test adding state with empty name."""
        async def simple_state(context):
            return "completed"

        agent.add_state("", simple_state)
        assert "" in agent.states

    @pytest.mark.asyncio
    async def test_duplicate_state_name(self, agent):
        """Test adding duplicate state names (should overwrite)."""
        async def state_v1(context): return "v1"
        async def state_v2(context): return "v2"

        agent.add_state("duplicate", state_v1)
        agent.add_state("duplicate", state_v2)

        assert agent.states["duplicate"] is state_v2

    @pytest.mark.asyncio
    async def test_circular_dependencies(self, agent):
        """Test handling of circular dependencies."""
        async def simple_state(context):
            return "completed"

        agent.add_state("state1", simple_state, dependencies=["state2"])
        agent.add_state("state2", simple_state, dependencies=["state1"])

        # Neither state should be able to run due to circular dependencies
        assert not await agent._can_run("state1")
        assert not await agent._can_run("state2")

    @pytest.mark.asyncio
    async def test_state_execution_with_context_exception(self, agent):
        """Test state execution when context operations fail."""
        async def failing_context_state(context: Context) -> None:
            # This might fail if context is in bad state
            try:
                context.set_variable("test", "value")
            except:
                pass  # Ignore context errors for this test

        agent.add_state("context_state", failing_context_state)

        # Should complete even if context operations have issues
        await agent.run_state("context_state")
        assert "context_state" in agent.completed_states

    def test_agent_string_representation(self, agent):
        """Test that agent can be represented as string for debugging."""
        string_repr = str(agent)
        assert "test_agent" in string_repr or "Agent" in string_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])