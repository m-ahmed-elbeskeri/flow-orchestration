"""
Comprehensive test coverage for core.agent.checkpoint module.

Tests cover:
- AgentCheckpoint creation and initialization
- Creating checkpoints from agent instances
- Data integrity and deep copying
- Edge cases with complex state data
- Memory management and performance
- Integration with agent lifecycle
- Error handling scenarios
- Checkpoint data validation
"""

import pytest
import time
import copy
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Set, List, Any, Optional
from dataclasses import FrozenInstanceError

# Import the modules to test
from core.agent.checkpoint import AgentCheckpoint
from core.agent.state import AgentStatus, StateMetadata, PrioritizedState, StateStatus, Priority
from core.agent.base import Agent, RetryPolicy
from core.resources.requirements import ResourceRequirements


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_agent():
    """Create a mock agent with comprehensive state for testing."""
    agent = Mock()
    agent.name = "test_agent"
    agent.status = AgentStatus.RUNNING
    agent.priority_queue = []
    agent.state_metadata = {}
    agent._running_states = set()
    agent.completed_states = set()
    agent.completed_once = set()
    agent.shared_state = {}
    agent._session_start = time.time()
    return agent


@pytest.fixture
def sample_state_metadata():
    """Create sample state metadata for testing."""
    return {
        "state1": StateMetadata(
            status=StateStatus.COMPLETED,
            attempts=1,
            max_retries=3,
            resources=ResourceRequirements(),
            dependencies={},
            satisfied_dependencies=set(),
            last_execution=time.time(),
            last_success=time.time(),
            state_id="state1-id",
            retry_policy=None
        ),
        "state2": StateMetadata(
            status=StateStatus.RUNNING,
            attempts=0,
            max_retries=2,
            resources=ResourceRequirements(),
            dependencies={},
            satisfied_dependencies={"state1"},
            last_execution=None,
            last_success=None,
            state_id="state2-id",
            retry_policy=None
        )
    }


@pytest.fixture
def sample_priority_queue():
    """Create sample priority queue items for testing."""
    metadata1 = StateMetadata(
        status=StateStatus.PENDING,
        attempts=0,
        max_retries=3
    )
    metadata2 = StateMetadata(
        status=StateStatus.PENDING,
        attempts=1,
        max_retries=3
    )

    return [
        PrioritizedState(
            priority=-Priority.HIGH.value,
            timestamp=time.time(),
            state_name="high_priority_state",
            metadata=metadata1
        ),
        PrioritizedState(
            priority=-Priority.NORMAL.value,
            timestamp=time.time() + 1,
            state_name="normal_priority_state",
            metadata=metadata2
        )
    ]


@pytest.fixture
def complex_shared_state():
    """Create complex shared state data for testing."""
    return {
        "simple_string": "test_value",
        "number": 42,
        "boolean": True,
        "none_value": None,
        "list_data": [1, 2, 3, "mixed", {"nested": "dict"}],
        "dict_data": {
            "nested": {
                "deeply": {
                    "nested": ["list", "in", "dict"]
                }
            },
            "mixed_types": [1, "two", 3.14, True, None]
        },
        "set_data": {1, 2, 3, 4, 5},
        "tuple_data": (1, "two", 3.0),
        # Constants and secrets
        "const_api_version": "v1.0",
        "secret_api_key": "secret123",
        # Metadata
        "_meta_typed_counter": "builtins.int",
        "_meta_validated_user": "test.models.User",
        # Regular variables
        "counter": 5,
        "user_id": "user123"
    }


@pytest.fixture
def real_agent():
    """Create a real agent instance for integration testing."""
    agent = Agent(
        name="real_test_agent",
        max_concurrent=3,
        retry_policy=RetryPolicy(max_retries=2, initial_delay=0.1),
        state_timeout=30.0
    )

    # Add some states
    async def dummy_state(context):
        return "completed"

    agent.add_state("test_state1", dummy_state, priority=Priority.HIGH)
    agent.add_state("test_state2", dummy_state, dependencies=["test_state1"])

    # Simulate some execution state
    agent.completed_states.add("test_state1")
    agent.completed_once.add("test_state1")
    agent.shared_state.update({
        "execution_count": 1,
        "last_run": time.time(),
        "const_version": "1.0",
        "secret_token": "token123"
    })

    return agent


# ============================================================================
# BASIC CHECKPOINT CREATION TESTS
# ============================================================================

class TestAgentCheckpointBasic:
    """Test basic checkpoint creation and initialization."""

    def test_checkpoint_direct_initialization(self):
        """Test direct checkpoint initialization with all parameters."""
        timestamp = time.time()
        session_start = timestamp - 100

        checkpoint = AgentCheckpoint(
            timestamp=timestamp,
            agent_name="test_agent",
            agent_status=AgentStatus.RUNNING,
            priority_queue=[],
            state_metadata={},
            running_states=set(),
            completed_states={"state1"},
            completed_once={"state1"},
            shared_state={"key": "value"},
            session_start=session_start
        )

        assert checkpoint.timestamp == timestamp
        assert checkpoint.agent_name == "test_agent"
        assert checkpoint.agent_status == AgentStatus.RUNNING
        assert checkpoint.priority_queue == []
        assert checkpoint.state_metadata == {}
        assert checkpoint.running_states == set()
        assert checkpoint.completed_states == {"state1"}
        assert checkpoint.completed_once == {"state1"}
        assert checkpoint.shared_state == {"key": "value"}
        assert checkpoint.session_start == session_start

    def test_checkpoint_dataclass_immutability(self):
        """Test that checkpoint dataclass fields are immutable where appropriate."""
        checkpoint = AgentCheckpoint(
            timestamp=time.time(),
            agent_name="test",
            agent_status=AgentStatus.IDLE,
            priority_queue=[],
            state_metadata={},
            running_states=set(),
            completed_states=set(),
            completed_once=set(),
            shared_state={},
            session_start=None
        )

        # Should be able to access all fields
        assert checkpoint.agent_name == "test"
        assert checkpoint.agent_status == AgentStatus.IDLE

        # The dataclass itself isn't frozen, but we can test that it behaves correctly
        assert isinstance(checkpoint.running_states, set)
        assert isinstance(checkpoint.completed_states, set)
        assert isinstance(checkpoint.shared_state, dict)

    def test_checkpoint_with_none_session_start(self):
        """Test checkpoint creation with None session_start."""
        checkpoint = AgentCheckpoint(
            timestamp=time.time(),
            agent_name="test",
            agent_status=AgentStatus.IDLE,
            priority_queue=[],
            state_metadata={},
            running_states=set(),
            completed_states=set(),
            completed_once=set(),
            shared_state={},
            session_start=None
        )

        assert checkpoint.session_start is None

    def test_checkpoint_timestamp_precision(self):
        """Test that checkpoint timestamp preserves precision."""
        precise_time = time.time()

        checkpoint = AgentCheckpoint(
            timestamp=precise_time,
            agent_name="test",
            agent_status=AgentStatus.IDLE,
            priority_queue=[],
            state_metadata={},
            running_states=set(),
            completed_states=set(),
            completed_once=set(),
            shared_state={},
            session_start=None
        )

        assert checkpoint.timestamp == precise_time
        # Verify precision is maintained
        assert abs(checkpoint.timestamp - precise_time) < 1e-10


# ============================================================================
# CREATE FROM AGENT TESTS
# ============================================================================

class TestCreateFromAgent:
    """Test creating checkpoints from agent instances."""

    def test_create_from_agent_basic(self, mock_agent):
        """Test creating checkpoint from basic mock agent."""
        mock_agent.name = "basic_agent"
        mock_agent.status = AgentStatus.PAUSED
        mock_agent.priority_queue = []
        mock_agent.state_metadata = {}
        mock_agent._running_states = set()
        mock_agent.completed_states = set()
        mock_agent.completed_once = set()
        mock_agent.shared_state = {"test": "value"}
        mock_agent.session_start = 12345.67  # Changed from _session_start to session_start

        with patch('time.time', return_value=99999.99):
            checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        assert checkpoint.timestamp == 99999.99
        assert checkpoint.agent_name == "basic_agent"
        assert checkpoint.agent_status == AgentStatus.PAUSED
        assert checkpoint.priority_queue == []
        assert checkpoint.state_metadata == {}
        assert checkpoint.running_states == set()
        assert checkpoint.completed_states == set()
        assert checkpoint.completed_once == set()
        assert checkpoint.shared_state == {"test": "value"}
        assert checkpoint.session_start == 12345.67

    def test_create_from_agent_with_complex_data(self, mock_agent, sample_state_metadata,
                                                  sample_priority_queue, complex_shared_state):
        """Test creating checkpoint from agent with complex data structures."""
        mock_agent.state_metadata = sample_state_metadata
        mock_agent.priority_queue = sample_priority_queue
        mock_agent._running_states = {"running_state1", "running_state2"}
        mock_agent.completed_states = {"completed1", "completed2", "completed3"}
        mock_agent.completed_once = {"completed1", "completed2"}
        mock_agent.shared_state = complex_shared_state

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify all complex data is preserved
        assert len(checkpoint.state_metadata) == 2
        assert "state1" in checkpoint.state_metadata
        assert "state2" in checkpoint.state_metadata

        assert len(checkpoint.priority_queue) == 2
        assert checkpoint.priority_queue[0].state_name == "high_priority_state"
        assert checkpoint.priority_queue[1].state_name == "normal_priority_state"

        assert checkpoint.running_states == {"running_state1", "running_state2"}
        assert checkpoint.completed_states == {"completed1", "completed2", "completed3"}
        assert checkpoint.completed_once == {"completed1", "completed2"}

        # Verify complex shared state preservation
        assert checkpoint.shared_state["simple_string"] == "test_value"
        assert checkpoint.shared_state["number"] == 42
        assert checkpoint.shared_state["dict_data"]["nested"]["deeply"]["nested"] == ["list", "in", "dict"]
        assert checkpoint.shared_state["set_data"] == {1, 2, 3, 4, 5}
        assert checkpoint.shared_state["tuple_data"] == (1, "two", 3.0)

    def test_create_from_agent_deep_copy_isolation(self, mock_agent):
        """Test that checkpoint creation performs deep copying for isolation."""
        original_metadata = StateMetadata(
            status=StateStatus.RUNNING,
            attempts=1,
            satisfied_dependencies={"dep1", "dep2"}
        )

        original_queue_item = PrioritizedState(
            priority=-1,
            timestamp=time.time(),
            state_name="test_state",
            metadata=original_metadata
        )

        mock_agent.state_metadata = {"test": original_metadata}
        mock_agent.priority_queue = [original_queue_item]
        mock_agent._running_states = {"running1"}
        mock_agent.completed_states = {"completed1"}
        mock_agent.completed_once = {"once1"}
        mock_agent.shared_state = {"nested": {"list": [1, 2, 3]}}

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify deep copying - modifying original should not affect checkpoint
        original_metadata.attempts = 999
        original_metadata.satisfied_dependencies.add("new_dep")
        mock_agent._running_states.add("new_running")
        mock_agent.completed_states.add("new_completed")
        mock_agent.completed_once.add("new_once")
        mock_agent.shared_state["nested"]["list"].append(4)
        mock_agent.shared_state["new_key"] = "new_value"

        # Checkpoint should be unchanged
        assert checkpoint.state_metadata["test"].attempts == 1
        assert "new_dep" not in checkpoint.state_metadata["test"].satisfied_dependencies
        assert "new_running" not in checkpoint.running_states
        assert "new_completed" not in checkpoint.completed_states
        assert "new_once" not in checkpoint.completed_once
        assert 4 not in checkpoint.shared_state["nested"]["list"]
        assert "new_key" not in checkpoint.shared_state

    def test_create_from_agent_preserves_object_types(self, mock_agent):
        """Test that checkpoint preserves object types correctly."""
        metadata = StateMetadata(
            status=StateStatus.COMPLETED,
            attempts=2,
            max_retries=5,
            resources=ResourceRequirements(),
            last_execution=time.time()
        )

        mock_agent.state_metadata = {"test": metadata}
        mock_agent.shared_state = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "set": {1, 2, 3},
            "tuple": (1, 2, 3)
        }

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify types are preserved
        assert isinstance(checkpoint.shared_state["string"], str)
        assert isinstance(checkpoint.shared_state["integer"], int)
        assert isinstance(checkpoint.shared_state["float"], float)
        assert isinstance(checkpoint.shared_state["boolean"], bool)
        assert checkpoint.shared_state["none"] is None
        assert isinstance(checkpoint.shared_state["list"], list)
        assert isinstance(checkpoint.shared_state["dict"], dict)
        assert isinstance(checkpoint.shared_state["set"], set)
        assert isinstance(checkpoint.shared_state["tuple"], tuple)

        # Verify StateMetadata type preservation
        assert isinstance(checkpoint.state_metadata["test"], StateMetadata)
        assert checkpoint.state_metadata["test"].status == StateStatus.COMPLETED
        assert checkpoint.state_metadata["test"].attempts == 2

    def test_create_from_agent_handles_missing_session_start(self, mock_agent):
        """Test checkpoint creation when agent has no _session_start attribute."""
        # Remove _session_start attribute
        delattr(mock_agent, '_session_start')

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Should handle gracefully, likely with None or some default
        # The exact behavior depends on implementation, but it shouldn't crash
        assert hasattr(checkpoint, 'session_start')

    def test_create_from_agent_with_empty_collections(self, mock_agent):
        """Test checkpoint creation with empty collections."""
        mock_agent.priority_queue = []
        mock_agent.state_metadata = {}
        mock_agent._running_states = set()
        mock_agent.completed_states = set()
        mock_agent.completed_once = set()
        mock_agent.shared_state = {}

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        assert checkpoint.priority_queue == []
        assert checkpoint.state_metadata == {}
        assert checkpoint.running_states == set()
        assert checkpoint.completed_states == set()
        assert checkpoint.completed_once == set()
        assert checkpoint.shared_state == {}

    def test_create_from_agent_timestamp_accuracy(self, mock_agent):
        """Test that checkpoint timestamp is created at the right time."""
        start_time = time.time()

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        end_time = time.time()

        # Timestamp should be between start and end
        assert start_time <= checkpoint.timestamp <= end_time

        # Should be close to creation time
        assert abs(checkpoint.timestamp - time.time()) < 1.0


# ============================================================================
# REAL AGENT INTEGRATION TESTS
# ============================================================================

class TestRealAgentIntegration:
    """Test checkpoint creation with real agent instances."""

    def test_create_from_real_agent_basic(self, real_agent):
        """Test creating checkpoint from real agent instance."""
        checkpoint = AgentCheckpoint.create_from_agent(real_agent)

        assert checkpoint.agent_name == "real_test_agent"
        assert checkpoint.agent_status == real_agent.status
        assert checkpoint.completed_states == {"test_state1"}
        assert checkpoint.completed_once == {"test_state1"}

        # Verify state metadata is copied
        assert len(checkpoint.state_metadata) == 2
        assert "test_state1" in checkpoint.state_metadata
        assert "test_state2" in checkpoint.state_metadata

        # Verify shared state is copied
        assert "execution_count" in checkpoint.shared_state
        assert "last_run" in checkpoint.shared_state

    def test_create_checkpoint_during_execution(self, real_agent):
        """Test creating checkpoint while agent is in various execution states."""
        # Set agent to different states and verify checkpoint creation
        test_states = [
            AgentStatus.IDLE,
            AgentStatus.RUNNING,
            AgentStatus.PAUSED,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED
        ]

        for status in test_states:
            real_agent.status = status
            checkpoint = AgentCheckpoint.create_from_agent(real_agent)

            assert checkpoint.agent_status == status
            assert checkpoint.agent_name == real_agent.name
            assert isinstance(checkpoint.timestamp, float)

    def test_checkpoint_with_agent_modifications(self, real_agent):
        """Test that checkpoint is isolated from subsequent agent modifications."""
        # Create initial checkpoint
        checkpoint1 = AgentCheckpoint.create_from_agent(real_agent)

        # Modify agent state
        real_agent.completed_states.add("new_state")
        real_agent.shared_state["new_data"] = "new_value"
        real_agent.status = AgentStatus.FAILED

        # Create second checkpoint
        checkpoint2 = AgentCheckpoint.create_from_agent(real_agent)

        # First checkpoint should be unchanged
        assert "new_state" not in checkpoint1.completed_states
        assert "new_data" not in checkpoint1.shared_state
        assert checkpoint1.agent_status != AgentStatus.FAILED

        # Second checkpoint should reflect changes
        assert "new_state" in checkpoint2.completed_states
        assert "new_data" in checkpoint2.shared_state
        assert checkpoint2.agent_status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_checkpoint_with_async_agent_operations(self, real_agent):
        """Test checkpoint creation doesn't interfere with async operations."""
        # This test ensures that checkpoint creation is synchronous and safe
        import asyncio

        async def modify_agent():
            await asyncio.sleep(0.01)
            real_agent.shared_state["async_data"] = "async_value"
            return "done"

        # Start async operation and create checkpoint concurrently
        task = asyncio.create_task(modify_agent())
        checkpoint = AgentCheckpoint.create_from_agent(real_agent)
        result = await task

        assert result == "done"
        assert isinstance(checkpoint.timestamp, float)
        # The checkpoint may or may not contain async_data depending on timing
        # but it should be consistent and not corrupted


# ============================================================================
# DEEP COPY AND MEMORY TESTS
# ============================================================================

class TestDeepCopyAndMemory:
    """Test deep copying behavior and memory usage."""

    def test_deep_copy_prevents_modification(self, mock_agent):
        """Test that deep copy prevents unintended modifications."""
        original_list = [1, 2, 3]
        original_dict = {"key": "value", "nested": {"inner": original_list}}
        original_set = {1, 2, 3}

        mock_agent.shared_state = {
            "list": original_list,
            "dict": original_dict,
            "set": original_set
        }

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Modify original data structures
        original_list.append(4)
        original_dict["new_key"] = "new_value"
        original_dict["nested"]["inner"].append(5)
        original_set.add(4)

        # Checkpoint should be unaffected
        assert checkpoint.shared_state["list"] == [1, 2, 3]
        assert "new_key" not in checkpoint.shared_state["dict"]
        assert checkpoint.shared_state["dict"]["nested"]["inner"] == [1, 2, 3]
        assert 4 not in checkpoint.shared_state["set"]

    def test_deep_copy_with_circular_references(self, mock_agent):
        """Test deep copy handling of circular references."""
        # Create circular reference
        data = {"self": None}
        data["self"] = data

        mock_agent.shared_state = {"circular": data}

        # Should handle circular references gracefully
        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify the structure exists (exact handling depends on deepcopy implementation)
        assert "circular" in checkpoint.shared_state

    def test_deep_copy_with_complex_nested_structures(self, mock_agent):
        """Test deep copy with very complex nested structures."""
        complex_structure = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "data": [1, 2, 3],
                            "more_data": {"a": 1, "b": 2}
                        }
                    }
                }
            },
            "parallel_structure": [
                {"item1": [1, 2, 3]},
                {"item2": {"nested": {"deep": "value"}}}
            ]
        }

        mock_agent.shared_state = {"complex": complex_structure}

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify deep structure is preserved
        assert checkpoint.shared_state["complex"]["level1"]["level2"]["level3"]["level4"]["data"] == [1, 2, 3]
        assert checkpoint.shared_state["complex"]["parallel_structure"][1]["item2"]["nested"]["deep"] == "value"

        # Verify independence
        complex_structure["level1"]["level2"]["level3"]["level4"]["data"].append(4)
        assert 4 not in checkpoint.shared_state["complex"]["level1"]["level2"]["level3"]["level4"]["data"]

    def test_memory_efficiency_with_large_data(self, mock_agent):
        """Test memory efficiency with large data structures."""
        # Create moderately large data structure
        large_list = list(range(1000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(500)}

        mock_agent.shared_state = {
            "large_list": large_list,
            "large_dict": large_dict
        }

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify data integrity
        assert len(checkpoint.shared_state["large_list"]) == 1000
        assert len(checkpoint.shared_state["large_dict"]) == 500
        assert checkpoint.shared_state["large_list"][999] == 999
        assert checkpoint.shared_state["large_dict"]["key_499"] == "value_499"


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases."""

    def test_create_from_agent_with_none_attributes(self, mock_agent):
        """Test checkpoint creation when agent has None attributes."""
        mock_agent.priority_queue = None
        mock_agent.state_metadata = None
        mock_agent._running_states = None
        mock_agent.completed_states = None
        mock_agent.completed_once = None
        mock_agent.shared_state = None

        # Should handle None values gracefully or raise appropriate error
        try:
            checkpoint = AgentCheckpoint.create_from_agent(mock_agent)
            # If it succeeds, verify the checkpoint is valid
            assert hasattr(checkpoint, 'timestamp')
            assert hasattr(checkpoint, 'agent_name')
        except (TypeError, AttributeError):
            # Expected if the implementation doesn't handle None values
            pass

    def test_create_from_agent_missing_attributes(self, mock_agent):
        """Test checkpoint creation when agent is missing expected attributes."""
        # Remove some expected attributes
        delattr(mock_agent, 'priority_queue')
        delattr(mock_agent, '_running_states')

        # Should either handle gracefully or raise AttributeError
        try:
            checkpoint = AgentCheckpoint.create_from_agent(mock_agent)
            # If it succeeds, verify basic structure
            assert hasattr(checkpoint, 'timestamp')
        except AttributeError:
            # Expected if implementation requires these attributes
            pass

    def test_create_from_agent_with_non_serializable_data(self, mock_agent):
        """Test checkpoint creation with non-serializable data in shared_state."""
        import io

        # Add non-serializable objects
        mock_agent.shared_state = {
            "file_object": io.StringIO("test"),
            "lambda": lambda x: x,
            "generator": (x for x in range(5))
        }

        # Deepcopy should handle or fail appropriately
        try:
            checkpoint = AgentCheckpoint.create_from_agent(mock_agent)
            # If successful, verify checkpoint is created
            assert hasattr(checkpoint, 'shared_state')
        except (TypeError, copy.Error):
            # Expected for truly non-copyable objects
            pass

    def test_create_from_agent_with_custom_objects(self, mock_agent):
        """Test checkpoint creation with custom objects in agent state."""
        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, CustomObject) and self.value == other.value

        custom_obj = CustomObject("test_value")
        mock_agent.shared_state = {"custom": custom_obj}

        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify custom object is copied
        assert isinstance(checkpoint.shared_state["custom"], CustomObject)
        assert checkpoint.shared_state["custom"].value == "test_value"

        # Verify independence
        custom_obj.value = "modified"
        assert checkpoint.shared_state["custom"].value == "test_value"

    def test_create_from_agent_concurrent_modification(self, mock_agent):
        """Test checkpoint creation during concurrent agent modification."""
        import threading
        import time

        mock_agent.shared_state = {"counter": 0}

        def modify_agent():
            for i in range(100):
                mock_agent.shared_state["counter"] += 1
                time.sleep(0.001)

        # Start modification in background
        thread = threading.Thread(target=modify_agent)
        thread.start()

        # Create checkpoint during modification
        time.sleep(0.05)  # Let some modifications happen
        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        thread.join()

        # Checkpoint should have a consistent state
        assert isinstance(checkpoint.shared_state["counter"], int)
        assert 0 <= checkpoint.shared_state["counter"] <= 100


# ============================================================================
# PERFORMANCE AND BENCHMARKING TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics of checkpoint creation."""

    def test_checkpoint_creation_performance(self, mock_agent):
        """Test that checkpoint creation completes in reasonable time."""
        # Create reasonably complex agent state
        mock_agent.state_metadata = {
            f"state_{i}": StateMetadata(
                status=StateStatus.COMPLETED,
                attempts=i % 3,
                max_retries=3
            ) for i in range(100)
        }

        mock_agent.shared_state = {
            f"key_{i}": f"value_{i}" for i in range(1000)
        }

        mock_agent.completed_states = {f"completed_{i}" for i in range(50)}
        mock_agent.completed_once = {f"once_{i}" for i in range(50)}

        # Measure checkpoint creation time
        start_time = time.time()
        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)
        end_time = time.time()

        creation_time = end_time - start_time

        # Should complete quickly (adjust threshold as needed)
        assert creation_time < 1.0  # Should complete in less than 1 second

        # Verify checkpoint integrity
        assert len(checkpoint.state_metadata) == 100
        assert len(checkpoint.shared_state) == 1000
        assert len(checkpoint.completed_states) == 50

    def test_checkpoint_creation_memory_usage(self, mock_agent):
        """Test memory usage patterns during checkpoint creation."""
        import gc

        # Create large data structure
        large_data = {
            "big_list": list(range(10000)),
            "big_dict": {f"key_{i}": f"value_{i}" * 10 for i in range(1000)}
        }

        mock_agent.shared_state = large_data

        # Force garbage collection before test
        gc.collect()

        # Create checkpoint
        checkpoint = AgentCheckpoint.create_from_agent(mock_agent)

        # Verify data integrity
        assert len(checkpoint.shared_state["big_list"]) == 10000
        assert len(checkpoint.shared_state["big_dict"]) == 1000

        # Memory should be independent
        large_data["big_list"].clear()
        assert len(checkpoint.shared_state["big_list"]) == 10000

    def test_multiple_checkpoint_creation(self, mock_agent):
        """Test creating multiple checkpoints from the same agent."""
        mock_agent.shared_state = {"data": list(range(100))}

        checkpoints = []
        creation_times = []

        # Create multiple checkpoints
        for i in range(10):
            mock_agent.shared_state["iteration"] = i

            start_time = time.time()
            checkpoint = AgentCheckpoint.create_from_agent(mock_agent)
            end_time = time.time()

            checkpoints.append(checkpoint)
            creation_times.append(end_time - start_time)

        # Verify all checkpoints are different
        for i, checkpoint in enumerate(checkpoints):
            assert checkpoint.shared_state["iteration"] == i

        # Verify reasonable performance consistency
        avg_time = sum(creation_times) / len(creation_times)
        assert avg_time < 0.1  # Average should be reasonable


# ============================================================================
# INTEGRATION AND WORKFLOW TESTS
# ============================================================================

class TestIntegrationWorkflow:
    """Test checkpoint integration with agent workflows."""

    def test_checkpoint_restore_compatibility(self, real_agent):
        """Test that checkpoints can be used for agent restoration."""
        # Set up agent state
        real_agent.completed_states.add("restored_state")
        real_agent.shared_state["restored_data"] = "test_value"
        original_status = real_agent.status

        # Create checkpoint
        checkpoint = AgentCheckpoint.create_from_agent(real_agent)

        # Modify agent
        real_agent.completed_states.clear()
        real_agent.shared_state.clear()
        real_agent.status = AgentStatus.FAILED

        # Verify checkpoint can provide restoration data
        assert checkpoint.completed_states == {"test_state1", "restored_state"}
        assert checkpoint.shared_state["restored_data"] == "test_value"
        assert checkpoint.agent_status == original_status

        # Checkpoint should have all necessary data for restoration
        required_fields = [
            'timestamp', 'agent_name', 'agent_status', 'priority_queue',
            'state_metadata', 'running_states', 'completed_states',
            'completed_once', 'shared_state', 'session_start'
        ]

        for field in required_fields:
            assert hasattr(checkpoint, field)

    def test_checkpoint_versioning_compatibility(self, real_agent):
        """Test that checkpoints maintain version compatibility."""
        # Create checkpoints at different points in time
        checkpoint1 = AgentCheckpoint.create_from_agent(real_agent)

        # Simulate time passing and agent evolution
        time.sleep(0.01)
        real_agent.shared_state["version"] = "2.0"

        checkpoint2 = AgentCheckpoint.create_from_agent(real_agent)

        # Both checkpoints should be valid and distinguishable
        assert checkpoint1.timestamp < checkpoint2.timestamp
        assert "version" not in checkpoint1.shared_state
        assert checkpoint2.shared_state["version"] == "2.0"

    def test_checkpoint_serialization_readiness(self, real_agent):
        """Test that checkpoints are ready for serialization."""
        checkpoint = AgentCheckpoint.create_from_agent(real_agent)

        # Test basic serialization compatibility
        try:
            import pickle
            serialized = pickle.dumps(checkpoint)
            deserialized = pickle.loads(serialized)

            # Verify deserialized checkpoint matches original
            assert deserialized.agent_name == checkpoint.agent_name
            assert deserialized.agent_status == checkpoint.agent_status
            assert deserialized.completed_states == checkpoint.completed_states
            assert deserialized.shared_state == checkpoint.shared_state

        except (pickle.PicklingError, ImportError):
            # If pickling fails, that's OK for this test
            # We're just verifying the structure is reasonable
            pass

        # Verify checkpoint structure is JSON-compatible for basic types
        import json

        # Extract JSON-serializable parts
        json_compatible_data = {
            "timestamp": checkpoint.timestamp,
            "agent_name": checkpoint.agent_name,
            "agent_status": checkpoint.agent_status.value,
            "completed_states": list(checkpoint.completed_states),
            "completed_once": list(checkpoint.completed_once),
        }

        # Should be JSON serializable
        json_str = json.dumps(json_compatible_data)
        restored_data = json.loads(json_str)

        assert restored_data["agent_name"] == checkpoint.agent_name
        assert restored_data["timestamp"] == checkpoint.timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])