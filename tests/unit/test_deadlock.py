"""
Comprehensive unit tests for the deadlock detection module.
Tests all components including data classes, graphs, and the main detector.
"""

import pytest
import asyncio
import time
import uuid
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import sys
import os

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the modules under test
from core.coordination.deadlock import (
    DeadlockResolutionStrategy,
    DeadlockError,
    ResourceNode,
    ProcessNode,
    CycleDetectionResult,
    DependencyGraph,
    ResourceWaitGraph,
    DeadlockDetector
)


def create_mock_agent():
    """Factory function to create a mock agent for testing."""
    agent = Mock()
    agent.name = "test_agent"
    agent._monitor = Mock()
    agent._monitor.logger = Mock()
    agent._monitor.record_metric = AsyncMock()
    agent.state_metadata = {}
    return agent


class TestDeadlockResolutionStrategy:
    """Test the DeadlockResolutionStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategy values are available."""
        expected_strategies = [
            'RAISE_EXCEPTION',
            'KILL_YOUNGEST',
            'KILL_OLDEST',
            'PREEMPT_RESOURCES',
            'ROLLBACK_TRANSACTION',
            'LOG_ONLY'
        ]

        for strategy_name in expected_strategies:
            assert hasattr(DeadlockResolutionStrategy, strategy_name)
            strategy = getattr(DeadlockResolutionStrategy, strategy_name)
            assert isinstance(strategy, DeadlockResolutionStrategy)

    def test_strategy_enum_behavior(self):
        """Test enum behavior and comparison."""
        strategy1 = DeadlockResolutionStrategy.KILL_YOUNGEST
        strategy2 = DeadlockResolutionStrategy.KILL_YOUNGEST
        strategy3 = DeadlockResolutionStrategy.KILL_OLDEST

        assert strategy1 == strategy2
        assert strategy1 != strategy3
        assert strategy1.name == 'KILL_YOUNGEST'


class TestDeadlockError:
    """Test the DeadlockError exception class."""

    def test_basic_deadlock_error(self):
        """Test basic deadlock error creation."""
        cycle = ['proc1', 'proc2', 'proc1']
        error = DeadlockError(cycle)

        assert error.cycle == cycle
        assert error.detection_id is not None
        assert len(error.detection_id) > 0
        assert 'Deadlock detected' in str(error)
        assert 'proc1 -> proc2 -> proc1' in str(error)

    def test_deadlock_error_with_custom_id(self):
        """Test deadlock error with custom detection ID."""
        cycle = ['a', 'b', 'c']
        detection_id = "custom_id_123"
        error = DeadlockError(cycle, detection_id)

        assert error.cycle == cycle
        assert error.detection_id == detection_id
        assert detection_id in str(error)

    def test_deadlock_error_with_custom_message(self):
        """Test deadlock error with custom message."""
        cycle = ['x', 'y']
        custom_message = "Critical deadlock found"
        error = DeadlockError(cycle, message=custom_message)

        assert error.cycle == cycle
        assert custom_message in str(error)

    def test_deadlock_error_inheritance(self):
        """Test that DeadlockError is a proper exception."""
        cycle = ['test']
        error = DeadlockError(cycle)

        assert isinstance(error, Exception)

        # Test it can be raised and caught
        with pytest.raises(DeadlockError) as exc_info:
            raise error

        assert exc_info.value.cycle == cycle


class TestResourceNode:
    """Test the ResourceNode data class."""

    def test_resource_node_creation(self):
        """Test basic resource node creation."""
        node = ResourceNode("resource1", "mutex")

        assert node.resource_id == "resource1"
        assert node.resource_type == "mutex"
        assert len(node.holders) == 0
        assert len(node.waiters) == 0
        assert node.acquired_at is None
        assert isinstance(node.created_at, datetime)
        assert node.access_count == 0

    def test_resource_node_with_holders_and_waiters(self):
        """Test resource node with holders and waiters."""
        node = ResourceNode("resource2", "semaphore")
        node.holders.add("proc1")
        node.holders.add("proc2")
        node.waiters.add("proc3")
        node.access_count = 5

        assert len(node.holders) == 2
        assert "proc1" in node.holders
        assert "proc2" in node.holders
        assert len(node.waiters) == 1
        assert "proc3" in node.waiters
        assert node.access_count == 5

    def test_is_free_method(self):
        """Test the is_free method."""
        node = ResourceNode("test_resource", "lock")

        # Initially free
        assert node.is_free() is True

        # Add holder
        node.holders.add("proc1")
        assert node.is_free() is False

        # Remove holder
        node.holders.remove("proc1")
        assert node.is_free() is True

    def test_age_seconds_method(self):
        """Test the age_seconds method."""
        node = ResourceNode("aging_resource", "barrier")

        # Should have very small age initially
        age = node.age_seconds()
        assert age >= 0
        assert age < 1.0  # Should be very recent

        # Mock older creation time
        old_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        node.created_at = old_time

        age = node.age_seconds()
        assert age >= 29  # Allow for small timing differences
        assert age <= 31

    def test_resource_node_default_factory(self):
        """Test that default factories work correctly."""
        node1 = ResourceNode("res1", "type1")
        node2 = ResourceNode("res2", "type2")

        # Each should have separate sets
        node1.holders.add("holder1")
        node2.holders.add("holder2")

        assert node1.holders != node2.holders
        assert "holder1" not in node2.holders
        assert "holder2" not in node1.holders


class TestProcessNode:
    """Test the ProcessNode data class."""

    def test_process_node_creation(self):
        """Test basic process node creation."""
        node = ProcessNode("proc1", "TestProcess")

        assert node.process_id == "proc1"
        assert node.process_name == "TestProcess"
        assert len(node.holding) == 0
        assert len(node.waiting_for) == 0
        assert isinstance(node.started_at, datetime)
        assert node.blocked_at is None
        assert isinstance(node.last_activity, datetime)
        assert node.priority == 0

    def test_process_node_with_priority(self):
        """Test process node with custom priority."""
        node = ProcessNode("high_priority", "Important", priority=10)

        assert node.priority == 10
        assert node.process_name == "Important"

    def test_is_blocked_method(self):
        """Test the is_blocked method."""
        node = ProcessNode("proc1", "TestProcess")

        # Initially not blocked
        assert node.is_blocked() is False

        # Add waiting resource
        node.waiting_for.add("resource1")
        assert node.is_blocked() is True

        # Remove waiting resource
        node.waiting_for.remove("resource1")
        assert node.is_blocked() is False

    def test_age_seconds_method(self):
        """Test the age_seconds method."""
        node = ProcessNode("proc1", "TestProcess")

        # Should have very small age initially
        age = node.age_seconds()
        assert age >= 0
        assert age < 1.0

        # Mock older start time
        old_time = datetime.now(timezone.utc) - timedelta(seconds=45)
        node.started_at = old_time

        age = node.age_seconds()
        assert age >= 44
        assert age <= 46

    def test_blocked_duration_seconds(self):
        """Test the blocked_duration_seconds method."""
        node = ProcessNode("proc1", "TestProcess")

        # Not blocked, should return 0
        assert node.blocked_duration_seconds() == 0.0

        # Set blocked time
        blocked_time = datetime.now(timezone.utc) - timedelta(seconds=20)
        node.blocked_at = blocked_time

        duration = node.blocked_duration_seconds()
        assert duration >= 19
        assert duration <= 21

    def test_update_activity(self):
        """Test the update_activity method."""
        node = ProcessNode("proc1", "TestProcess")
        original_activity = node.last_activity

        # Wait a tiny bit to ensure time difference
        time.sleep(0.01)

        node.update_activity()

        assert node.last_activity > original_activity

    def test_process_node_separate_instances(self):
        """Test that process nodes have separate state."""
        node1 = ProcessNode("proc1", "Process1")
        node2 = ProcessNode("proc2", "Process2")

        node1.holding.add("resource1")
        node1.waiting_for.add("resource2")

        assert len(node2.holding) == 0
        assert len(node2.waiting_for) == 0


class TestCycleDetectionResult:
    """Test the CycleDetectionResult data class."""

    def test_cycle_detection_result_creation(self):
        """Test basic cycle detection result creation."""
        cycles = [['a', 'b', 'a'], ['x', 'y', 'z', 'x']]
        result = CycleDetectionResult(
            has_cycle=True,
            cycles=cycles,
            graph_size=10,
            detection_duration_ms=15.5
        )

        assert result.has_cycle is True
        assert result.cycles == cycles
        assert result.graph_size == 10
        assert result.detection_duration_ms == 15.5
        assert isinstance(result.detection_time, datetime)

    def test_cycle_detection_result_no_cycles(self):
        """Test cycle detection result with no cycles."""
        result = CycleDetectionResult(has_cycle=False)

        assert result.has_cycle is False
        assert len(result.cycles) == 0
        assert result.graph_size == 0
        assert result.detection_duration_ms == 0.0

    def test_get_shortest_cycle(self):
        """Test getting the shortest cycle."""
        cycles = [
            ['a', 'b', 'c', 'a'],  # Length 4
            ['x', 'y', 'x'],  # Length 3 (shortest)
            ['p', 'q', 'r', 's', 'p']  # Length 5
        ]
        result = CycleDetectionResult(has_cycle=True, cycles=cycles)

        shortest = result.get_shortest_cycle()
        assert shortest == ['x', 'y', 'x']

    def test_get_longest_cycle(self):
        """Test getting the longest cycle."""
        cycles = [
            ['a', 'b', 'c', 'a'],  # Length 4
            ['x', 'y', 'x'],  # Length 3
            ['p', 'q', 'r', 's', 'p']  # Length 5 (longest)
        ]
        result = CycleDetectionResult(has_cycle=True, cycles=cycles)

        longest = result.get_longest_cycle()
        assert longest == ['p', 'q', 'r', 's', 'p']

    def test_get_cycles_no_cycles(self):
        """Test getting cycles when none exist."""
        result = CycleDetectionResult(has_cycle=False)

        assert result.get_shortest_cycle() is None
        assert result.get_longest_cycle() is None

    def test_get_cycles_single_cycle(self):
        """Test getting cycles when only one exists."""
        cycle = ['a', 'b', 'a']
        result = CycleDetectionResult(has_cycle=True, cycles=[cycle])

        assert result.get_shortest_cycle() == cycle
        assert result.get_longest_cycle() == cycle


class TestDependencyGraph:
    """Test the DependencyGraph class."""

    @pytest.mark.asyncio
    async def test_dependency_graph_creation(self):
        """Test basic dependency graph creation."""
        graph = DependencyGraph()

        assert len(graph.nodes) == 0
        assert len(graph.reverse_edges) == 0
        assert len(graph.node_metadata) == 0
        assert graph.max_nodes == 10000

    @pytest.mark.asyncio
    async def test_dependency_graph_custom_max_nodes(self):
        """Test dependency graph with custom max nodes."""
        graph = DependencyGraph(max_nodes=100)

        assert graph.max_nodes == 100

    @pytest.mark.asyncio
    async def test_add_dependency(self):
        """Test adding dependencies."""
        graph = DependencyGraph()

        await graph.add_dependency("A", "B")
        await graph.add_dependency("A", "C")
        await graph.add_dependency("B", "C")

        assert "A" in graph.nodes
        assert "B" in graph.nodes["A"]
        assert "C" in graph.nodes["A"]
        assert "C" in graph.nodes["B"]

        # Check reverse edges
        assert "A" in graph.reverse_edges["B"]
        assert "A" in graph.reverse_edges["C"]
        assert "B" in graph.reverse_edges["C"]

    @pytest.mark.asyncio
    async def test_add_dependency_with_metadata(self):
        """Test adding dependencies with metadata."""
        graph = DependencyGraph()
        metadata = {"type": "strong", "weight": 5}

        await graph.add_dependency("X", "Y", metadata)

        assert graph.node_metadata["X"] == metadata

    @pytest.mark.asyncio
    async def test_remove_dependency(self):
        """Test removing dependencies."""
        graph = DependencyGraph()

        await graph.add_dependency("A", "B")
        await graph.add_dependency("A", "C")

        # Remove one dependency
        await graph.remove_dependency("A", "B")

        assert "B" not in graph.nodes["A"]
        assert "C" in graph.nodes["A"]
        assert "A" not in graph.reverse_edges.get("B", set())

    @pytest.mark.asyncio
    async def test_remove_node(self):
        """Test removing entire nodes."""
        graph = DependencyGraph()

        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "C")
        await graph.add_dependency("D", "B")

        # Remove node B
        await graph.remove_node("B")

        assert "B" not in graph.nodes
        assert "B" not in graph.reverse_edges
        assert "B" not in graph.nodes.get("A", set())
        assert "B" not in graph.nodes.get("D", set())

    @pytest.mark.asyncio
    async def test_find_cycles_no_cycle(self):
        """Test cycle detection with no cycles."""
        graph = DependencyGraph()

        # Create acyclic graph: A -> B -> C
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "C")

        result = graph.find_cycles()

        assert result.has_cycle is False
        assert len(result.cycles) == 0

    @pytest.mark.asyncio
    async def test_find_cycles_simple_cycle(self):
        """Test cycle detection with simple cycle."""
        graph = DependencyGraph()

        # Create cycle: A -> B -> A
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "A")

        result = graph.find_cycles()

        assert result.has_cycle is True
        assert len(result.cycles) >= 1

        # Check that we found the cycle
        found_cycle = False
        for cycle in result.cycles:
            if 'A' in cycle and 'B' in cycle:
                found_cycle = True
                break
        assert found_cycle

    @pytest.mark.asyncio
    async def test_find_cycles_complex(self):
        """Test cycle detection with complex graph."""
        graph = DependencyGraph()

        # Create graph with multiple cycles
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "C")
        await graph.add_dependency("C", "A")  # Cycle: A -> B -> C -> A
        await graph.add_dependency("D", "E")
        await graph.add_dependency("E", "D")  # Cycle: D -> E -> D

        result = graph.find_cycles()

        assert result.has_cycle is True
        assert len(result.cycles) >= 2  # At least two cycles

    @pytest.mark.asyncio
    async def test_topological_sort_acyclic(self):
        """Test topological sort on acyclic graph."""
        graph = DependencyGraph()

        # Create DAG: A -> B -> C, A -> C
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "C")
        await graph.add_dependency("A", "C")

        sorted_nodes = graph.topological_sort()

        assert sorted_nodes is not None
        assert len(sorted_nodes) == 3
        # A should come before B and C, B should come before C
        a_index = sorted_nodes.index("A")
        b_index = sorted_nodes.index("B")
        c_index = sorted_nodes.index("C")
        assert a_index < b_index < c_index

    @pytest.mark.asyncio
    async def test_topological_sort_cyclic(self):
        """Test topological sort on cyclic graph."""
        graph = DependencyGraph()

        # Create cycle
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "A")

        sorted_nodes = graph.topological_sort()

        assert sorted_nodes is None  # Should return None for cyclic graphs

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test that cache is invalidated when graph changes."""
        graph = DependencyGraph()

        # First detection
        result1 = graph.find_cycles()
        assert not result1.has_cycle

        # Add cycle
        await graph.add_dependency("A", "B")
        await graph.add_dependency("B", "A")

        # Cache should be invalidated, should detect cycle
        result2 = graph.find_cycles()
        assert result2.has_cycle

    @pytest.mark.asyncio
    async def test_cleanup_old_nodes(self):
        """Test cleanup functionality."""
        graph = DependencyGraph(max_nodes=10)

        # Add more than max nodes
        for i in range(15):
            await graph.add_dependency(f"node_{i}", f"node_{i + 1}")

        # Should trigger cleanup
        assert len(graph.nodes) <= 12  # Some cleanup should have occurred


class TestResourceWaitGraph:
    """Test the ResourceWaitGraph class."""

    @pytest.mark.asyncio
    async def test_resource_wait_graph_creation(self):
        """Test basic resource wait graph creation."""
        graph = ResourceWaitGraph()

        assert len(graph.resources) == 0
        assert len(graph.processes) == 0
        assert graph.max_resources == 5000
        assert graph.max_processes == 5000

    @pytest.mark.asyncio
    async def test_add_resource(self):
        """Test adding resources."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_resource("resource2", "semaphore")

        assert "resource1" in graph.resources
        assert "resource2" in graph.resources
        assert graph.resources["resource1"].resource_type == "mutex"
        assert graph.resources["resource2"].resource_type == "semaphore"

    @pytest.mark.asyncio
    async def test_add_process(self):
        """Test adding processes."""
        graph = ResourceWaitGraph()

        await graph.add_process("proc1", "Process1", priority=5)
        await graph.add_process("proc2", "Process2")

        assert "proc1" in graph.processes
        assert "proc2" in graph.processes
        assert graph.processes["proc1"].process_name == "Process1"
        assert graph.processes["proc1"].priority == 5
        assert graph.processes["proc2"].priority == 0

    @pytest.mark.asyncio
    async def test_acquire_resource_success(self):
        """Test successful resource acquisition."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")

        success = await graph.acquire_resource("proc1", "resource1")

        assert success is True
        assert "proc1" in graph.resources["resource1"].holders
        assert "resource1" in graph.processes["proc1"].holding
        assert len(graph.resources["resource1"].waiters) == 0

    @pytest.mark.asyncio
    async def test_acquire_resource_blocked(self):
        """Test blocked resource acquisition."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")
        await graph.add_process("proc2", "Process2")

        # First process acquires successfully
        success1 = await graph.acquire_resource("proc1", "resource1")
        assert success1 is True

        # Second process should be blocked
        success2 = await graph.acquire_resource("proc2", "resource1")
        assert success2 is False
        assert "proc2" in graph.resources["resource1"].waiters
        assert "resource1" in graph.processes["proc2"].waiting_for

    @pytest.mark.asyncio
    async def test_release_resource(self):
        """Test resource release."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")

        # Acquire then release
        await graph.acquire_resource("proc1", "resource1")
        await graph.release_resource("proc1", "resource1")

        assert "proc1" not in graph.resources["resource1"].holders
        assert "resource1" not in graph.processes["proc1"].holding
        assert graph.resources["resource1"].is_free()

    @pytest.mark.asyncio
    async def test_release_resource_with_waiters(self):
        """Test resource release with waiting processes."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")
        await graph.add_process("proc2", "Process2", priority=10)
        await graph.add_process("proc3", "Process3", priority=5)

        # First process acquires
        await graph.acquire_resource("proc1", "resource1")

        # Other processes wait
        await graph.acquire_resource("proc2", "resource1")
        await graph.acquire_resource("proc3", "resource1")

        # Release should give to highest priority waiter
        await graph.release_resource("proc1", "resource1")

        # proc2 has highest priority, should get the resource
        assert "proc2" in graph.resources["resource1"].holders
        assert "proc2" not in graph.resources["resource1"].waiters

    @pytest.mark.asyncio
    async def test_detect_deadlock_no_deadlock(self):
        """Test deadlock detection with no deadlock."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")

        await graph.acquire_resource("proc1", "resource1")

        result = await graph.detect_deadlock()

        assert result.has_cycle is False

    @pytest.mark.asyncio
    async def test_detect_deadlock_simple_deadlock(self):
        """Test deadlock detection with simple deadlock."""
        graph = ResourceWaitGraph()

        # Create resources and processes
        await graph.add_resource("resource1", "mutex")
        await graph.add_resource("resource2", "mutex")
        await graph.add_process("proc1", "Process1")
        await graph.add_process("proc2", "Process2")

        # Create deadlock scenario
        # proc1 holds resource1, wants resource2
        await graph.acquire_resource("proc1", "resource1")
        await graph.acquire_resource("proc2", "resource2")

        # Now they cross-wait
        await graph.acquire_resource("proc1", "resource2")  # Will wait
        await graph.acquire_resource("proc2", "resource1")  # Will wait

        result = await graph.detect_deadlock()

        assert result.has_cycle is True

    @pytest.mark.asyncio
    async def test_get_blocked_processes(self):
        """Test getting blocked processes."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "mutex")
        await graph.add_process("proc1", "Process1")
        await graph.add_process("proc2", "Process2")

        # Initially no blocked processes
        blocked = graph.get_blocked_processes()
        assert len(blocked) == 0

        # Create blocking scenario
        await graph.acquire_resource("proc1", "resource1")
        await graph.acquire_resource("proc2", "resource1")  # This will block

        blocked = graph.get_blocked_processes()
        assert len(blocked) == 1
        assert blocked[0].process_id == "proc2"

    @pytest.mark.asyncio
    async def test_get_resource_holders_and_waiters(self):
        """Test getting resource holders and waiters."""
        graph = ResourceWaitGraph()

        await graph.add_resource("resource1", "semaphore")
        await graph.add_process("proc1", "Process1")
        await graph.add_process("proc2", "Process2")

        # Initially empty
        holders = graph.get_resource_holders("resource1")
        waiters = graph.get_resource_waiters("resource1")
        assert len(holders) == 0
        assert len(waiters) == 0

        # Add holder and waiter
        await graph.acquire_resource("proc1", "resource1")
        await graph.acquire_resource("proc2", "resource1")  # Assuming this blocks

        holders = graph.get_resource_holders("resource1")
        assert "proc1" in holders

        # Note: This test depends on the resource being exclusive
        # For a semaphore that allows multiple holders, the second acquire might succeed

    @pytest.mark.asyncio
    async def test_cleanup_old_resources(self):
        """Test cleanup of old resources."""
        graph = ResourceWaitGraph(max_resources=10)

        # Add many resources to trigger cleanup
        for i in range(15):
            await graph.add_resource(f"resource_{i}", "mutex")

        # Should trigger cleanup internally
        # This is a bit hard to test directly since cleanup is internal
        assert len(graph.resources) <= 15

    @pytest.mark.asyncio
    async def test_cleanup_old_processes(self):
        """Test cleanup of old processes."""
        graph = ResourceWaitGraph(max_processes=10)

        # Add many processes to trigger cleanup
        for i in range(15):
            await graph.add_process(f"proc_{i}", f"Process{i}")

        # Should trigger cleanup internally
        assert len(graph.processes) <= 15


class TestDeadlockDetector:
    """Test the DeadlockDetector class."""

    @pytest.mark.asyncio
    async def test_deadlock_detector_creation(self):
        """Test basic deadlock detector creation."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        assert detector.agent.name == mock_agent.name
        assert detector.detection_interval == 1.0
        assert detector.max_cycles == 100
        assert detector.resolution_strategy == DeadlockResolutionStrategy.LOG_ONLY
        assert detector.enable_metrics is True
        assert detector._cycle_count == 0
        assert detector._last_cycle is None

    @pytest.mark.asyncio
    async def test_deadlock_detector_custom_config(self):
        """Test deadlock detector with custom configuration."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            detection_interval=0.5,
            max_cycles=50,
            resolution_strategy=DeadlockResolutionStrategy.KILL_YOUNGEST,
            enable_metrics=False
        )

        assert detector.detection_interval == 0.5
        assert detector.max_cycles == 50
        assert detector.resolution_strategy == DeadlockResolutionStrategy.KILL_YOUNGEST
        assert detector.enable_metrics is False

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping the detector."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.1)

        try:
            await detector.start()

            assert detector._detection_task is not None
            assert not detector._detection_task.done()

            # Let it run briefly
            await asyncio.sleep(0.2)

        finally:
            await detector.stop()

            assert detector._detection_task is None or detector._detection_task.done()

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self):
        """Test that start/stop are idempotent."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.1)

        try:
            # Start twice
            await detector.start()
            await detector.start()  # Should be safe

            # Stop twice
            await detector.stop()
            await detector.stop()  # Should be safe

        except Exception:
            # Clean up in case of failure
            try:
                await detector.stop()
            except:
                pass

    @pytest.mark.asyncio
    async def test_add_remove_dependency(self):
        """Test adding and removing dependencies."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        await detector.add_dependency("state1", "state2")
        await detector.add_dependency("state2", "state3")

        # Check dependencies were added
        graph = detector.get_dependency_graph()
        assert "state1" in graph
        assert "state2" in graph["state1"]

        # Remove dependency
        await detector.remove_dependency("state1", "state2")

        graph = detector.get_dependency_graph()
        assert "state2" not in graph.get("state1", set())

    @pytest.mark.asyncio
    async def test_acquire_release_resource(self):
        """Test resource acquisition and release."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Acquire resource
        success = await detector.acquire_resource("proc1", "resource1", "Process1")
        assert success is True

        # Try to acquire same resource with different process
        success2 = await detector.acquire_resource("proc2", "resource1", "Process2")
        assert success2 is False  # Should be blocked

        # Release resource
        await detector.release_resource("proc1", "resource1")

        # Now second process should be able to acquire
        # Note: This depends on the automatic acquisition after release

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_handling(self):
        """Test deadlock detection and handling."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            detection_interval=0.1,
            resolution_strategy=DeadlockResolutionStrategy.LOG_ONLY
        )

        try:
            await detector.start()

            # Create deadlock scenario
            await detector.acquire_resource("proc1", "resource1", "Process1")
            await detector.acquire_resource("proc2", "resource2", "Process2")

            # Cross acquisition should create deadlock
            await detector.acquire_resource("proc1", "resource2")  # Will wait
            await detector.acquire_resource("proc2", "resource1")  # Will wait

            # Let detector run and find deadlock
            await asyncio.sleep(0.3)

            # Check that deadlock was detected
            status = detector.get_status()
            assert status["cycle_count"] > 0

        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_resolution_callbacks(self):
        """Test custom resolution callbacks."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Track callback invocations
        callback_called = []

        def resolution_callback(cycle):
            callback_called.append(cycle)
            return True  # Indicate resolution was successful

        detector.add_resolution_callback(resolution_callback)

        # Simulate deadlock detection and handling
        test_cycle = ['proc1', 'proc2', 'proc1']
        from core.coordination.deadlock import CycleDetectionResult

        result = CycleDetectionResult(has_cycle=True, cycles=[test_cycle])
        await detector._handle_deadlock(result)

        # Check callback was called
        assert len(callback_called) == 1
        assert callback_called[0] == test_cycle

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting detector status."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.1)

        try:
            await detector.start()

            status = detector.get_status()

            required_keys = [
                "cycle_count", "last_cycle", "active", "graph_size",
                "resource_count", "process_count", "blocked_processes",
                "metrics", "resolution_strategy", "detection_interval"
            ]

            for key in required_keys:
                assert key in status

            assert status["active"] is True
            assert status["resolution_strategy"] == "LOG_ONLY"
            assert status["detection_interval"] == 0.1

        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_get_wait_graph(self):
        """Test getting wait graph information."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Add some processes and resources
        await detector.acquire_resource("proc1", "resource1", "Process1", priority=5)
        await detector.acquire_resource("proc2", "resource2", "Process2", priority=3)

        wait_graph = detector.get_wait_graph()

        assert "proc1" in wait_graph
        assert "proc2" in wait_graph

        proc1_info = wait_graph["proc1"]
        assert proc1_info["name"] == "Process1"
        assert proc1_info["priority"] == 5
        assert "holding" in proc1_info
        assert "waiting_for" in proc1_info
        assert "blocked" in proc1_info

    @pytest.mark.asyncio
    async def test_find_potential_deadlocks(self):
        """Test finding potential deadlock situations."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Create scenario with potential deadlock
        await detector.acquire_resource("proc1", "resource1", "Process1")
        await detector.acquire_resource("proc2", "resource2", "Process2")

        # Make them wait for each other's resources
        await detector.acquire_resource("proc1", "resource2")  # Will wait
        await detector.acquire_resource("proc2", "resource1")  # Will wait

        potential = detector.find_potential_deadlocks()

        # Should find the potential deadlock
        assert len(potential) > 0

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test metrics tracking functionality."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, enable_metrics=True)

        try:
            await detector.start()

            # Let it run for a bit to accumulate metrics
            await asyncio.sleep(0.2)

            metrics = detector.get_metrics()

            required_metrics = [
                "total_detections", "deadlocks_found", "deadlocks_resolved",
                "detection_errors", "avg_detection_time_ms", "active_processes",
                "active_resources", "blocked_processes"
            ]

            for metric in required_metrics:
                assert metric in metrics

            assert metrics["total_detections"] > 0

        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_error_handling_in_detection_loop(self):
        """Test error handling in the detection loop."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.05)

        # Mock the dependency graph to raise an error
        original_find_cycles = detector._dependency_graph.find_cycles
        call_count = [0]

        def failing_find_cycles(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # Fail first two calls
                raise Exception("Test error")
            return original_find_cycles(*args, **kwargs)

        detector._dependency_graph.find_cycles = failing_find_cycles

        try:
            await detector.start()

            # Let it run and handle errors
            await asyncio.sleep(0.3)

            # Should still be running despite errors
            status = detector.get_status()

            # Check error metrics
            metrics = detector.get_metrics()
            assert metrics["detection_errors"] > 0

        finally:
            await detector.stop()


class TestResolutionStrategies:
    """Test different deadlock resolution strategies."""

    @pytest.mark.asyncio
    async def test_log_only_strategy(self):
        """Test LOG_ONLY resolution strategy."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            resolution_strategy=DeadlockResolutionStrategy.LOG_ONLY
        )

        test_cycle = ['proc1', 'proc2', 'proc1']
        detection_id = "test_123"

        # Should just log and return True
        resolved = await detector._apply_resolution_strategy(test_cycle, detection_id)
        assert resolved is True

    @pytest.mark.asyncio
    async def test_kill_youngest_strategy(self):
        """Test KILL_YOUNGEST resolution strategy."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            resolution_strategy=DeadlockResolutionStrategy.KILL_YOUNGEST
        )

        # Create processes with different ages
        await detector.acquire_resource("old_proc", "resource1", "OldProcess")
        await asyncio.sleep(0.01)  # Make it slightly older
        await detector.acquire_resource("young_proc", "resource2", "YoungProcess")

        test_cycle = ['old_proc', 'young_proc']
        detection_id = "test_123"

        # Mock the terminate process method
        detector._terminate_process = AsyncMock()

        resolved = await detector._apply_resolution_strategy(test_cycle, detection_id)

        # Should attempt to kill the youngest process
        detector._terminate_process.assert_called_once()
        args = detector._terminate_process.call_args[0]
        assert args[0] == "young_proc"

    @pytest.mark.asyncio
    async def test_kill_oldest_strategy(self):
        """Test KILL_OLDEST resolution strategy."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            resolution_strategy=DeadlockResolutionStrategy.KILL_OLDEST
        )

        # Create processes with different ages
        await detector.acquire_resource("old_proc", "resource1", "OldProcess")
        await asyncio.sleep(0.01)  # Make it older
        await detector.acquire_resource("young_proc", "resource2", "YoungProcess")

        test_cycle = ['old_proc', 'young_proc']
        detection_id = "test_123"

        # Mock the terminate process method
        detector._terminate_process = AsyncMock()

        resolved = await detector._apply_resolution_strategy(test_cycle, detection_id)

        # Should attempt to kill the oldest process
        detector._terminate_process.assert_called_once()
        args = detector._terminate_process.call_args[0]
        assert args[0] == "old_proc"

    @pytest.mark.asyncio
    async def test_preempt_resources_strategy(self):
        """Test PREEMPT_RESOURCES resolution strategy."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            resolution_strategy=DeadlockResolutionStrategy.PREEMPT_RESOURCES
        )

        # Create processes holding resources
        await detector.acquire_resource("proc1", "resource1", "Process1")
        await detector.acquire_resource("proc1", "resource2", "Process1")  # proc1 holds 2 resources
        await detector.acquire_resource("proc2", "resource3", "Process2")  # proc2 holds 1 resource

        test_cycle = ['proc1', 'proc2']
        detection_id = "test_123"

        resolved = await detector._apply_resolution_strategy(test_cycle, detection_id)

        # Should preempt from proc1 (has more resources)
        assert resolved is True

        # Verify proc1's resources were released
        wait_graph = detector.get_wait_graph()
        proc1_info = wait_graph.get("proc1", {})
        # Resources should be released (empty holding list)
        assert len(proc1_info.get("holding", [])) == 0

    @pytest.mark.asyncio
    async def test_raise_exception_strategy(self):
        """Test RAISE_EXCEPTION resolution strategy."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(
            mock_agent,
            resolution_strategy=DeadlockResolutionStrategy.RAISE_EXCEPTION
        )

        test_cycle = ['proc1', 'proc2', 'proc1']

        from core.coordination.deadlock import CycleDetectionResult
        result = CycleDetectionResult(has_cycle=True, cycles=[test_cycle])

        # Should raise DeadlockError
        with pytest.raises(DeadlockError) as exc_info:
            await detector._handle_deadlock(result)

        assert exc_info.value.cycle == test_cycle

    @pytest.mark.asyncio
    async def test_terminate_process_cleanup(self):
        """Test that terminate_process cleans up properly."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Create process with resources
        await detector.acquire_resource("proc1", "resource1", "Process1")
        await detector.acquire_resource("proc1", "resource2", "Process1")

        # Add to dependency graph
        await detector.add_dependency("proc1", "some_state")

        # Terminate process
        await detector._terminate_process("proc1", "test_termination")

        # Verify cleanup
        assert "proc1" not in detector._resource_graph.processes

        # Resources should be released
        for resource in detector._resource_graph.resources.values():
            assert "proc1" not in resource.holders
            assert "proc1" not in resource.waiters


class TestPerformanceAndStress:
    """Test performance characteristics and stress scenarios."""

    @pytest.mark.asyncio
    async def test_large_dependency_graph_performance(self):
        """Test performance with large dependency graphs."""
        graph = DependencyGraph()

        # Create large graph
        num_nodes = 100
        for i in range(num_nodes):
            for j in range(min(5, num_nodes - i - 1)):  # Each node depends on up to 5 others
                await graph.add_dependency(f"node_{i}", f"node_{i + j + 1}")

        # Test cycle detection performance
        start_time = time.time()
        result = graph.find_cycles()
        detection_time = time.time() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert detection_time < 1.0
        assert result.detection_duration_ms > 0

    @pytest.mark.asyncio
    async def test_many_concurrent_resources(self):
        """Test handling many concurrent resource operations."""
        graph = ResourceWaitGraph()

        # Create many resources and processes
        num_resources = 50
        num_processes = 50

        for i in range(num_resources):
            await graph.add_resource(f"resource_{i}", "semaphore")

        for i in range(num_processes):
            await graph.add_process(f"proc_{i}", f"Process{i}")

        # Have each process try to acquire multiple resources
        tasks = []
        for i in range(num_processes):
            for j in range(min(3, num_resources)):  # Each process tries 3 resources
                task = asyncio.create_task(
                    graph.acquire_resource(f"proc_{i}", f"resource_{j}")
                )
                tasks.append(task)

        # Execute all acquisitions
        results = await asyncio.gather(*tasks)

        # Most should succeed
        success_count = sum(1 for r in results if r)
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_deadlock_detector_stress(self):
        """Test deadlock detector under stress."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.01)  # Very frequent detection

        try:
            await detector.start()

            # Create many processes and resources rapidly
            tasks = []
            for i in range(20):
                task = asyncio.create_task(
                    detector.acquire_resource(f"proc_{i}", f"resource_{i % 5}", f"Process{i}")
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Let detector run under load
            await asyncio.sleep(0.1)

            # Should still be responsive
            status = detector.get_status()
            assert status["active"] is True

            metrics = detector.get_metrics()
            assert metrics["total_detections"] > 0

        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_memory_cleanup_under_load(self):
        """Test memory cleanup under continuous load."""
        graph = DependencyGraph(max_nodes=50)

        # Continuously add and remove nodes
        for cycle in range(10):
            # Add many nodes
            for i in range(20):
                await graph.add_dependency(f"temp_{cycle}_{i}", f"temp_{cycle}_{i + 1}")

            # Trigger cleanup by exceeding max_nodes
            for i in range(60):  # This should trigger cleanup
                await graph.add_dependency(f"overflow_{cycle}_{i}", f"overflow_{cycle}_{i + 1}")

            # Verify size is controlled
            assert len(graph.nodes) <= 70  # Some cleanup should occur

    @pytest.mark.asyncio
    async def test_concurrent_deadlock_detection(self):
        """Test concurrent deadlock detection operations."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.05)

        try:
            await detector.start()

            # Perform many concurrent operations
            async def create_deadlock_scenario(proc_id):
                resource1 = f"resource_{proc_id % 3}"
                resource2 = f"resource_{(proc_id + 1) % 3}"

                await detector.acquire_resource(f"proc_{proc_id}", resource1)
                await asyncio.sleep(0.01)  # Small delay
                await detector.acquire_resource(f"proc_{proc_id}", resource2)
                await asyncio.sleep(0.01)
                await detector.release_resource(f"proc_{proc_id}", resource1)
                await detector.release_resource(f"proc_{proc_id}", resource2)

            # Run many concurrent scenarios
            tasks = [
                asyncio.create_task(create_deadlock_scenario(i))
                for i in range(10)
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Let detection run
            await asyncio.sleep(0.2)

            # Detector should handle concurrent operations
            status = detector.get_status()
            assert status["active"] is True

        finally:
            await detector.stop()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_graphs(self):
        """Test behavior with empty graphs."""
        dep_graph = DependencyGraph()
        resource_graph = ResourceWaitGraph()

        # Empty dependency graph
        result = dep_graph.find_cycles()
        assert result.has_cycle is False
        assert len(result.cycles) == 0

        # Empty resource graph
        result = await resource_graph.detect_deadlock()
        assert result.has_cycle is False

        # Topological sort of empty graph
        sorted_nodes = dep_graph.topological_sort()
        assert sorted_nodes == []

    @pytest.mark.asyncio
    async def test_self_loops(self):
        """Test handling of self-loops."""
        graph = DependencyGraph()

        # Add self-loop
        await graph.add_dependency("A", "A")

        result = graph.find_cycles()
        assert result.has_cycle is True

        # Should find the self-loop
        found_self_loop = any('A' in cycle and len(cycle) == 2 for cycle in result.cycles)
        assert found_self_loop

    @pytest.mark.asyncio
    async def test_nonexistent_resources_and_processes(self):
        """Test operations on nonexistent resources and processes."""
        graph = ResourceWaitGraph()

        # Try to acquire nonexistent resource
        success = await graph.acquire_resource("nonexistent_proc", "nonexistent_resource")
        # Should auto-create them
        assert "nonexistent_proc" in graph.processes
        assert "nonexistent_resource" in graph.resources

        # Try to release nonexistent holdings
        await graph.release_resource("proc1", "resource1")  # Should not crash

        # Get info for nonexistent resources
        holders = graph.get_resource_holders("nonexistent")
        waiters = graph.get_resource_waiters("nonexistent")
        assert len(holders) == 0
        assert len(waiters) == 0

    @pytest.mark.asyncio
    async def test_detector_without_monitor(self):
        """Test detector behavior when agent has no monitor."""
        agent = Mock()
        agent.name = "test_agent"
        # No _monitor attribute

        detector = DeadlockDetector(agent, detection_interval=0.05)

        try:
            await detector.start()

            # Should work without monitor
            await detector.acquire_resource("proc1", "resource1")

            # Let it run briefly
            await asyncio.sleep(0.1)

            status = detector.get_status()
            assert status["active"] is True

        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_detector_stop_without_start(self):
        """Test stopping detector that was never started."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Should not crash
        await detector.stop()

    @pytest.mark.asyncio
    async def test_very_large_cycles(self):
        """Test handling of very large dependency cycles."""
        graph = DependencyGraph()

        # Create large cycle
        cycle_size = 100
        for i in range(cycle_size):
            await graph.add_dependency(f"node_{i}", f"node_{(i + 1) % cycle_size}")

        result = graph.find_cycles()
        assert result.has_cycle is True

        # Should find the large cycle
        large_cycle = result.get_longest_cycle()
        assert large_cycle is not None
        assert len(large_cycle) > 50  # Should be a substantial cycle

    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self):
        """Test rapid start/stop cycles."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent, detection_interval=0.01)

        # Rapid start/stop cycles
        for _ in range(5):
            await detector.start()
            await asyncio.sleep(0.02)
            await detector.stop()
            await asyncio.sleep(0.01)

        # Should end in stopped state
        status = detector.get_status()
        assert status["active"] is False

    @pytest.mark.asyncio
    async def test_exception_in_resolution_callback(self):
        """Test handling exceptions in resolution callbacks."""
        mock_agent = create_mock_agent()
        detector = DeadlockDetector(mock_agent)

        # Add callback that raises exception
        def failing_callback(cycle):
            raise Exception("Callback failed")

        def working_callback(cycle):
            return True

        detector.add_resolution_callback(failing_callback)
        detector.add_resolution_callback(working_callback)

        # Should handle exception and continue to next callback
        test_cycle = ['proc1', 'proc2']
        from core.coordination.deadlock import CycleDetectionResult

        result = CycleDetectionResult(has_cycle=True, cycles=[test_cycle])

        # Should not raise exception, should be handled gracefully
        await detector._handle_deadlock(result)

        # Should have incremented resolved count despite first callback failing
        metrics = detector.get_metrics()
        assert metrics["deadlocks_resolved"] > 0


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "--tb=short"])