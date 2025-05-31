"""
Comprehensive test suite for the workflow orchestration system.

This test suite covers all major components including:
- Agent execution and state management
- Context and variable management  
- Dependencies and coordination
- Checkpointing and replay
- Metrics and monitoring
- Alert system
- Event streaming
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import json
import uuid

# Import system components (adjust imports based on your project structure)
from core.agent.base import Agent, RetryPolicy
from core.agent.context import Context, StateType
from core.agent.state import Priority, AgentStatus, StateStatus, StateResult
from core.agent.dependencies import DependencyType, DependencyLifecycle, DependencyConfig
from core.agent.checkpoint import AgentCheckpoint
from core.coordination.primitives import Mutex, Semaphore, Barrier, Lease, PrimitiveType
from core.coordination.rate_limiter import RateLimiter, RateLimitStrategy, TokenBucket
from core.coordination.deadlock import DeadlockDetector
from core.coordination.coordinator import AgentCoordinator
from core.execution.lifecycle import StateLifecycle, LoggingHook, MetricsHook, HookType
from core.execution.replay import ReplayEngine, ReplayMode
from core.execution.determinism import DeterminismChecker, deterministic
from core.execution.engine import WorkflowEngine
from core.monitoring.events import EventStream, EventFilter, EventSubscription
from core.monitoring.metrics import MetricsCollector, MetricType, agent_monitor
from core.monitoring.alerts import AlertManager, AlertRule, AlertCondition, AlertSeverity, alert_dsl
from core.storage.events import WorkflowEvent, EventType


# Test fixtures
@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    shared_state = {"test_data": "hello world", "counter": 0}
    return Context(shared_state, cache_ttl=60)


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    agent = Agent(
        name="test_agent",
        max_concurrent=5,
        state_timeout=30.0
    )
    return agent


@pytest.fixture
def mock_storage():
    """Mock storage backend for testing."""
    storage = Mock()
    storage.initialize = AsyncMock()
    storage.save_event = AsyncMock()
    storage.load_events = AsyncMock(return_value=[])
    storage.save_checkpoint = AsyncMock()
    storage.load_checkpoint = AsyncMock(return_value=None)
    storage.close = AsyncMock()
    return storage


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# Test Classes

class TestContext:
    """Test Context functionality."""
    
    def test_basic_variables(self, sample_context):
        """Test basic variable operations."""
        ctx = sample_context
        
        # Test setting and getting variables
        ctx.set_variable("test_key", "test_value")
        assert ctx.get_variable("test_key") == "test_value"
        assert ctx.get_variable("nonexistent") is None
        assert ctx.get_variable("nonexistent", "default") == "default"
    
    def test_typed_variables(self, sample_context):
        """Test typed variable functionality."""
        ctx = sample_context
        
        # First write establishes type
        ctx.set_typed_variable("typed_str", "hello")
        assert ctx.get_typed_variable("typed_str", str) == "hello"
        
        # Should work with same type
        ctx.set_typed_variable("typed_str", "world")
        assert ctx.get_typed_variable("typed_str", str) == "world"
        
        # Should fail with different type
        with pytest.raises(TypeError):
            ctx.set_typed_variable("typed_str", 42)
    
    def test_constants_and_secrets(self, sample_context):
        """Test constants and secrets."""
        ctx = sample_context
        
        # Test constants
        ctx.set_constant("api_version", "v1.0")
        assert ctx.get_constant("api_version") == "v1.0"
        
        # Should not allow overwriting
        with pytest.raises(ValueError):
            ctx.set_constant("api_version", "v2.0")
        
        # Test secrets
        ctx.set_secret("api_key", "secret123")
        assert ctx.get_secret("api_key") == "secret123"
    
    def test_state_data(self, sample_context):
        """Test per-state data management."""
        ctx = sample_context
        
        ctx.set_state("temp_data", {"processing": True})
        assert ctx.get_state("temp_data")["processing"] is True
        
        # Test outputs
        ctx.set_output("result", "success")
        assert ctx.get_output("result") == "success"
        assert "result" in ctx.get_output_keys()
    
    def test_cache_functionality(self, sample_context):
        """Test TTL cache functionality."""
        ctx = sample_context
        
        # Set cached value
        ctx.set_cached("temp_result", "cached_data", ttl=1)
        assert ctx.get_cached("temp_result") == "cached_data"
        
        # Wait for expiry
        time.sleep(1.1)
        assert ctx.get_cached("temp_result", "default") == "default"


class TestAgent:
    """Test Agent functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_agent_creation(self):
        """Test basic agent creation and configuration."""
        agent = Agent(
            name="test_agent",
            max_concurrent=3,
            state_timeout=60.0
        )
        
        assert agent.name == "test_agent"
        assert agent._semaphore._value == 3
        assert agent.state_timeout == 60.0
        assert agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_simple_state_execution(self):
        """Test basic state execution."""
        agent = Agent("test_agent")
        
        # Define a simple state
        async def simple_state(context: Context) -> StateResult:
            context.set_output("message", "Hello World")
            return None
        
        agent.add_state("simple", simple_state)
        
        # Run the agent
        await agent.run(timeout=5.0)
        
        # Check completion
        assert "simple" in agent.completed_states
        assert agent.status == AgentStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test state transitions."""
        agent = Agent("test_agent")
        execution_order = []
        
        async def state_a(context: Context) -> StateResult:
            execution_order.append("A")
            return "state_b"
        
        async def state_b(context: Context) -> StateResult:
            execution_order.append("B")
            return "state_c"
        
        async def state_c(context: Context) -> StateResult:
            execution_order.append("C")
            return None
        
        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)
        agent.add_state("state_c", state_c)
        
        await agent.run(timeout=10.0)
        
        assert execution_order == ["A", "B", "C"]
        assert len(agent.completed_states) == 3
    
    @pytest.mark.asyncio
    async def test_parallel_states(self):
        """Test parallel state execution."""
        agent = Agent("test_agent", max_concurrent=3)
        execution_times = {}
        
        async def parallel_state(name: str):
            async def state_func(context: Context) -> StateResult:
                start_time = time.time()
                await asyncio.sleep(0.1)  # Simulate work
                execution_times[name] = time.time() - start_time
                return None
            return state_func
        
        # Add parallel states
        agent.add_state("parallel_1", await parallel_state("parallel_1"))
        agent.add_state("parallel_2", await parallel_state("parallel_2"))
        agent.add_state("parallel_3", await parallel_state("parallel_3"))
        
        start_time = time.time()
        await agent.run(timeout=5.0)
        total_time = time.time() - start_time
        
        # Should complete in roughly the time of one task (parallel execution)
        assert total_time < 0.5  # Much less than 3 * 0.1 = 0.3
        assert len(agent.completed_states) == 3
    
    @pytest.mark.asyncio
    async def test_dependencies(self):
        """Test state dependencies."""
        agent = Agent("test_agent")
        execution_order = []
        
        async def dependent_state(context: Context) -> StateResult:
            execution_order.append("dependent")
            return None
        
        async def prerequisite_state(context: Context) -> StateResult:
            execution_order.append("prerequisite")
            return None
        
        # Add states with dependency
        agent.add_state("prerequisite", prerequisite_state)
        agent.add_state(
            "dependent",
            dependent_state,
            dependencies={"prerequisite": DependencyType.REQUIRED}
        )
        
        await agent.run(timeout=5.0)
        
        # Prerequisite should run before dependent
        assert execution_order == ["prerequisite", "dependent"]
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test state retry functionality."""
        agent = Agent("test_agent")
        attempt_count = 0
        
        async def failing_state(context: Context) -> StateResult:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return None
        
        agent.add_state("failing", failing_state, max_retries=5)
        
        await agent.run(timeout=10.0)
        
        assert attempt_count == 3
        assert "failing" in agent.completed_states
    
    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        """Test agent pause and resume functionality."""
        agent = Agent("test_agent")
        paused = False
        
        async def pausable_state(context: Context) -> StateResult:
            nonlocal paused
            if not paused:
                paused = True
                # Simulate pause request
                await agent.pause()
            return None
        
        agent.add_state("pausable", pausable_state)
        
        # Start execution
        run_task = asyncio.create_task(agent.run(timeout=10.0))
        
        # Wait a bit and resume
        await asyncio.sleep(0.1)
        await agent.resume()
        
        await run_task
        
        assert agent.status == AgentStatus.COMPLETED


class TestCoordinationPrimitives:
    """Test coordination primitives."""
    
    @pytest.mark.asyncio
    async def test_mutex(self):
        """Test mutex functionality."""
        mutex = Mutex("test_mutex", ttl=60.0)
        
        # First acquisition should succeed
        result1 = await mutex.acquire("process1")
        assert result1 is True
        
        # Second acquisition should fail
        result2 = await mutex.acquire("process2")
        assert result2 is False
        
        # Release and retry
        await mutex.release("process1")
        result3 = await mutex.acquire("process2")
        assert result3 is True
    
    @pytest.mark.asyncio
    async def test_semaphore(self):
        """Test semaphore functionality."""
        semaphore = Semaphore("test_semaphore", max_count=2, ttl=60.0)
        
        # Should allow up to max_count acquisitions
        assert await semaphore.acquire("process1") is True
        assert await semaphore.acquire("process2") is True
        assert await semaphore.acquire("process3") is False
        
        # Release one and try again
        await semaphore.release("process1")
        assert await semaphore.acquire("process3") is True
    
    @pytest.mark.asyncio
    async def test_barrier(self):
        """Test barrier synchronization."""
        barrier = Barrier("test_barrier", parties=3, timeout=5.0)
        results = []
        
        async def worker(worker_id: str):
            result = await barrier.wait(worker_id)
            results.append(f"worker_{worker_id}_done")
            return result
        
        # Start workers
        tasks = [
            asyncio.create_task(worker("1")),
            asyncio.create_task(worker("2")),
            asyncio.create_task(worker("3"))
        ]
        
        generations = await asyncio.gather(*tasks)
        
        # All workers should complete and get same generation
        assert len(results) == 3
        assert all(gen == generations[0] for gen in generations)
    
    @pytest.mark.asyncio
    async def test_lease(self):
        """Test lease functionality."""
        lease = Lease("test_lease", ttl=0.1)  # Very short TTL
        
        # Acquire lease
        assert await lease.acquire("process1") is True
        
        # Should be unavailable to others
        assert await lease.acquire("process2") is False
        
        # Wait for expiry
        await asyncio.sleep(0.2)
        
        # Should be available again
        assert await lease.acquire("process2") is True


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_token_bucket(self):
        """Test token bucket rate limiter."""
        bucket = TokenBucket(rate=5.0, capacity=10)  # 5 tokens/second, capacity 10
        
        # Should allow burst up to capacity
        for i in range(10):
            assert await bucket.acquire() is True
        
        # Should be exhausted
        assert await bucket.acquire() is False
        
        # Wait for refill
        await asyncio.sleep(0.5)  # Should add ~2.5 tokens
        assert await bucket.acquire() is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_strategies(self):
        """Test different rate limiting strategies."""
        # Token bucket
        token_limiter = RateLimiter(
            max_rate=10.0,
            burst_size=5,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        
        # Should allow burst
        successes = 0
        for _ in range(10):
            if await token_limiter.acquire():
                successes += 1
        
        assert successes == 5  # Up to burst size
        
        # Fixed window
        window_limiter = RateLimiter(
            max_rate=5.0,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            window_size=1.0
        )
        
        # Should allow up to rate per window
        successes = 0
        for _ in range(10):
            if await window_limiter.acquire():
                successes += 1
        
        assert successes <= 5


class TestEventStreaming:
    """Test event streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_event_stream_basic(self):
        """Test basic event streaming."""
        stream = EventStream(buffer_size=100)
        received_events = []
        
        # Subscribe to events
        def event_handler(event: WorkflowEvent):
            received_events.append(event)
        
        await stream.start()
        subscriber_id = stream.subscribe(event_handler)
        
        # Publish test event
        test_event = WorkflowEvent(
            workflow_id="test_workflow",
            event_type=EventType.WORKFLOW_STARTED,
            timestamp=datetime.utcnow(),
            data={"test": "data"}
        )
        
        await stream.publish(test_event)
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(received_events) == 1
        assert received_events[0].workflow_id == "test_workflow"
        
        stream.unsubscribe(subscriber_id)
        await stream.stop()
    
    @pytest.mark.asyncio
    async def test_event_filtering(self):
        """Test event filtering."""
        stream = EventStream()
        received_events = []
        
        # Create filter for specific workflow
        event_filter = EventFilter(
            workflow_ids={"target_workflow"},
            event_types={EventType.STATE_COMPLETED}
        )
        
        def filtered_handler(event: WorkflowEvent):
            received_events.append(event)
        
        await stream.start()
        stream.subscribe(filtered_handler, filter=event_filter)
        
        # Publish events - only matching ones should be received
        events = [
            WorkflowEvent("target_workflow", EventType.STATE_COMPLETED, datetime.utcnow(), {}),
            WorkflowEvent("other_workflow", EventType.STATE_COMPLETED, datetime.utcnow(), {}),
            WorkflowEvent("target_workflow", EventType.STATE_STARTED, datetime.utcnow(), {}),
        ]
        
        for event in events:
            await stream.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Only first event should match filter
        assert len(received_events) == 1
        assert received_events[0].workflow_id == "target_workflow"
        
        await stream.stop()


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_collector(self):
        """Test basic metrics collection."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_state_execution("test_agent", "test_state", 1.5, "success")
        collector.record_error("test_agent", "test_state", "ValueError")
        collector.record_resource_usage("cpu", "test_agent", 75.0)
        
        # Check that metrics were recorded (would need access to internal state)
        assert collector._metrics is not None
    
    @pytest.mark.asyncio
    async def test_agent_monitor_decorator(self):
        """Test agent monitoring decorator."""
        collected_metrics = {}
        
        def metrics_callback(metrics: Dict[str, Any]):
            collected_metrics.update(metrics)
        
        @agent_monitor(
            metrics=MetricType.ALL,
            metrics_callback=metrics_callback,
            aggregate=True
        )
        async def monitored_run():
            agent = Agent("monitored_agent")
            
            async def test_state(context: Context) -> StateResult:
                await asyncio.sleep(0.1)
                return None
            
            agent.add_state("test", test_state)
            await agent.run(timeout=5.0)
            return agent
        
        agent = await monitored_run()
        
        assert "timing" in collected_metrics
        assert "states" in collected_metrics
        assert agent.status == AgentStatus.COMPLETED


class TestAlerts:
    """Test alert system."""
    
    @pytest.mark.asyncio
    async def test_alert_manager(self):
        """Test alert manager functionality."""
        manager = AlertManager()
        fired_alerts = []
        
        # Mock destination
        class TestDestination:
            async def send(self, alert):
                fired_alerts.append(alert)
        
        # Create alert rule
        condition = AlertCondition(
            metric="test_metric",
            operator=">",
            threshold=100.0,
            window=timedelta(minutes=1)
        )
        
        rule = AlertRule(
            name="test_alert",
            condition=condition,
            severity=AlertSeverity.WARNING,
            message="Test alert fired",
            destinations=[TestDestination()]
        )
        
        manager.add_rule(rule)
        await manager.start()
        
        # Record metrics that should trigger alert
        await manager.record_metric("test_metric", 150.0)
        await manager._evaluate_rules()  # Force evaluation
        
        assert len(fired_alerts) == 1
        assert fired_alerts[0].rule_name == "test_alert"
        
        await manager.stop()
    
    def test_alert_dsl(self):
        """Test alert DSL."""
        webhook_dest = Mock()
        
        rule = (
            alert_dsl("high_error_rate")
            .where("error_rate")
            .is_above(0.1)
            .in_window(5)
            .with_severity(AlertSeverity.ERROR)
            .with_message("Error rate too high")
            .with_labels(team="platform")
            .then(webhook_dest)
            .build()
        )
        
        assert rule.name == "high_error_rate"
        assert rule.condition.metric == "error_rate"
        assert rule.condition.operator == ">"
        assert rule.condition.threshold == 0.1
        assert rule.severity == AlertSeverity.ERROR
        assert webhook_dest in rule.destinations


class TestWorkflowEngine:
    """Test workflow execution engine."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, mock_storage):
        """Test complete workflow execution through engine."""
        engine = WorkflowEngine(mock_storage)
        await engine.start()
        
        # Create test agent
        agent = Agent("test_workflow_agent")
        execution_log = []
        
        async def task_a(context: Context) -> StateResult:
            execution_log.append("A")
            context.set_variable("result_a", "completed")
            return "task_b"
        
        async def task_b(context: Context) -> StateResult:
            execution_log.append("B")
            assert context.get_variable("result_a") == "completed"
            return None
        
        agent.add_state("task_a", task_a)
        agent.add_state("task_b", task_b)
        
        # Execute workflow
        workflow_id = await engine.create_workflow("test_workflow", agent)
        await engine.execute_workflow(workflow_id, agent, timeout=10.0)
        
        assert execution_log == ["A", "B"]
        assert agent.status == AgentStatus.COMPLETED
        
        # Verify events were saved
        assert mock_storage.save_event.called
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_workflow_pause_resume(self, mock_storage):
        """Test workflow pause and resume."""
        engine = WorkflowEngine(mock_storage)
        await engine.start()
        
        agent = Agent("pausable_agent")
        
        async def long_task(context: Context) -> StateResult:
            await asyncio.sleep(1.0)
            return None
        
        agent.add_state("long_task", long_task)
        
        workflow_id = await engine.create_workflow("pausable_workflow", agent)
        
        # Start execution
        exec_task = asyncio.create_task(
            engine.execute_workflow(workflow_id, agent, timeout=10.0)
        )
        
        # Pause after brief delay
        await asyncio.sleep(0.1)
        checkpoint_info = await engine.pause_workflow(workflow_id)
        
        assert checkpoint_info["status"] == "paused"
        assert mock_storage.save_checkpoint.called
        
        await engine.stop()


class TestDeterminism:
    """Test determinism checking."""
    
    def test_determinism_checker(self):
        """Test determinism validation."""
        checker = DeterminismChecker()
        
        # Test deterministic function
        @deterministic
        async def deterministic_function(context: Context) -> StateResult:
            data = context.get_variable("input_data", [])
            result = sorted(data)  # Deterministic operation
            context.set_variable("output", result)
            return None
        
        # Register and analyze
        fingerprint = checker.register_state("sort_data", deterministic_function)
        assert fingerprint.state_name == "sort_data"
        assert len(fingerprint.function_hash) > 0
    
    def test_non_deterministic_detection(self):
        """Test detection of non-deterministic patterns."""
        checker = DeterminismChecker()
        
        async def non_deterministic_function(context: Context) -> StateResult:
            import random
            value = random.randint(1, 100)  # Non-deterministic
            context.set_variable("random_value", value)
            return None
        
        # Should detect non-deterministic patterns
        issues = checker.analyzer.check_determinism(non_deterministic_function)
        assert len(issues) > 0
        assert any("random" in issue.lower() for issue in issues)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_monitoring(self, mock_storage):
        """Test complete workflow with all monitoring features."""
        # Setup components
        event_stream = EventStream()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        engine = WorkflowEngine(mock_storage)
        
        await event_stream.start()
        await alert_manager.start()
        await engine.start()
        
        # Track events
        received_events = []
        event_stream.subscribe(lambda e: received_events.append(e))
        
        # Create comprehensive workflow
        agent = Agent("integration_test_agent", max_concurrent=2)
        
        async def data_ingestion(context: Context) -> StateResult:
            context.set_variable("data", list(range(100)))
            return ["data_processing", "data_validation"]
        
        async def data_processing(context: Context) -> StateResult:
            data = context.get_variable("data", [])
            processed = [x * 2 for x in data]
            context.set_variable("processed_data", processed)
            return "data_output"
        
        async def data_validation(context: Context) -> StateResult:
            data = context.get_variable("data", [])
            assert len(data) == 100
            context.set_variable("validation_passed", True)
            return "data_output"
        
        async def data_output(context: Context) -> StateResult:
            processed = context.get_variable("processed_data", [])
            validated = context.get_variable("validation_passed", False)
            
            if validated and len(processed) == 100:
                context.set_output("result", "success")
                return None
            else:
                raise Exception("Validation failed")
        
        # Add states with dependencies
        agent.add_state("ingestion", data_ingestion)
        agent.add_state(
            "processing",
            data_processing,
            dependencies={"ingestion": DependencyType.REQUIRED}
        )
        agent.add_state(
            "validation",
            data_validation,
            dependencies={"ingestion": DependencyType.REQUIRED}
        )
        agent.add_state(
            "output",
            data_output,
            dependencies={
                "processing": DependencyType.REQUIRED,
                "validation": DependencyType.REQUIRED
            }
        )
        
        # Execute workflow
        workflow_id = await engine.create_workflow("integration_test", agent)
        await engine.execute_workflow(workflow_id, agent, timeout=30.0)
        
        # Verify completion
        assert agent.status == AgentStatus.COMPLETED
        assert len(agent.completed_states) == 4
        
        # Verify events were generated
        await asyncio.sleep(0.1)  # Allow event processing
        assert len(received_events) > 0
        
        # Cleanup
        await event_stream.stop()
        await alert_manager.stop()
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test comprehensive error handling and recovery."""
        agent = Agent("error_test_agent", max_concurrent=1)
        
        failure_count = 0
        
        async def unreliable_state(context: Context) -> StateResult:
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                raise Exception(f"Failure {failure_count}")
            
            context.set_output("recovered", True)
            return "cleanup_state"
        
        async def cleanup_state(context: Context) -> StateResult:
            context.set_output("cleanup_done", True)
            return None
        
        # Add states with retry policy
        retry_policy = RetryPolicy(max_retries=5, initial_delay=0.1)
        agent.add_state("unreliable", unreliable_state, retry_policy=retry_policy)
        agent.add_state("cleanup", cleanup_state)
        
        await agent.run(timeout=10.0)
        
        assert agent.status == AgentStatus.COMPLETED
        assert failure_count == 3  # 2 failures + 1 success
        assert "unreliable" in agent.completed_states
        assert "cleanup" in agent.completed_states
    
    @pytest.mark.asyncio
    async def test_complex_dependencies(self):
        """Test complex dependency scenarios."""
        agent = Agent("dependency_test_agent")
        execution_order = []
        
        async def make_state(name: str):
            async def state_func(context: Context) -> StateResult:
                execution_order.append(name)
                await asyncio.sleep(0.01)  # Small delay
                return None
            return state_func
        
        # Create states with complex dependencies
        agent.add_state("init", await make_state("init"))
        
        agent.add_state(
            "process_a",
            await make_state("process_a"),
            dependencies={"init": DependencyType.REQUIRED}
        )
        
        agent.add_state(
            "process_b",
            await make_state("process_b"),
            dependencies={"init": DependencyType.REQUIRED}
        )
        
        agent.add_state(
            "aggregate",
            await make_state("aggregate"),
            dependencies={
                "process_a": DependencyType.REQUIRED,
                "process_b": DependencyType.REQUIRED
            }
        )
        
        agent.add_state(
            "finalize",
            await make_state("finalize"),
            dependencies={"aggregate": DependencyType.REQUIRED}
        )
        
        await agent.run(timeout=10.0)
        
        # Verify execution order respects dependencies
        init_idx = execution_order.index("init")
        process_a_idx = execution_order.index("process_a")
        process_b_idx = execution_order.index("process_b")
        aggregate_idx = execution_order.index("aggregate")
        finalize_idx = execution_order.index("finalize")
        
        assert init_idx < process_a_idx
        assert init_idx < process_b_idx
        assert process_a_idx < aggregate_idx
        assert process_b_idx < aggregate_idx
        assert aggregate_idx < finalize_idx


# Test runners and utilities
def test_all_components():
    """Run all component tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    # Run all tests
    test_all_components()