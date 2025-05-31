"""
Integration and end-to-end tests for the workflow orchestration system.

These tests validate the system as a whole, testing interactions between
components and realistic workflow scenarios.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import uuid

from core.agent.base import Agent
from core.agent.context import Context
from core.agent.state import StateResult, Priority
from core.agent.dependencies import DependencyType, DependencyLifecycle
from core.coordination.coordinator import AgentCoordinator, enhance_agent
from core.coordination.primitives import PrimitiveType
from core.coordination.rate_limiter import RateLimitStrategy
from core.execution.engine import WorkflowEngine
from core.execution.lifecycle import StateLifecycle, LoggingHook, MetricsHook, HookType
from core.execution.replay import ReplayEngine, ReplayMode
from core.execution.determinism import DeterminismChecker
from core.monitoring.events import EventStream, EventFilter
from core.monitoring.metrics import MetricsCollector, agent_monitor, MetricType
from core.monitoring.alerts import AlertManager, AlertRule, AlertCondition, AlertSeverity, alert_dsl
from core.storage.events import WorkflowEvent, EventType


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self):
        """Test a complete data processing pipeline."""
        # Setup monitoring
        event_stream = EventStream()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        
        await event_stream.start()
        await alert_manager.start()
        
        # Track events and metrics
        captured_events = []
        captured_metrics = {}
        
        event_stream.subscribe(lambda e: captured_events.append(e))
        
        def metrics_callback(metrics):
            captured_metrics.update(metrics)
        
        # Create enhanced agent with monitoring
        @agent_monitor(
            metrics=MetricType.ALL,
            metrics_callback=metrics_callback,
            realtime=False
        )
        async def run_data_pipeline():
            agent = Agent("data_pipeline", max_concurrent=3)
            
            # Data ingestion stage
            async def ingest_data(context: Context) -> StateResult:
                # Simulate data ingestion
                raw_data = [{"id": i, "value": i * 2} for i in range(100)]
                context.set_variable("raw_data", raw_data)
                context.set_variable("ingestion_timestamp", datetime.utcnow().isoformat())
                return ["validate_data", "enrich_data"]
            
            # Data validation (parallel with enrichment)
            async def validate_data(context: Context) -> StateResult:
                raw_data = context.get_variable("raw_data", [])
                
                # Validate data quality
                valid_records = [r for r in raw_data if r["value"] >= 0]
                invalid_count = len(raw_data) - len(valid_records)
                
                context.set_variable("valid_data", valid_records)
                context.set_variable("validation_errors", invalid_count)
                
                if invalid_count > 10:  # More than 10% invalid
                    raise Exception(f"Too many invalid records: {invalid_count}")
                
                return "process_data"
            
            # Data enrichment (parallel with validation)
            async def enrich_data(context: Context) -> StateResult:
                raw_data = context.get_variable("raw_data", [])
                
                # Add enrichment fields
                enriched_data = []
                for record in raw_data:
                    enriched = record.copy()
                    enriched["processed_at"] = datetime.utcnow().isoformat()
                    enriched["category"] = "A" if record["value"] < 50 else "B"
                    enriched_data.append(enriched)
                
                context.set_variable("enriched_data", enriched_data)
                return "process_data"
            
            # Data processing (depends on both validation and enrichment)
            async def process_data(context: Context) -> StateResult:
                valid_data = context.get_variable("valid_data", [])
                enriched_data = context.get_variable("enriched_data", [])
                
                # Merge validation and enrichment results
                processed_records = []
                for i, record in enumerate(valid_data):
                    if i < len(enriched_data):
                        merged = {**record, **enriched_data[i]}
                        merged["processing_score"] = record["value"] * 1.5
                        processed_records.append(merged)
                
                context.set_variable("processed_data", processed_records)
                context.set_variable("processing_count", len(processed_records))
                
                return ["store_data", "generate_report"]
            
            # Data storage (parallel with reporting)
            async def store_data(context: Context) -> StateResult:
                processed_data = context.get_variable("processed_data", [])
                
                # Simulate storing to database
                storage_results = {
                    "stored_count": len(processed_data),
                    "storage_location": "s3://data-lake/processed/",
                    "storage_timestamp": datetime.utcnow().isoformat()
                }
                
                context.set_variable("storage_results", storage_results)
                return "finalize_pipeline"
            
            # Report generation (parallel with storage)
            async def generate_report(context: Context) -> StateResult:
                processed_data = context.get_variable("processed_data", [])
                
                # Generate summary report
                total_value = sum(r["processing_score"] for r in processed_data)
                category_counts = {}
                for record in processed_data:
                    cat = record.get("category", "unknown")
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                report = {
                    "total_records": len(processed_data),
                    "total_value": total_value,
                    "average_value": total_value / len(processed_data) if processed_data else 0,
                    "category_distribution": category_counts,
                    "report_timestamp": datetime.utcnow().isoformat()
                }
                
                context.set_variable("pipeline_report", report)
                return "finalize_pipeline"
            
            # Finalization
            async def finalize_pipeline(context: Context) -> StateResult:
                storage_results = context.get_variable("storage_results")
                report = context.get_variable("pipeline_report")
                
                # Combine final results
                final_results = {
                    "pipeline_status": "completed",
                    "storage": storage_results,
                    "report": report,
                    "completion_timestamp": datetime.utcnow().isoformat()
                }
                
                context.set_output("pipeline_results", final_results)
                return None
            
            # Add states with dependencies
            agent.add_state("ingest", ingest_data)
            
            agent.add_state(
                "validate",
                validate_data,
                dependencies={"ingest": DependencyType.REQUIRED}
            )
            
            agent.add_state(
                "enrich",
                enrich_data,
                dependencies={"ingest": DependencyType.REQUIRED}
            )
            
            agent.add_state(
                "process",
                process_data,
                dependencies={
                    "validate": DependencyType.REQUIRED,
                    "enrich": DependencyType.REQUIRED
                }
            )
            
            agent.add_state(
                "store",
                store_data,
                dependencies={"process": DependencyType.REQUIRED}
            )
            
            agent.add_state(
                "report",
                generate_report,
                dependencies={"process": DependencyType.REQUIRED}
            )
            
            agent.add_state(
                "finalize",
                finalize_pipeline,
                dependencies={
                    "store": DependencyType.REQUIRED,
                    "report": DependencyType.REQUIRED
                }
            )
            
            # Execute pipeline
            await agent.run(timeout=30.0)
            return agent
        
        # Run the pipeline
        agent = await run_data_pipeline()
        
        # Verify successful completion
        assert agent.status.value == "completed"
        assert len(agent.completed_states) == 7
        
        # Verify data flow
        pipeline_results = agent.shared_state.get("output_pipeline_results")
        assert pipeline_results is not None
        assert pipeline_results["pipeline_status"] == "completed"
        assert pipeline_results["report"]["total_records"] == 100
        
        # Verify monitoring data was captured
        assert len(captured_events) > 0
        assert len(captured_metrics) > 0
        
        # Cleanup
        await event_stream.stop()
        await alert_manager.stop()
    
    @pytest.mark.asyncio
    async def test_microservices_orchestration(self):
        """Test orchestration of multiple microservices."""
        # Simulate multiple microservice agents
        agents = {}
        results = {}
        
        # User Service
        user_agent = Agent("user_service")
        
        async def get_user_profile(context: Context) -> StateResult:
            user_id = context.get_variable("user_id")
            # Simulate API call
            await asyncio.sleep(0.1)
            profile = {
                "id": user_id,
                "name": f"User {user_id}",
                "email": f"user{user_id}@example.com",
                "preferences": {"theme": "dark", "notifications": True}
            }
            context.set_output("user_profile", profile)
            return None
        
        user_agent.add_state("get_profile", get_user_profile)
        agents["user"] = user_agent
        
        # Order Service
        order_agent = Agent("order_service")
        
        async def get_user_orders(context: Context) -> StateResult:
            user_id = context.get_variable("user_id")
            # Simulate database query
            await asyncio.sleep(0.15)
            orders = [
                {"id": f"order_{i}", "amount": i * 10, "status": "completed"}
                for i in range(5)
            ]
            context.set_output("user_orders", orders)
            return None
        
        order_agent.add_state("get_orders", get_user_orders)
        agents["order"] = order_agent
        
        # Recommendation Service
        recommendation_agent = Agent("recommendation_service")
        
        async def generate_recommendations(context: Context) -> StateResult:
            # Simulate ML inference
            await asyncio.sleep(0.2)
            recommendations = [
                {"product_id": f"product_{i}", "score": 0.9 - i * 0.1}
                for i in range(3)
            ]
            context.set_output("recommendations", recommendations)
            return None
        
        recommendation_agent.add_state("generate", generate_recommendations)
        agents["recommendation"] = recommendation_agent
        
        # Orchestrator Service
        orchestrator = Agent("orchestrator")
        
        async def fetch_user_data(context: Context) -> StateResult:
            user_id = context.get_variable("user_id", "123")
            
            # Run microservices in parallel
            tasks = []
            
            # Set user_id for all services
            for agent in agents.values():
                agent.shared_state["user_id"] = user_id
                tasks.append(agent.run(timeout=10.0))
            
            # Wait for all services
            await asyncio.gather(*tasks)
            
            # Collect results
            user_data = {
                "profile": agents["user"].shared_state.get("output_user_profile"),
                "orders": agents["order"].shared_state.get("output_user_orders"),
                "recommendations": agents["recommendation"].shared_state.get("output_recommendations")
            }
            
            context.set_output("aggregated_data", user_data)
            return None
        
        orchestrator.add_state("fetch_data", fetch_user_data)
        
        # Set initial data
        orchestrator.shared_state["user_id"] = "123"
        
        # Execute orchestration
        await orchestrator.run(timeout=15.0)
        
        # Verify results
        assert orchestrator.status.value == "completed"
        
        aggregated_data = orchestrator.shared_state.get("output_aggregated_data")
        assert aggregated_data is not None
        assert aggregated_data["profile"]["id"] == "123"
        assert len(aggregated_data["orders"]) == 5
        assert len(aggregated_data["recommendations"]) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_with_error_recovery(self):
        """Test workflow with comprehensive error handling and recovery."""
        agent = Agent("error_recovery_agent")
        
        # State that fails initially but recovers
        failure_count = 0
        
        async def unreliable_service_call(context: Context) -> StateResult:
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                # Simulate different types of failures
                if failure_count == 1:
                    raise ConnectionError("Network timeout")
                else:
                    raise Exception("Service temporarily unavailable")
            
            # Success on third try
            context.set_variable("service_result", "success_after_retries")
            return "process_result"
        
        async def process_result(context: Context) -> StateResult:
            result = context.get_variable("service_result")
            processed = f"processed_{result}"
            context.set_output("final_result", processed)
            return None
        
        # Add states with retry configuration
        agent.add_state("unreliable_call", unreliable_service_call, max_retries=5)
        agent.add_state("process", process_result)
        
        # Execute with error recovery
        await agent.run(timeout=20.0)
        
        # Verify successful recovery
        assert agent.status.value == "completed"
        assert failure_count == 3  # 2 failures + 1 success
        
        final_result = agent.shared_state.get("output_final_result")
        assert final_result == "processed_success_after_retries"
    
    @pytest.mark.asyncio
    async def test_workflow_with_conditional_paths(self):
        """Test workflow with conditional execution paths."""
        agent = Agent("conditional_workflow")
        
        async def analyze_data(context: Context) -> StateResult:
            # Simulate data analysis
            data_quality = 0.85  # 85% quality score
            context.set_variable("quality_score", data_quality)
            
            if data_quality >= 0.9:
                return "high_quality_path"
            elif data_quality >= 0.7:
                return "medium_quality_path"
            else:
                return "low_quality_path"
        
        async def high_quality_processing(context: Context) -> StateResult:
            context.set_variable("processing_type", "advanced")
            context.set_variable("confidence", 0.95)
            return "finalize"
        
        async def medium_quality_processing(context: Context) -> StateResult:
            context.set_variable("processing_type", "standard")
            context.set_variable("confidence", 0.8)
            return "additional_validation"
        
        async def low_quality_processing(context: Context) -> StateResult:
            context.set_variable("processing_type", "basic")
            context.set_variable("confidence", 0.6)
            return "data_cleaning"
        
        async def additional_validation(context: Context) -> StateResult:
            confidence = context.get_variable("confidence")
            context.set_variable("confidence", confidence + 0.1)
            return "finalize"
        
        async def data_cleaning(context: Context) -> StateResult:
            context.set_variable("cleaned", True)
            confidence = context.get_variable("confidence")
            context.set_variable("confidence", confidence + 0.2)
            return "finalize"
        
        async def finalize_processing(context: Context) -> StateResult:
            processing_type = context.get_variable("processing_type")
            confidence = context.get_variable("confidence")
            
            result = {
                "processing_type": processing_type,
                "final_confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context.set_output("processing_result", result)
            return None
        
        # Add all states
        agent.add_state("analyze", analyze_data)
        agent.add_state("high_quality", high_quality_processing)
        agent.add_state("medium_quality", medium_quality_processing)
        agent.add_state("low_quality", low_quality_processing)
        agent.add_state("validate", additional_validation)
        agent.add_state("clean", data_cleaning)
        agent.add_state("finalize", finalize_processing)
        
        # Execute workflow
        await agent.run(timeout=15.0)
        
        # Verify conditional path was taken
        assert agent.status.value == "completed"
        
        result = agent.shared_state.get("output_processing_result")
        assert result["processing_type"] == "standard"  # Medium quality path
        assert result["final_confidence"] == 0.9  # 0.8 + 0.1 from validation


@pytest.mark.integration  
class TestWorkflowEngineIntegration:
    """Integration tests for the workflow engine."""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock storage for engine tests."""
        storage = Mock()
        storage.initialize = AsyncMock()
        storage.save_event = AsyncMock()
        storage.load_events = AsyncMock(return_value=[])
        storage.save_checkpoint = AsyncMock()
        storage.load_checkpoint = AsyncMock(return_value=None)
        storage.close = AsyncMock()
        return storage
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle_management(self, mock_storage):
        """Test complete engine lifecycle."""
        engine = WorkflowEngine(mock_storage)
        
        # Start engine
        await engine.start()
        assert mock_storage.initialize.called
        
        # Create and execute workflow
        agent = Agent("test_workflow")
        
        async def simple_task(context: Context) -> StateResult:
            context.set_output("result", "completed")
            return None
        
        agent.add_state("task", simple_task)
        
        workflow_id = await engine.create_workflow("test", agent)
        assert workflow_id is not None
        
        # Execute workflow
        await engine.execute_workflow(workflow_id, agent, timeout=10.0)
        
        # Verify workflow completed
        status = await engine.get_workflow_status(workflow_id)
        assert status["is_running"] is False
        
        # Stop engine
        await engine.stop()
        assert mock_storage.close.called
    
    @pytest.mark.asyncio
    async def test_engine_pause_resume_workflow(self, mock_storage):
        """Test workflow pause and resume through engine."""
        engine = WorkflowEngine(mock_storage)
        await engine.start()
        
        agent = Agent("pausable_workflow")
        execution_log = []
        
        async def pausable_task(context: Context) -> StateResult:
            execution_log.append("started")
            await asyncio.sleep(0.5)  # Give time for pause
            execution_log.append("completed")
            return None
        
        agent.add_state("pausable", pausable_task)
        
        workflow_id = await engine.create_workflow("pausable", agent)
        
        # Start execution
        exec_task = asyncio.create_task(
            engine.execute_workflow(workflow_id, agent, timeout=10.0)
        )
        
        # Pause after brief delay
        await asyncio.sleep(0.1)
        pause_result = await engine.pause_workflow(workflow_id)
        
        assert pause_result["status"] == "paused"
        assert mock_storage.save_checkpoint.called
        
        # Resume execution
        await engine.resume_workflow(workflow_id, agent)
        await exec_task
        
        assert len(execution_log) == 2
        
        await engine.stop()


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_monitoring_pipeline(self):
        """Test complete monitoring pipeline with events, metrics, and alerts."""
        # Setup monitoring components
        event_stream = EventStream()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        
        await event_stream.start()
        await alert_manager.start()
        
        # Track monitoring data
        events_received = []
        metrics_received = {}
        alerts_fired = []
        
        # Subscribe to events
        event_stream.subscribe(lambda e: events_received.append(e))
        
        # Create test alert destination
        class TestAlertDestination:
            async def send(self, alert):
                alerts_fired.append(alert)
        
        # Configure high error rate alert
        high_error_alert = AlertRule(
            name="high_error_rate",
            condition=AlertCondition(
                metric="error_rate",
                operator=">",
                threshold=0.2,
                window=timedelta(seconds=10)
            ),
            severity=AlertSeverity.ERROR,
            message="Error rate exceeded threshold",
            destinations=[TestAlertDestination()]
        )
        
        alert_manager.add_rule(high_error_alert)
        
        # Create monitored workflow
        @agent_monitor(
            metrics=MetricType.ALL,
            metrics_callback=lambda m: metrics_received.update(m)
        )
        async def monitored_workflow():
            agent = Agent("monitored_workflow")
            
            # States with different success/failure patterns
            async def success_state(context: Context) -> StateResult:
                context.set_output("result", "success")
                return None
            
            failure_count = 0
            async def failing_state(context: Context) -> StateResult:
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 3:
                    raise Exception("Simulated failure")
                return None
            
            agent.add_state("success", success_state)
            agent.add_state("failing", failing_state, max_retries=5)
            
            await agent.run(timeout=15.0)
            return agent
        
        # Execute monitored workflow
        agent = await monitored_workflow()
        
        # Simulate high error rate for alerts
        for i in range(5):
            await alert_manager.record_metric("error_rate", 0.3)
        
        # Trigger alert evaluation
        await alert_manager._evaluate_rules()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify monitoring data
        assert agent.status.value == "completed"
        assert len(metrics_received) > 0
        assert "states" in metrics_received
        
        # Verify alert was fired
        assert len(alerts_fired) > 0
        assert alerts_fired[0].rule_name == "high_error_rate"
        
        # Cleanup
        await event_stream.stop()
        await alert_manager.stop()


@pytest.mark.integration
class TestReplayAndDeterminism:
    """Integration tests for replay and determinism checking."""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock storage with event history."""
        storage = Mock()
        
        # Mock events for replay
        mock_events = [
            WorkflowEvent("test_workflow", EventType.WORKFLOW_STARTED, datetime.utcnow(), {}),
            WorkflowEvent("test_workflow", EventType.STATE_STARTED, datetime.utcnow(), {"state_name": "step1"}),
            WorkflowEvent("test_workflow", EventType.STATE_COMPLETED, datetime.utcnow(), {"state_name": "step1", "result": "done"}),
            WorkflowEvent("test_workflow", EventType.WORKFLOW_COMPLETED, datetime.utcnow(), {}),
        ]
        
        storage.load_events = AsyncMock(return_value=mock_events)
        storage.load_checkpoint = AsyncMock(return_value=None)
        return storage
    
    @pytest.mark.asyncio
    async def test_deterministic_workflow_replay(self, mock_storage):
        """Test replay of deterministic workflow."""
        determinism_checker = DeterminismChecker()
        replay_engine = ReplayEngine(mock_storage, determinism_checker)
        
        # Create deterministic workflow
        agent = Agent("deterministic_workflow")
        
        async def deterministic_step(context: Context) -> StateResult:
            # Deterministic operations only
            data = context.get_variable("input_data", [1, 3, 2, 5, 4])
            sorted_data = sorted(data)  # Deterministic sort
            context.set_variable("sorted_result", sorted_data)
            return None
        
        agent.add_state("step1", deterministic_step)
        
        # Register for determinism checking
        determinism_checker.register_state("step1", deterministic_step)
        
        # Replay workflow
        result = await replay_engine.replay(
            "test_workflow",
            agent,
            mode=ReplayMode.VALIDATE
        )
        
        assert result.success
        assert result.events_replayed > 0
    
    @pytest.mark.asyncio  
    async def test_workflow_state_consistency(self):
        """Test workflow state consistency across executions."""
        # Run same workflow multiple times and verify consistent results
        results = []
        
        for run in range(3):
            agent = Agent(f"consistency_test_{run}")
            
            async def consistent_computation(context: Context) -> StateResult:
                # Deterministic computation
                data = list(range(10))
                result = sum(x * x for x in data)  # Sum of squares
                context.set_output("computation_result", result)
                return None
            
            agent.add_state("compute", consistent_computation)
            await agent.run(timeout=5.0)
            
            result = agent.shared_state.get("output_computation_result")
            results.append(result)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        assert results[0] == 285  # Sum of squares 0^2 + 1^2 + ... + 9^2


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])