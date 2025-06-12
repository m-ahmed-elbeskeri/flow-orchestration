"""
Complete execution monitoring router with real functionality
"""

import asyncio
import time
from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uuid  

from core.agent.state import StateStatus
from core.monitoring.events import EventType
from core.resources.requirements import ResourceType

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/workflows/{workflow_id}/executions/{execution_id}")

@router.get("/")
async def get_execution_details(workflow_id: str, execution_id: str):
    """Get actual execution details."""
    try:
        from ...api.rest.app import app_state
        
        execution = app_state["executions"].get(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        if execution.get("workflow_id") != workflow_id:
            raise HTTPException(status_code=404, detail="Execution not found for this workflow")
        
        # Get agent if still running
        agent = app_state["agents"].get(execution_id)
        if agent:
            execution["agent_status"] = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
            execution["completed_states"] = list(agent.completed_states)
            execution["running_states"] = list(agent._running_states) if hasattr(agent, '_running_states') else []
            execution["total_states"] = len(agent.states)
            
        return execution
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/states")
async def get_execution_states(workflow_id: str, execution_id: str):
    """Get real execution state information."""
    try:
        from ...api.rest.app import app_state
        
        # Check if execution exists
        if execution_id not in app_state["executions"]:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        agent = app_state["agents"].get(execution_id)
        if not agent:
            # If agent is not running, try to get from storage/events
            storage = app_state.get("storage")
            if storage:
                try:
                    events = await storage.load_events(workflow_id)
                    states = []
                    state_data = {}
                    
                    for event in events:
                        if event.data.get("execution_id") == execution_id and event.state_name:
                            state_name = event.state_name
                            
                            if state_name not in state_data:
                                state_data[state_name] = {
                                    "name": state_name,
                                    "status": "pending",
                                    "startTime": None,
                                    "endTime": None,
                                    "duration": None,
                                    "attempts": 1,
                                    "dependencies": [],
                                    "transitions": [],
                                    "error": None
                                }
                            
                            if event.event_type == EventType.STATE_STARTED:
                                state_data[state_name]["status"] = "running"
                                state_data[state_name]["startTime"] = event.timestamp.isoformat() + "Z"
                            elif event.event_type == EventType.STATE_COMPLETED:
                                state_data[state_name]["status"] = "completed"
                                state_data[state_name]["endTime"] = event.timestamp.isoformat() + "Z"
                                if state_data[state_name]["startTime"]:
                                    start_time = datetime.fromisoformat(state_data[state_name]["startTime"].replace('Z', '+00:00'))
                                    end_time = event.timestamp
                                    state_data[state_name]["duration"] = int((end_time - start_time).total_seconds() * 1000)
                            elif event.event_type == EventType.STATE_FAILED:
                                state_data[state_name]["status"] = "failed"
                                state_data[state_name]["endTime"] = event.timestamp.isoformat() + "Z"
                                state_data[state_name]["error"] = event.data.get("error", "Unknown error")
                    
                    return list(state_data.values())
                except Exception as e:
                    logger.warning(f"Could not load events from storage: {str(e)}")
            
            # Return empty states if no agent and no storage
            return []
        
        # Get live state information from agent
        states = []
        for state_name, metadata in agent.state_metadata.items():
            state_data = {
                "name": state_name,
                "status": metadata.status.value if hasattr(metadata.status, 'value') else str(metadata.status),
                "startTime": datetime.fromtimestamp(metadata.last_execution).isoformat() + "Z" if metadata.last_execution else None,
                "endTime": datetime.fromtimestamp(metadata.last_success).isoformat() + "Z" if metadata.last_success else None,
                "duration": int((metadata.last_success - metadata.last_execution) * 1000) if metadata.last_success and metadata.last_execution else None,
                "attempts": metadata.attempts,
                "dependencies": list(metadata.dependencies.keys()),
                "transitions": [],
                "error": None
            }
            
            # Add error information if state failed
            if metadata.status == StateStatus.FAILED:
                state_data["error"] = "State execution failed"
            
            states.append(state_data)
        
        return states
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution states: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_execution_metrics(workflow_id: str, execution_id: str):
    """Get real execution metrics."""
    try:
        from ...api.rest.app import app_state
        
        # Check if execution exists
        if execution_id not in app_state["executions"]:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        agent = app_state["agents"].get(execution_id)
        execution_data = app_state["executions"][execution_id]
        
        # If no agent but execution exists, return basic metrics
        if not agent:
            return {
                "totalStates": 0,
                "completedStates": 0,
                "failedStates": 0,
                "activeStates": 0,
                "totalExecutionTime": 0,
                "avgStateTime": 0,
                "resourceUtilization": {"cpu": 0, "memory": 0, "network": 0},
                "throughput": 0,
                "errorRate": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "execution_status": execution_data.get("status", "unknown")
            }
        
        # Calculate metrics from live agent
        total_states = len(agent.states)
        completed_states = len(agent.completed_states)
        running_states = len(agent._running_states) if hasattr(agent, '_running_states') else 0
        failed_states = len([m for m in agent.state_metadata.values() if m.status == StateStatus.FAILED])
        
        # Calculate resource utilization - handle None resource_pool
        resource_pool = app_state.get("resource_pool")
        resource_util = {"cpu": 0, "memory": 0, "network": 0}
        if resource_pool:
            try:
                stats = resource_pool.get_usage_stats()
                resource_util = {
                    "cpu": getattr(stats.get(ResourceType.CPU, type('obj', (object,), {'current_usage': 0})()), 'current_usage', 0),
                    "memory": getattr(stats.get(ResourceType.MEMORY, type('obj', (object,), {'current_usage': 0})()), 'current_usage', 0),
                    "network": getattr(stats.get(ResourceType.NETWORK, type('obj', (object,), {'current_usage': 0})()), 'current_usage', 0)
                }
            except Exception as e:
                logger.warning(f"Could not get resource utilization: {str(e)}")
        
        # Calculate execution time
        start_time = agent.session_start if hasattr(agent, 'session_start') and agent.session_start else time.time()
        execution_time = int((time.time() - start_time) * 1000)
        
        metrics = {
            "totalStates": total_states,
            "completedStates": completed_states,
            "failedStates": failed_states,
            "activeStates": running_states,
            "totalExecutionTime": execution_time,
            "avgStateTime": 0,
            "resourceUtilization": resource_util,
            "throughput": completed_states / (time.time() - start_time) if start_time else 0,
            "errorRate": failed_states / total_states if total_states > 0 else 0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Calculate average state time
        total_time = 0
        executed_states = 0
        for metadata in agent.state_metadata.values():
            if metadata.last_execution and metadata.last_success:
                total_time += (metadata.last_success - metadata.last_execution)
                executed_states += 1
        
        if executed_states > 0:
            metrics["avgStateTime"] = int((total_time / executed_states) * 1000)
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events")
async def get_execution_events(
    workflow_id: str, 
    execution_id: str,
    limit: int = Query(50, ge=1, le=1000),
    level: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    since: Optional[str] = Query(None)
):
    """Get real execution events."""
    try:
        from ...api.rest.app import app_state
        
        storage = app_state.get("storage")
        if not storage:
            # Return basic events if no storage
            return []
        
        # Load events from storage
        try:
            events = await storage.load_events(workflow_id)
        except Exception as e:
            logger.warning(f"Could not load events from storage: {str(e)}")
            return []
        
        # Filter events for this execution
        execution_events = []
        for event in events:
            if event.data.get("execution_id") == execution_id:
                event_data = {
                    "id": event.event_id or str(uuid.uuid4()),
                    "timestamp": event.timestamp.isoformat() + "Z",
                    "type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                    "state": event.state_name,
                    "message": event.data.get("message", f"Event: {event.event_type}"),
                    "level": event.data.get("level", "info"),
                    "metadata": event.data
                }
                execution_events.append(event_data)
        
        # Apply filters
        if level:
            execution_events = [e for e in execution_events if e["level"] == level]
        if event_type:
            execution_events = [e for e in execution_events if e["type"] == event_type]
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                execution_events = [e for e in execution_events 
                                  if datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')) >= since_dt]
            except ValueError:
                logger.warning(f"Invalid since parameter: {since}")
        
        # Sort by timestamp descending and apply limit
        execution_events.sort(key=lambda x: x["timestamp"], reverse=True)
        execution_events = execution_events[:limit]
        
        return execution_events
        
    except Exception as e:
        logger.error(f"Failed to get execution events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_execution_alerts(workflow_id: str, execution_id: str):
    """Get real execution alerts."""
    try:
        from ...api.rest.app import app_state
        
        alert_manager = app_state.get("alert_manager")
        if not alert_manager:
            return []
        
        # Get alerts related to this execution
        alerts = []
        try:
            for alert in alert_manager.active_alerts:
                if alert.labels.get("execution_id") == execution_id:
                    alert_data = {
                        "id": alert.alert_id,
                        "type": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
                        "message": alert.message,
                        "timestamp": alert.fired_at.isoformat() + "Z" if alert.fired_at else datetime.utcnow().isoformat() + "Z",
                        "state": alert.labels.get("state_name"),
                        "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
                    }
                    alerts.append(alert_data)
        except Exception as e:
            logger.warning(f"Could not get alerts: {str(e)}")
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get execution alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pause")
async def pause_execution(workflow_id: str, execution_id: str):
    """Pause actual execution."""
    try:
        from ...api.rest.app import app_state
        
        agent = app_state["agents"].get(execution_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Execution not found or not running")
        
        checkpoint = await agent.pause()
        
        # Update execution status
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "paused"
            from ...api.rest.app import save_executions_to_storage
            save_executions_to_storage()
        
        return {"success": True, "message": "Execution paused", "checkpoint_id": checkpoint.timestamp}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resume")
async def resume_execution(workflow_id: str, execution_id: str):
    """Resume actual execution."""
    try:
        from ...api.rest.app import app_state
        
        agent = app_state["agents"].get(execution_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        await agent.resume()
        
        # Update execution status
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "running"
            from ...api.rest.app import save_executions_to_storage
            save_executions_to_storage()
        
        return {"success": True, "message": "Execution resumed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel")
async def cancel_execution(workflow_id: str, execution_id: str):
    """Cancel actual execution."""
    try:
        from ...api.rest.app import app_state
        
        agent = app_state["agents"].get(execution_id)
        if agent:
            await agent.cancel_all()
            del app_state["agents"][execution_id]
        
        # Update execution status
        if execution_id in app_state["executions"]:
            app_state["executions"][execution_id]["status"] = "cancelled"
            app_state["executions"][execution_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            from ...api.rest.app import save_executions_to_storage
            save_executions_to_storage()
        
        return {"success": True, "message": "Execution cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/states/{state_name}/retry")
async def retry_state(workflow_id: str, execution_id: str, state_name: str):
    """Retry a specific state."""
    try:
        from ...api.rest.app import app_state
        
        agent = app_state["agents"].get(execution_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Execution not found or not running")
        
        # Reset state metadata and retry
        if state_name in agent.state_metadata:
            metadata = agent.state_metadata[state_name]
            metadata.status = StateStatus.PENDING
            metadata.attempts = 0
            
            # Run the state
            await agent.run_state(state_name)
        else:
            raise HTTPException(status_code=404, detail=f"State '{state_name}' not found in execution")
        
        return {"success": True, "message": f"State '{state_name}' retried"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def execution_websocket(websocket: WebSocket, workflow_id: str, execution_id: str):
    """Real-time execution monitoring websocket."""
    await websocket.accept()
    logger.info(f"WebSocket connected for execution {execution_id}")
    
    try:
        from ...api.rest.app import app_state
        
        while True:
            # Check if execution still exists
            if execution_id not in app_state["executions"]:
                await websocket.send_json({
                    "type": "execution_ended",
                    "data": {"message": "Execution not found"},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                break
            
            # Send real-time updates
            agent = app_state["agents"].get(execution_id)
            execution_data = app_state["executions"][execution_id]
            
            try:
                if agent:
                    # Get current metrics and states
                    metrics_response = await get_execution_metrics(workflow_id, execution_id)
                    states_response = await get_execution_states(workflow_id, execution_id)
                    
                    update = {
                        "type": "execution_update",
                        "data": {
                            "execution": execution_data,
                            "metrics": metrics_response,
                            "states": states_response,
                            "agent_status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
                        },
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                else:
                    # Send execution status without agent data
                    update = {
                        "type": "execution_update",
                        "data": {
                            "execution": execution_data,
                            "message": f"Execution {execution_data.get('status', 'unknown')}"
                        },
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                
                await websocket.send_json(update)
                
                # If execution is completed/failed, send final message and break
                if execution_data.get("status") in ["completed", "failed", "cancelled"]:
                    await websocket.send_json({
                        "type": "execution_ended",
                        "data": {"message": f"Execution {execution_data.get('status')}"},
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    break
                    
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Error getting updates: {str(e)}"},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info(f"WebSocket closed for execution {execution_id}")