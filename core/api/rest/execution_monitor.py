# core/api/rest/execution_monitor.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta

from core.monitoring.metrics import MetricsCollector
from core.monitoring.events import get_event_stream
from core.monitoring.alerts import AlertManager
from core.execution.engine import WorkflowEngine

router = APIRouter(prefix="/api/v1/workflows/{workflow_id}/executions/{execution_id}")

@router.get("/")
async def get_execution(workflow_id: str, execution_id: str):
    """Get execution details"""
    try:
        engine = get_workflow_engine()
        execution = await engine.get_execution(workflow_id, execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/states")
async def get_execution_states(workflow_id: str, execution_id: str):
    """Get all states in the execution with their current status"""
    try:
        engine = get_workflow_engine()
        states = await engine.get_execution_states(workflow_id, execution_id)
        return states
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_execution_metrics(workflow_id: str, execution_id: str):
    """Get real-time execution metrics"""
    try:
        collector = MetricsCollector()
        metrics = await collector.get_execution_metrics(workflow_id, execution_id)
        
        # Calculate derived metrics
        total_states = len(metrics.get('states', []))
        completed_states = len([s for s in metrics.get('states', []) if s['status'] == 'completed'])
        failed_states = len([s for s in metrics.get('states', []) if s['status'] == 'failed'])
        active_states = len([s for s in metrics.get('states', []) if s['status'] == 'running'])
        
        return {
            "totalStates": total_states,
            "completedStates": completed_states,
            "failedStates": failed_states,
            "activeStates": active_states,
            "totalExecutionTime": metrics.get('total_execution_time', 0),
            "avgStateTime": metrics.get('avg_state_time', 0),
            "resourceUtilization": {
                "cpu": metrics.get('cpu_usage', 0),
                "memory": metrics.get('memory_usage', 0),
                "network": metrics.get('network_usage', 0)
            },
            "throughput": metrics.get('throughput', 0),
            "errorRate": metrics.get('error_rate', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events")
async def get_execution_events(
    workflow_id: str, 
    execution_id: str,
    limit: int = 100,
    level: Optional[str] = None,
    event_type: Optional[str] = None
):
    """Get execution events/logs"""
    try:
        event_stream = get_event_stream()
        events = await event_stream.get_execution_events(
            workflow_id, 
            execution_id, 
            limit=limit,
            level=level,
            event_type=event_type
        )
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_execution_alerts(workflow_id: str, execution_id: str):
    """Get active alerts for the execution"""
    try:
        alert_manager = AlertManager()
        alerts = await alert_manager.get_execution_alerts(workflow_id, execution_id)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pause")
async def pause_execution(workflow_id: str, execution_id: str):
    """Pause the execution"""
    try:
        engine = get_workflow_engine()
        result = await engine.pause_execution(workflow_id, execution_id)
        return {"status": "paused", "checkpoint": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resume")
async def resume_execution(workflow_id: str, execution_id: str):
    """Resume the execution"""
    try:
        engine = get_workflow_engine()
        await engine.resume_execution(workflow_id, execution_id)
        return {"status": "resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel")
async def cancel_execution(workflow_id: str, execution_id: str):
    """Cancel the execution"""
    try:
        engine = get_workflow_engine()
        await engine.cancel_execution(workflow_id, execution_id)
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/states/{state_name}/retry")
async def retry_state(workflow_id: str, execution_id: str, state_name: str):
    """Retry a failed state"""
    try:
        engine = get_workflow_engine()
        await engine.retry_state(workflow_id, execution_id, state_name)
        return {"status": "retrying", "state": state_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@router.websocket("/ws")
async def execution_websocket(websocket: WebSocket, workflow_id: str, execution_id: str):
    """WebSocket endpoint for real-time execution monitoring"""
    await websocket.accept()
    
    try:
        # Subscribe to execution events
        event_stream = get_event_stream()
        
        async def send_updates():
            while True:
                # Get latest metrics and events
                metrics = await get_execution_metrics(workflow_id, execution_id)
                events = await get_execution_events(workflow_id, execution_id, limit=10)
                alerts = await get_execution_alerts(workflow_id, execution_id)
                
                update = {
                    "type": "execution_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "metrics": metrics,
                        "recent_events": events,
                        "alerts": alerts
                    }
                }
                
                await websocket.send_json(update)
                await asyncio.sleep(2)  # Update every 2 seconds
        
        await send_updates()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})