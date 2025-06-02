"""SQLite storage backend implementation."""

import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiosqlite
from pathlib import Path
import os

from core.storage.interface import StorageBackend
from core.storage.events import WorkflowEvent, EventType
from core.agent.checkpoint import AgentCheckpoint


class SQLiteBackend(StorageBackend):
    """SQLite implementation of storage backend."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.db_path = self._parse_database_url(database_url)
        self._connection: Optional[aiosqlite.Connection] = None
    
    def _parse_database_url(self, database_url: str) -> str:
        """Parse database URL to get file path."""
        # Handle various SQLite URL formats
        if database_url.startswith("sqlite+aiosqlite:///"):
            return database_url.replace("sqlite+aiosqlite:///", "")
        elif database_url.startswith("sqlite:///"):
            return database_url.replace("sqlite:///", "")
        elif database_url.startswith("sqlite://"):
            # Remove sqlite:// but keep the path
            path = database_url.replace("sqlite://", "")
            # Handle Windows paths like sqlite://./workflows.db
            if path.startswith("./"):
                return path
            return path
        else:
            # Assume it's already a file path
            return database_url
    
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        try:
            # Ensure the file path is absolute and valid
            if not os.path.isabs(self.db_path):
                # Make it relative to current working directory
                self.db_path = os.path.abspath(self.db_path)
            
            # Ensure directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self._connection = await aiosqlite.connect(self.db_path)
            
            # Create tables
            await self._create_tables()
            
            print(f"✅ SQLite database initialized at: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Failed to initialize SQLite database: {e}")
            print(f"Database URL: {self.database_url}")
            print(f"Parsed path: {self.db_path}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS workflow_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                workflow_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                state_name TEXT,
                timestamp DATETIME NOT NULL,
                data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_workflow_events_workflow_id 
                ON workflow_events(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_workflow_events_timestamp 
                ON workflow_events(timestamp);
            
            CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                checkpoint_data BLOB NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_workflow_id 
                ON workflow_checkpoints(workflow_id);
            
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                parameters TEXT,
                result TEXT,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_executions_workflow_id 
                ON executions(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_executions_status 
                ON executions(status);
        """)
        
        await self._connection.commit()
    
    async def close(self) -> None:
        """Close the storage backend."""
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    async def save_event(self, event: WorkflowEvent) -> None:
        """Save a workflow event."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        await self._connection.execute(
            """
            INSERT INTO workflow_events 
                (event_id, workflow_id, event_type, state_name, timestamp, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.workflow_id,
                event.event_type.value,
                event.state_name,
                event.timestamp.isoformat(),
                json.dumps(event.data)
            )
        )
        
        # Update workflow status if needed
        if event.event_type == EventType.WORKFLOW_CREATED:
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO workflows (workflow_id, name, agent_name, status, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.workflow_id,
                    event.data.get("name", ""),
                    event.data.get("agent_name", ""),
                    "created",
                    json.dumps(event.data.get("metadata", {}))
                )
            )
        else:
            status_map = {
                EventType.WORKFLOW_STARTED: "running",
                EventType.WORKFLOW_COMPLETED: "completed",
                EventType.WORKFLOW_FAILED: "failed",
                EventType.WORKFLOW_CANCELLED: "cancelled",
                EventType.WORKFLOW_PAUSED: "paused",
                EventType.WORKFLOW_RESUMED: "running"
            }
            
            if event.event_type in status_map:
                await self._connection.execute(
                    """
                    UPDATE workflows 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE workflow_id = ?
                    """,
                    (status_map[event.event_type], event.workflow_id)
                )
        
        await self._connection.commit()
    
    async def load_events(
        self,
        workflow_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[WorkflowEvent]:
        """Load events for a workflow."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        query = "SELECT * FROM workflow_events WHERE workflow_id = ?"
        params = [workflow_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        
        events = []
        for row in rows:
            event = WorkflowEvent(
                workflow_id=row[2],
                event_type=EventType(row[3]),
                timestamp=datetime.fromisoformat(row[5]),
                data=json.loads(row[6]),
                state_name=row[4],
                event_id=row[1]
            )
            events.append(event)
        
        return events
    
    async def save_checkpoint(
        self,
        workflow_id: str,
        checkpoint: AgentCheckpoint
    ) -> None:
        """Save an agent checkpoint."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        checkpoint_data = pickle.dumps(checkpoint)
        
        await self._connection.execute(
            """
            INSERT INTO workflow_checkpoints (workflow_id, checkpoint_data, timestamp)
            VALUES (?, ?, ?)
            """,
            (workflow_id, checkpoint_data, datetime.utcnow().isoformat())
        )
        
        await self._connection.commit()
    
    async def load_checkpoint(
        self,
        workflow_id: str
    ) -> Optional[AgentCheckpoint]:
        """Load the latest checkpoint for a workflow."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        cursor = await self._connection.execute(
            """
            SELECT checkpoint_data FROM workflow_checkpoints
            WHERE workflow_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (workflow_id,)
        )
        
        row = await cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        
        return None
    
    async def list_workflows(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filtering."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        query = "SELECT * FROM workflows"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if key in ["name", "agent_name", "status"]:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        
        workflows = []
        for row in rows:
            workflows.append({
                "workflow_id": row[0],
                "name": row[1],
                "agent_name": row[2],
                "status": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
                "updated_at": row[6]
            })
        
        return workflows
    
    async def save_execution(
        self,
        execution_id: str,
        workflow_id: str,
        status: str,
        started_at: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Save execution data."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        await self._connection.execute(
            """
            INSERT OR REPLACE INTO executions 
                (execution_id, workflow_id, status, started_at, completed_at, parameters, result, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                workflow_id,
                status,
                started_at.isoformat(),
                completed_at.isoformat() if completed_at else None,
                json.dumps(parameters) if parameters else None,
                json.dumps(result) if result else None,
                error_message
            )
        )
        
        await self._connection.commit()
    
    async def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution by ID."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        cursor = await self._connection.execute(
            "SELECT * FROM executions WHERE execution_id = ?",
            (execution_id,)
        )
        
        row = await cursor.fetchone()
        if row:
            return {
                "execution_id": row[0],
                "workflow_id": row[1],
                "status": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "parameters": json.loads(row[5]) if row[5] else None,
                "result": json.loads(row[6]) if row[6] else None,
                "error_message": row[7],
                "created_at": row[8]
            }
        
        return None
    
    async def delete_workflow(self, workflow_id: str) -> None:
        """Delete all data for a workflow."""
        if not self._connection:
            raise RuntimeError("Storage backend not initialized")
        
        await self._connection.execute(
            "DELETE FROM executions WHERE workflow_id = ?",
            (workflow_id,)
        )
        
        await self._connection.execute(
            "DELETE FROM workflow_events WHERE workflow_id = ?",
            (workflow_id,)
        )
        
        await self._connection.execute(
            "DELETE FROM workflow_checkpoints WHERE workflow_id = ?",
            (workflow_id,)
        )
        
        await self._connection.execute(
            "DELETE FROM workflows WHERE workflow_id = ?",
            (workflow_id,)
        )
        
        await self._connection.commit()