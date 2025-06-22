"""Enhanced deadlock detection for workflow execution with improved reliability and performance"""

from typing import Dict, Set, Optional, List, Any, Tuple, Callable
import asyncio
import weakref
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from collections import defaultdict, deque
from enum import Enum, auto
import uuid


logger = logging.getLogger(__name__)


class DeadlockResolutionStrategy(Enum):
    """Strategies for resolving deadlocks"""
    RAISE_EXCEPTION = auto()
    KILL_YOUNGEST = auto()
    KILL_OLDEST = auto()
    PREEMPT_RESOURCES = auto()
    ROLLBACK_TRANSACTION = auto()
    LOG_ONLY = auto()


class DeadlockError(Exception):
    """Raised when a deadlock is detected"""

    def __init__(self, cycle: List[str], detection_id: str = None, message: str = "Deadlock detected"):
        self.cycle = cycle
        self.detection_id = detection_id or str(uuid.uuid4())
        super().__init__(f"{message}: {' -> '.join(cycle)} (ID: {self.detection_id})")


@dataclass
class ResourceNode:
    """Node in resource wait graph"""
    resource_id: str
    resource_type: str
    holders: Set[str] = field(default_factory=set)
    waiters: Set[str] = field(default_factory=set)
    acquired_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    def is_free(self) -> bool:
        """Check if resource is free"""
        return len(self.holders) == 0

    def age_seconds(self) -> float:
        """Get age of resource in seconds"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class ProcessNode:
    """Node representing a process/state in wait graph"""
    process_id: str
    process_name: str
    holding: Set[str] = field(default_factory=set)
    waiting_for: Set[str] = field(default_factory=set)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0

    def is_blocked(self) -> bool:
        """Check if process is blocked"""
        return len(self.waiting_for) > 0

    def age_seconds(self) -> float:
        """Get age of process in seconds"""
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def blocked_duration_seconds(self) -> float:
        """Get how long process has been blocked"""
        if self.blocked_at:
            return (datetime.now(timezone.utc) - self.blocked_at).total_seconds()
        return 0.0

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


@dataclass
class CycleDetectionResult:
    """Result of cycle detection"""
    has_cycle: bool
    cycles: List[List[str]] = field(default_factory=list)
    detection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    graph_size: int = 0
    detection_duration_ms: float = 0.0

    def get_shortest_cycle(self) -> Optional[List[str]]:
        """Get the shortest detected cycle"""
        if not self.cycles:
            return None
        return min(self.cycles, key=len)

    def get_longest_cycle(self) -> Optional[List[str]]:
        """Get the longest detected cycle"""
        if not self.cycles:
            return None
        return max(self.cycles, key=len)


class DependencyGraph:
    """Enhanced graph for tracking dependencies and detecting cycles"""

    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[str, Set[str]] = {}
        self.reverse_edges: Dict[str, Set[str]] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.max_nodes = max_nodes
        self._cycle_cache: Dict[str, CycleDetectionResult] = {}
        self._cache_ttl = 5.0  # Cache for 5 seconds

    async def add_dependency(self, node: str, depends_on: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a dependency edge with optional metadata"""
        async with self._lock:
            # Check for capacity
            if len(self.nodes) >= self.max_nodes:
                await self._cleanup_old_nodes()

            if node not in self.nodes:
                self.nodes[node] = set()
                self.node_metadata[node] = metadata or {}
            if depends_on not in self.reverse_edges:
                self.reverse_edges[depends_on] = set()
                if depends_on not in self.node_metadata:
                    self.node_metadata[depends_on] = {}

            self.nodes[node].add(depends_on)
            self.reverse_edges[depends_on].add(node)

            # Invalidate cache when graph changes
            self._cycle_cache.clear()

    async def remove_dependency(self, node: str, depends_on: str):
        """Remove a dependency edge"""
        async with self._lock:
            if node in self.nodes:
                self.nodes[node].discard(depends_on)
                if not self.nodes[node]:
                    del self.nodes[node]
                    self.node_metadata.pop(node, None)

            if depends_on in self.reverse_edges:
                self.reverse_edges[depends_on].discard(node)
                if not self.reverse_edges[depends_on]:
                    del self.reverse_edges[depends_on]

            self._cycle_cache.clear()

    async def remove_node(self, node: str):
        """Remove a node and all its edges"""
        async with self._lock:
            # Remove outgoing edges
            if node in self.nodes:
                for dep in self.nodes[node]:
                    if dep in self.reverse_edges:
                        self.reverse_edges[dep].discard(node)
                del self.nodes[node]
                self.node_metadata.pop(node, None)

            # Remove incoming edges
            if node in self.reverse_edges:
                for dependent in self.reverse_edges[node]:
                    if dependent in self.nodes:
                        self.nodes[dependent].discard(node)
                del self.reverse_edges[node]

            self._cycle_cache.clear()

    async def _cleanup_old_nodes(self):
        """Remove old nodes to prevent memory growth"""
        # This is a simple LRU-style cleanup
        # You might want more sophisticated cleanup strategies
        nodes_to_remove = list(self.nodes.keys())[:len(self.nodes) // 4]
        for node in nodes_to_remove:
            await self.remove_node(node)

    def find_cycles(self, use_cache: bool = True) -> CycleDetectionResult:
        """Find all cycles in the graph using optimized DFS"""
        start_time = time.time()

        # Check cache first
        if use_cache:
            cache_key = self._get_graph_hash()
            cached_result = self._cycle_cache.get(cache_key)
            if cached_result and (time.time() - cached_result.detection_time.timestamp()) < self._cache_ttl:
                return cached_result

        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            """Optimized DFS to detect cycles"""
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all dependencies
            for neighbor in self.nodes.get(node, []):
                if dfs(neighbor):
                    # Don't return immediately to find all cycles
                    pass

            path.pop()
            rec_stack.remove(node)
            return False

        # Check all nodes
        for node in list(self.nodes.keys()):
            if node not in visited:
                dfs(node)

        detection_duration = (time.time() - start_time) * 1000  # Convert to ms

        result = CycleDetectionResult(
            has_cycle=len(cycles) > 0,
            cycles=cycles,
            graph_size=len(self.nodes),
            detection_duration_ms=detection_duration
        )

        # Cache the result
        if use_cache:
            cache_key = self._get_graph_hash()
            self._cycle_cache[cache_key] = result

        return result

    def _get_graph_hash(self) -> str:
        """Get a hash representing the current graph state"""
        # Simple hash based on node count and edge count
        edge_count = sum(len(deps) for deps in self.nodes.values())
        return f"{len(self.nodes)}:{edge_count}"

    def topological_sort(self) -> Optional[List[str]]:
        """Perform topological sort if no cycles exist"""
        # Check for cycles first
        if self.find_cycles().has_cycle:
            return None

        # Kahn's algorithm
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for node in self.nodes:
            if node not in in_degree:
                in_degree[node] = 0
            for dep in self.nodes[node]:
                in_degree[dep] += 1

        # Find nodes with no incoming edges
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Remove edges from this node
            for neighbor in self.nodes.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if all nodes were processed
        return result if len(result) == len(self.nodes) else None


class ResourceWaitGraph:
    """Enhanced wait-for graph for resource-based deadlock detection"""

    def __init__(self, max_resources: int = 5000, max_processes: int = 5000):
        self.resources: Dict[str, ResourceNode] = {}
        self.processes: Dict[str, ProcessNode] = {}
        self._lock = asyncio.Lock()
        self.max_resources = max_resources
        self.max_processes = max_processes
        self._wait_graph_cache: Optional[DependencyGraph] = None
        self._cache_invalidated = True

    async def add_resource(self, resource_id: str, resource_type: str = "generic"):
        """Add a resource to the graph"""
        async with self._lock:
            if len(self.resources) >= self.max_resources:
                await self._cleanup_old_resources()

            if resource_id not in self.resources:
                self.resources[resource_id] = ResourceNode(
                    resource_id=resource_id,
                    resource_type=resource_type
                )
                self._cache_invalidated = True

    async def add_process(self, process_id: str, process_name: str = "", priority: int = 0):
        """Add a process to the graph"""
        async with self._lock:
            if len(self.processes) >= self.max_processes:
                await self._cleanup_old_processes()

            if process_id not in self.processes:
                self.processes[process_id] = ProcessNode(
                    process_id=process_id,
                    process_name=process_name or process_id,
                    priority=priority
                )
                self._cache_invalidated = True

    async def acquire_resource(self, process_id: str, resource_id: str) -> bool:
        """Process acquires a resource"""
        async with self._lock:
            if resource_id not in self.resources:
                await self.add_resource(resource_id)
            if process_id not in self.processes:
                await self.add_process(process_id)

            resource = self.resources[resource_id]
            process = self.processes[process_id]

            # Check if resource is available
            if resource.is_free() or process_id in resource.holders:
                resource.holders.add(process_id)
                resource.acquired_at = datetime.now(timezone.utc)
                resource.access_count += 1
                process.holding.add(resource_id)
                process.waiting_for.discard(resource_id)
                process.update_activity()

                # Clear blocked status if not waiting for anything
                if not process.waiting_for:
                    process.blocked_at = None

                self._cache_invalidated = True
                return True
            else:
                # Process must wait
                resource.waiters.add(process_id)
                process.waiting_for.add(resource_id)
                if process.blocked_at is None:
                    process.blocked_at = datetime.now(timezone.utc)

                self._cache_invalidated = True
                return False

    async def release_resource(self, process_id: str, resource_id: str):
        """Process releases a resource"""
        async with self._lock:
            if resource_id in self.resources and process_id in self.processes:
                resource = self.resources[resource_id]
                process = self.processes[process_id]

                resource.holders.discard(process_id)
                process.holding.discard(resource_id)
                process.update_activity()

                # Check if any waiters can acquire (priority-based)
                if resource.is_free() and resource.waiters:
                    # Sort waiters by priority (higher priority first)
                    sorted_waiters = sorted(
                        resource.waiters,
                        key=lambda pid: self.processes.get(pid, ProcessNode("", "")).priority,
                        reverse=True
                    )

                    next_process = sorted_waiters[0]
                    resource.waiters.remove(next_process)

                    # Recursive acquisition
                    await self.acquire_resource(next_process, resource_id)

                self._cache_invalidated = True

    async def _cleanup_old_resources(self):
        """Clean up old unused resources"""
        now = datetime.now(timezone.utc)
        old_resources = [
            rid for rid, resource in self.resources.items()
            if resource.is_free() and
               len(resource.waiters) == 0 and
               (now - resource.created_at).total_seconds() > 300  # 5 minutes old
        ]

        for rid in old_resources[:len(self.resources) // 4]:  # Remove 25%
            del self.resources[rid]

    async def _cleanup_old_processes(self):
        """Clean up old inactive processes"""
        now = datetime.now(timezone.utc)
        old_processes = [
            pid for pid, process in self.processes.items()
            if len(process.holding) == 0 and
               len(process.waiting_for) == 0 and
               (now - process.last_activity).total_seconds() > 300  # 5 minutes inactive
        ]

        for pid in old_processes[:len(self.processes) // 4]:  # Remove 25%
            del self.processes[pid]

    async def detect_deadlock(self) -> CycleDetectionResult:
        """Detect deadlocks using optimized cycle detection in wait-for graph"""
        async with self._lock:
            # Build or reuse wait-for graph
            if self._cache_invalidated or self._wait_graph_cache is None:
                self._wait_graph_cache = DependencyGraph()

                # Add edges: if P1 waits for resource held by P2, add edge P1 -> P2
                for resource in self.resources.values():
                    for waiter in resource.waiters:
                        for holder in resource.holders:
                            if waiter != holder:
                                await self._wait_graph_cache.add_dependency(waiter, holder)

                self._cache_invalidated = False

            # Find cycles
            return self._wait_graph_cache.find_cycles()

    def get_blocked_processes(self) -> List[ProcessNode]:
        """Get all currently blocked processes"""
        return [
            proc for proc in self.processes.values()
            if proc.is_blocked()
        ]

    def get_resource_holders(self, resource_id: str) -> Set[str]:
        """Get processes holding a resource"""
        if resource_id in self.resources:
            return self.resources[resource_id].holders.copy()
        return set()

    def get_resource_waiters(self, resource_id: str) -> Set[str]:
        """Get processes waiting for a resource"""
        if resource_id in self.resources:
            return self.resources[resource_id].waiters.copy()
        return set()


class DeadlockDetector:
    """Enhanced deadlock detection with improved reliability and performance"""

    def __init__(
        self,
        agent: Any,
        detection_interval: float = 1.0,
        max_cycles: int = 100,
        resolution_strategy: DeadlockResolutionStrategy = DeadlockResolutionStrategy.LOG_ONLY,
        enable_metrics: bool = True
    ):
        self.agent = weakref.proxy(agent)
        self.detection_interval = detection_interval
        self.max_cycles = max_cycles
        self.resolution_strategy = resolution_strategy
        self.enable_metrics = enable_metrics

        self._dependency_graph = DependencyGraph()
        self._resource_graph = ResourceWaitGraph()
        self._lock = asyncio.Lock()
        self._detection_task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle: Optional[List[str]] = None
        self._detection_history: deque = deque(maxlen=100)
        self._metrics = {
            'total_detections': 0,
            'deadlocks_found': 0,
            'deadlocks_resolved': 0,
            'detection_errors': 0,
            'avg_detection_time_ms': 0.0
        }
        self._resolution_callbacks: List[Callable[[List[str]], bool]] = []

    async def start(self):
        """Start deadlock detection"""
        async with self._lock:
            if self._detection_task and not self._detection_task.done():
                logger.warning("Deadlock detector already running")
                return

            self._detection_task = asyncio.create_task(self._detect_deadlocks())
            logger.info("Deadlock detection started")

            if hasattr(self.agent, '_monitor'):
                self.agent._monitor.logger.info("deadlock_detection_started")

    async def stop(self):
        """Stop deadlock detection"""
        async with self._lock:
            if self._detection_task:
                self._detection_task.cancel()
                try:
                    await asyncio.wait_for(self._detection_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.warning("Deadlock detection task did not stop gracefully")
                finally:
                    self._detection_task = None

            logger.info("Deadlock detection stopped")

            if hasattr(self.agent, '_monitor'):
                self.agent._monitor.logger.info("deadlock_detection_stopped")

    def add_resolution_callback(self, callback: Callable[[List[str]], bool]):
        """Add a callback for custom deadlock resolution"""
        self._resolution_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive detector status"""
        return {
            "cycle_count": self._cycle_count,
            "last_cycle": self._last_cycle,
            "active": bool(self._detection_task and not self._detection_task.done()),
            "graph_size": len(self._dependency_graph.nodes),
            "resource_count": len(self._resource_graph.resources),
            "process_count": len(self._resource_graph.processes),
            "blocked_processes": len(self._resource_graph.get_blocked_processes()),
            "metrics": self._metrics.copy(),
            "resolution_strategy": self.resolution_strategy.name,
            "detection_interval": self.detection_interval
        }

    async def add_dependency(self, from_state: str, to_state: str):
        """Add a dependency between states"""
        await self._dependency_graph.add_dependency(from_state, to_state)

    async def remove_dependency(self, from_state: str, to_state: str):
        """Remove a dependency between states"""
        await self._dependency_graph.remove_dependency(from_state, to_state)

    async def acquire_resource(
        self,
        process_id: str,
        resource_id: str,
        process_name: Optional[str] = None,
        priority: int = 0
    ) -> bool:
        """Process attempts to acquire a resource"""
        if process_name:
            await self._resource_graph.add_process(process_id, process_name, priority)

        success = await self._resource_graph.acquire_resource(process_id, resource_id)

        # Check for deadlock immediately after failed acquisition
        if not success:
            try:
                result = await self._resource_graph.detect_deadlock()
                if result.has_cycle:
                    await self._handle_deadlock(result)
            except Exception as e:
                logger.error(f"Error during immediate deadlock check: {e}")

        return success

    async def release_resource(self, process_id: str, resource_id: str):
        """Process releases a resource"""
        await self._resource_graph.release_resource(process_id, resource_id)

    async def _detect_deadlocks(self):
        """Main deadlock detection loop with enhanced error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                await asyncio.sleep(self.detection_interval)

                detection_start = time.time()
                self._metrics['total_detections'] += 1

                # Check state dependencies
                state_result = self._dependency_graph.find_cycles()
                if state_result.has_cycle:
                    await self._handle_deadlock(state_result)

                # Check resource wait graph
                resource_result = await self._resource_graph.detect_deadlock()
                if resource_result.has_cycle:
                    await self._handle_deadlock(resource_result)

                # Update metrics
                detection_duration = (time.time() - detection_start) * 1000
                self._update_detection_metrics(detection_duration)

                # Keep detection history
                self._detection_history.append(state_result)

                # Reset error counter on successful detection
                consecutive_errors = 0

            except asyncio.CancelledError:
                logger.info("Deadlock detection cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                self._metrics['detection_errors'] += 1

                logger.error(f"Deadlock detection error: {e}")

                # Implement exponential backoff on errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping deadlock detection")
                    break

                # Exponential backoff
                error_delay = min(self.detection_interval * (2 ** consecutive_errors), 60.0)
                await asyncio.sleep(error_delay)

    def _update_detection_metrics(self, duration_ms: float):
        """Update detection performance metrics"""
        if self.enable_metrics:
            # Update average detection time with exponential moving average
            alpha = 0.1
            self._metrics['avg_detection_time_ms'] = (
                alpha * duration_ms +
                (1 - alpha) * self._metrics['avg_detection_time_ms']
            )

    async def _handle_deadlock(self, result: CycleDetectionResult):
        """Handle detected deadlock with configurable resolution strategies"""
        self._cycle_count += 1
        self._last_cycle = result.get_shortest_cycle()
        self._metrics['deadlocks_found'] += 1

        detection_id = str(uuid.uuid4())

        logger.error(
            f"Deadlock detected (ID: {detection_id}): "
            f"cycle_count={self._cycle_count}, "
            f"cycle={self._last_cycle}, "
            f"total_cycles={len(result.cycles)}"
        )

        # Notify agent monitor if available
        if hasattr(self.agent, '_monitor'):
            self.agent._monitor.logger.error(
                f"deadlock_detected: cycle={self._last_cycle}, id={detection_id}"
            )

        # Try custom resolution callbacks first
        resolved = False
        for callback in self._resolution_callbacks:
            try:
                if callback(self._last_cycle):
                    resolved = True
                    self._metrics['deadlocks_resolved'] += 1
                    logger.info(f"Deadlock {detection_id} resolved by custom callback")
                    break
            except Exception as e:
                logger.error(f"Resolution callback failed: {e}")

        # Apply configured resolution strategy if not resolved
        if not resolved:
            resolved = await self._apply_resolution_strategy(self._last_cycle, detection_id)

        # If still not resolved and strategy is to raise exception
        if not resolved and self.resolution_strategy == DeadlockResolutionStrategy.RAISE_EXCEPTION:
            raise DeadlockError(self._last_cycle, detection_id)

    async def _apply_resolution_strategy(self, cycle: List[str], detection_id: str) -> bool:
        """Apply the configured resolution strategy"""
        if self.resolution_strategy == DeadlockResolutionStrategy.LOG_ONLY:
            return True  # Just log, don't actually resolve

        elif self.resolution_strategy == DeadlockResolutionStrategy.KILL_YOUNGEST:
            return await self._kill_youngest_process(cycle, detection_id)

        elif self.resolution_strategy == DeadlockResolutionStrategy.KILL_OLDEST:
            return await self._kill_oldest_process(cycle, detection_id)

        elif self.resolution_strategy == DeadlockResolutionStrategy.PREEMPT_RESOURCES:
            return await self._preempt_resources(cycle, detection_id)

        return False

    async def _kill_youngest_process(self, cycle: List[str], detection_id: str) -> bool:
        """Kill the youngest process in the cycle"""
        try:
            youngest_process = min(
                (pid for pid in cycle if pid in self._resource_graph.processes),
                key=lambda pid: self._resource_graph.processes[pid].age_seconds()
            )

            await self._terminate_process(youngest_process, f"deadlock_resolution_{detection_id}")
            self._metrics['deadlocks_resolved'] += 1
            logger.info(f"Killed youngest process {youngest_process} to resolve deadlock {detection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to kill youngest process: {e}")
            return False

    async def _kill_oldest_process(self, cycle: List[str], detection_id: str) -> bool:
        """Kill the oldest process in the cycle"""
        try:
            oldest_process = max(
                (pid for pid in cycle if pid in self._resource_graph.processes),
                key=lambda pid: self._resource_graph.processes[pid].age_seconds()
            )

            await self._terminate_process(oldest_process, f"deadlock_resolution_{detection_id}")
            self._metrics['deadlocks_resolved'] += 1
            logger.info(f"Killed oldest process {oldest_process} to resolve deadlock {detection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to kill oldest process: {e}")
            return False

    async def _preempt_resources(self, cycle: List[str], detection_id: str) -> bool:
        """Preempt resources from processes in the cycle"""
        try:
            # Find process with most resources to preempt from
            processes_in_cycle = [
                pid for pid in cycle
                if pid in self._resource_graph.processes
            ]

            if not processes_in_cycle:
                return False

            victim_process = max(
                processes_in_cycle,
                key=lambda pid: len(self._resource_graph.processes[pid].holding)
            )

            process = self._resource_graph.processes[victim_process]
            resources_to_preempt = list(process.holding)

            # Release all resources held by victim process
            for resource_id in resources_to_preempt:
                await self._resource_graph.release_resource(victim_process, resource_id)

            self._metrics['deadlocks_resolved'] += 1
            logger.info(
                f"Preempted {len(resources_to_preempt)} resources from process "
                f"{victim_process} to resolve deadlock {detection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to preempt resources: {e}")
            return False

    async def _terminate_process(self, process_id: str, reason: str):
        """Terminate a process and clean up its resources"""
        if process_id in self._resource_graph.processes:
            process = self._resource_graph.processes[process_id]

            # Release all held resources
            for resource_id in list(process.holding):
                await self._resource_graph.release_resource(process_id, resource_id)

            # Remove from waiting lists
            for resource_id in list(process.waiting_for):
                if resource_id in self._resource_graph.resources:
                    self._resource_graph.resources[resource_id].waiters.discard(process_id)

            # Remove process
            del self._resource_graph.processes[process_id]

            # Remove from dependency graph
            await self._dependency_graph.remove_node(process_id)

    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get current dependency graph"""
        return dict(self._dependency_graph.nodes)

    def get_wait_graph(self) -> Dict[str, Dict[str, Any]]:
        """Get current wait-for graph with enhanced information"""
        graph = {}

        for process_id, process in self._resource_graph.processes.items():
            graph[process_id] = {
                "name": process.process_name,
                "holding": list(process.holding),
                "waiting_for": list(process.waiting_for),
                "blocked": process.is_blocked(),
                "blocked_duration_seconds": process.blocked_duration_seconds(),
                "age_seconds": process.age_seconds(),
                "priority": process.priority,
                "last_activity": process.last_activity.isoformat()
            }

        return graph

    def find_potential_deadlocks(self) -> List[Tuple[str, str]]:
        """Find potential deadlock situations before they occur"""
        potential = []

        # Check for circular wait conditions
        for p1_id, p1 in self._resource_graph.processes.items():
            for p2_id, p2 in self._resource_graph.processes.items():
                if p1_id == p2_id:
                    continue

                # Check if P1 holds what P2 wants and vice versa
                p1_holds_p2_wants = bool(p1.holding & p2.waiting_for)
                p2_holds_p1_wants = bool(p2.holding & p1.waiting_for)

                if p1_holds_p2_wants and p2_holds_p1_wants:
                    potential.append((p1_id, p2_id))

        return potential

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            **self._metrics,
            "detection_history_length": len(self._detection_history),
            "active_processes": len(self._resource_graph.processes),
            "active_resources": len(self._resource_graph.resources),
            "blocked_processes": len(self._resource_graph.get_blocked_processes())
        }