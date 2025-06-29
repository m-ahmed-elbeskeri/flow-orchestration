"""State management types and enums."""

from enum import Enum, IntEnum
from typing import Set, Dict, Union, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import uuid

# Import directly from specific modules to avoid circular imports
if TYPE_CHECKING:
    from core.agent.base import Agent, RetryPolicy
    from core.agent.dependencies import DependencyConfig

# Type definitions
StateResult = Union[str, List[Union[str, Tuple["Agent", str]]], None]


class Priority(IntEnum):
    """Priority levels for state execution."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateStatus(str, Enum):
    """State execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


from typing import Protocol, runtime_checkable
from core.agent.context import Context


@runtime_checkable
class StateFunction(Protocol):
    """Protocol for state functions."""
    async def __call__(self, context: Context) -> StateResult: ...


@dataclass
class StateMetadata:
    """Metadata for state execution."""
    status: StateStatus
    attempts: int = 0
    max_retries: int = 3
    resources: Optional["ResourceRequirements"] = None
    dependencies: Dict[str, "DependencyConfig"] = field(default_factory=dict)
    satisfied_dependencies: Set[str] = field(default_factory=set)
    last_execution: Optional[float] = None
    last_success: Optional[float] = None
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_policy: Optional["RetryPolicy"] = None
    priority: Priority = Priority.NORMAL

    def __post_init__(self):
        """Initialize resources if not provided."""
        if self.resources is None:
            from core.resources.requirements import ResourceRequirements
            self.resources = ResourceRequirements()


@dataclass(order=True)
class PrioritizedState:
    """State with priority for queue management."""
    priority: int
    timestamp: float
    state_name: str = field(compare=False)
    metadata: StateMetadata = field(compare=False)