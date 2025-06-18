"""Resource requirements definitions."""

from dataclasses import dataclass
from enum import Flag, auto
from typing import Optional


class ResourceType(Flag):
    """Types of resources that can be managed."""
    NONE = 0
    CPU = auto()
    MEMORY = auto()
    IO = auto()
    NETWORK = auto()
    GPU = auto()
    ALL = CPU | MEMORY | IO | NETWORK | GPU


@dataclass
class ResourceRequirements:
    """Resource requirements for state execution."""
    cpu_units: float = 1.0
    memory_mb: float = 100.0
    io_weight: float = 1.0
    network_weight: float = 1.0
    gpu_units: float = 0.0
    priority_boost: int = 0  # Use int instead of Priority enum to avoid circular import
    timeout: Optional[float] = None
    resource_types: ResourceType = ResourceType.ALL

    @property
    def priority(self):
        """Get priority from priority_boost."""
        from core.agent.state import Priority
        # Map priority_boost to Priority enum
        if self.priority_boost >= 3:
            return Priority.CRITICAL
        elif self.priority_boost >= 2:
            return Priority.HIGH
        elif self.priority_boost >= 1:
            return Priority.NORMAL
        else:
            return Priority.LOW

    @priority.setter
    def priority(self, value):
        """Set priority_boost from Priority enum."""
        if hasattr(value, 'value'):
            self.priority_boost = value.value
        else:
            self.priority_boost = int(value)