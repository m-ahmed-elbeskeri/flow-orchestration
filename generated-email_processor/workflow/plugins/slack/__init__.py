"""
Slack integration for messaging and workspace operations

Standalone plugin implementation.
Original: slack v1.0.0
"""

from pathlib import Path
from typing import Dict, Type

from ..base import Plugin, PluginState, PluginManifest

# Import state classes
from .states import (
    SendMessageState,
    ReadMessagesState,
    UploadFileState,
    CreateChannelState,
)


class SlackPlugin(Plugin):
    """Slack integration for messaging and workspace operations"""
    
    def __init__(self):
        # Create manifest
        manifest = PluginManifest(
            name="slack",
            version="1.0.0",
            description="Slack integration for messaging and workspace operations",
            author="Workflow Orchestrator Team",
            states={},
            inputs={},
            outputs={},
            resources={},
            dependencies=["slack-sdk", "aiofiles"]
        )
        super().__init__(manifest)
    
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register slack states."""
        return {
            "send_message": SendMessageState,
            "read_messages": ReadMessagesState,
            "upload_file": UploadFileState,
            "create_channel": CreateChannelState
        }
