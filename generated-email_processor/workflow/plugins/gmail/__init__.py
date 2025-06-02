"""
Gmail integration for sending and reading emails

Standalone plugin implementation.
Original: gmail v1.0.0
"""

from pathlib import Path
from typing import Dict, Type

from ..base import Plugin, PluginState, PluginManifest

# Import state classes
from .states import (
    SendEmailState,
    ReadEmailsState,
    SearchEmailsState,
)


class GmailPlugin(Plugin):
    """Gmail integration for sending and reading emails"""
    
    def __init__(self):
        # Create manifest
        manifest = PluginManifest(
            name="gmail",
            version="1.0.0",
            description="Gmail integration for sending and reading emails",
            author="Workflow Orchestrator Team",
            states={},
            inputs={},
            outputs={},
            resources={},
            dependencies=["google-auth", "google-auth-oauthlib", "google-auth-httplib2", "google-api-python-client"]
        )
        super().__init__(manifest)
    
    def register_states(self) -> Dict[str, Type[PluginState]]:
        """Register gmail states."""
        return {
            "send_email": SendEmailState,
            "read_emails": ReadEmailsState,
            "search_emails": SearchEmailsState
        }
