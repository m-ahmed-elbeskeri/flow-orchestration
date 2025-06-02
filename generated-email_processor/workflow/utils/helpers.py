"""
Helper utilities for email_processor
"""
import os
from pathlib import Path
from typing import Any, Dict

def load_config() -> Dict[str, Any]:
    """Load configuration from environment and config files."""
    config: Dict[str, Any] = {}
    # Load from environment variables (with defaults from spec)
    for key, default in {
        "LOG_LEVEL": "INFO",
        "EMAIL_CHECK_QUERY": "is:unread label:urgent",
    }.items():
        config[key] = os.getenv(key, default)
    return config

def get_project_root() -> Path:
    """Get project root directory."""
    # Assumes this file lives at <project_root>/utils/helpers.py
    return Path(__file__).parent.parent
