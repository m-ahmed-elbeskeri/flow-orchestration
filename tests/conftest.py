"""
Pytest configuration and fixtures for the flow-orchestration project.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pytest and common testing utilities
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
