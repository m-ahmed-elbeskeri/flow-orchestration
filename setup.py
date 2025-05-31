# setup.py
"""Setup script for Workflow Orchestrator."""

from setuptools import setup, find_packages

setup(
    name="workflow-orchestrator",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0",
        "pyyaml>=6.0",
        "aiohttp>=3.8",
        "pydantic>=2.0",
    ],
    extras_require={
        "enterprise": [
            "openai>=1.0",  # For OpenRouter compatibility
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "flake8>=6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "workflow=cli.main:cli",
            "wf=cli.main:cli",  # Short alias
        ],
    },
    python_requires=">=3.8",
)