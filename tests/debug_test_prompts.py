#!/usr/bin/env python3
"""Test script to show LLM prompts with detailed debugging."""

import sys
import os
from pathlib import Path

# Debug: Show current working directory and script location
print("🔍 DEBUG INFO:")
print(f"   Current working directory: {os.getcwd()}")
print(f"   Script file location: {__file__}")
print(f"   Script parent directory: {Path(__file__).parent}")

# Determine project root (go up one level from tests directory)
script_dir = Path(__file__).parent
if script_dir.name == "tests":
    project_root = script_dir.parent
    print(f"   Detected tests directory, using parent: {project_root}")
else:
    project_root = script_dir
    print(f"   Using script directory as root: {project_root}")

print(f"   Final project root: {project_root}")
print(f"   Project root exists: {project_root.exists()}")

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"   Added to Python path: {project_root}")
else:
    print(f"   Already in Python path: {project_root}")

print(f"   Python path (first 3): {sys.path[:3]}")

# Check if enterprise directory exists
enterprise_dir = project_root / "enterprise"
enterprise_ai_dir = enterprise_dir / "ai"

print(f"\n📁 DIRECTORY CHECK:")
print(f"   enterprise dir: {enterprise_dir} -> {enterprise_dir.exists()}")
print(f"   enterprise/ai dir: {enterprise_ai_dir} -> {enterprise_ai_dir.exists()}")

# Check for required files
required_files = [
    enterprise_dir / "__init__.py",
    enterprise_ai_dir / "__init__.py",
    enterprise_ai_dir / "plugin_discovery.py",
    enterprise_ai_dir / "flow_copilot.py",
    enterprise_ai_dir / "openrouter_api.py"
]

print(f"\n📄 FILE CHECK:")
for file_path in required_files:
    exists = file_path.exists()
    print(f"   {file_path.name}: {file_path} -> {'✅' if exists else '❌'} {exists}")

# Create missing __init__.py files if needed
print(f"\n🔧 CREATING MISSING FILES:")
init_files = [
    enterprise_dir / "__init__.py",
    enterprise_ai_dir / "__init__.py"
]

for init_file in init_files:
    if not init_file.exists() and init_file.parent.exists():
        try:
            init_file.write_text('"""Package initialization."""\n')
            print(f"   Created: {init_file}")
        except Exception as e:
            print(f"   Failed to create {init_file}: {e}")
    elif init_file.exists():
        print(f"   Already exists: {init_file}")
    else:
        print(f"   Cannot create (parent missing): {init_file}")

# Try importing with detailed error reporting
print(f"\n🐍 IMPORT TESTING:")

# Test individual imports
modules_to_test = [
    "enterprise",
    "enterprise.ai", 
    "enterprise.ai.plugin_discovery",
    "enterprise.ai.flow_copilot",
    "enterprise.ai.openrouter_api"
]

for module_name in modules_to_test:
    try:
        module = __import__(module_name)
        print(f"   ✅ {module_name}: {module}")
    except ImportError as e:
        print(f"   ❌ {module_name}: {e}")
    except Exception as e:
        print(f"   ⚠️  {module_name}: {e}")

# Try importing the classes
print(f"\n🎯 CLASS IMPORT TESTING:")
ENTERPRISE_AVAILABLE = False

try:
    from enterprise.ai.plugin_discovery import PluginDiscovery
    print(f"   ✅ PluginDiscovery imported successfully")
    
    try:
        from enterprise.ai.flow_copilot import FlowCopilot  
        print(f"   ✅ FlowCopilot imported successfully")
        ENTERPRISE_AVAILABLE = True
    except ImportError as e:
        print(f"   ❌ FlowCopilot import failed: {e}")
        
except ImportError as e:
    print(f"   ❌ PluginDiscovery import failed: {e}")

print(f"\n🎉 RESULT: ENTERPRISE_AVAILABLE = {ENTERPRISE_AVAILABLE}")

# If still not available, show troubleshooting steps
if not ENTERPRISE_AVAILABLE:
    print(f"\n🛠️  TROUBLESHOOTING STEPS:")
    print(f"   1. Make sure you're in the correct directory:")
    print(f"      Current: {os.getcwd()}")
    print(f"      Should have: enterprise/ai/*.py files")
    
    print(f"\n   2. Check file structure:")
    print(f"      Puffinflow/")
    print(f"      ├── enterprise/")
    print(f"      │   ├── __init__.py")
    print(f"      │   └── ai/")
    print(f"      │       ├── __init__.py")
    print(f"      │       ├── plugin_discovery.py")
    print(f"      │       ├── flow_copilot.py")
    print(f"      │       └── openrouter_api.py")
    print(f"      └── tests/")
    print(f"          └── debug_test_prompts.py")
    
    print(f"\n   3. Create missing files:")
    if not (enterprise_dir / "__init__.py").exists():
        print(f"      echo. > enterprise\\__init__.py")
    if not (enterprise_ai_dir / "__init__.py").exists():
        print(f"      echo. > enterprise\\ai\\__init__.py")
    
    print(f"\n   4. Run from project root (not tests directory):")
    print(f"      cd {project_root}")
    print(f"      python tests\\debug_test_prompts.py")

# Continue with original functionality if available
def show_plugin_capabilities():
    """Show the plugin capabilities prompt."""
    if not ENTERPRISE_AVAILABLE:
        print("❌ Enterprise features not available")
        return
    
    print("="*80)
    print("🔌 PLUGIN CAPABILITIES PROMPT")
    print("="*80)
    
    try:
        discovery = PluginDiscovery()
        capabilities = discovery.get_plugin_capabilities_prompt()
        
        if capabilities.strip():
            print(capabilities)
        else:
            print("No plugins found. Here's what the prompt would look like with plugins:")
            print("""
Plugin: example_plugin
  Description: An example plugin for demonstration
  Version: 1.0.0
  States:
    - example_plugin.send_email: Send an email notification
      Inputs:
        - recipient (string, required): Email recipient
        - subject (string, required): Email subject
        - body (string, optional): Email body content
      Outputs:
        - message_id (string): Unique message identifier
        - status (string): Delivery status
      Examples:
        - Send welcome email to new users
        - Send daily report notifications
""")
    except Exception as e:
        print(f"❌ Error getting plugin capabilities: {e}")


def show_analysis_prompt(request: str):
    """Show the analysis prompt for a given request."""
    if not ENTERPRISE_AVAILABLE:
        print("❌ Enterprise features not available")
        return
    
    print("="*80)
    print("🔍 ANALYSIS PROMPT")
    print("="*80)
    print(f"User Request: {request}")
    print("-"*80)
    
    try:
        # Get plugin capabilities
        discovery = PluginDiscovery()
        plugin_capabilities = discovery.get_plugin_capabilities_prompt()
        
        # Build the analysis system prompt
        system_prompt = f"""You are an AI assistant analyzing requests for the Workflow Orchestrator system.

AVAILABLE PLUGINS AND CAPABILITIES:
{plugin_capabilities if plugin_capabilities.strip() else "No plugins currently available"}

WORKFLOW FEATURES:
- State-based execution with dependencies
- Priority-based scheduling
- Resource management and quotas
- Retry policies with exponential backoff
- Checkpointing and resume capabilities
- Parallel and sequential execution
- Conditional branching
- Event-driven triggers

VARIABLE SYSTEM:
1. Environment variables: Accessed via context.get_secret() or context.get_constant()
2. Local variables: Accessed via context.get_variable() and context.set_variable()
3. State outputs: Referenced as state_id.output_name
4. Typed variables: Type-safe variables with context.get_typed_variable()

TASK: Analyze the request and identify missing information needed to generate a complete workflow.

Return a JSON object with these fields:
{{
    "clear_enough": boolean,
    "missing_information": ["list of missing details"],
    "clarification_questions": ["specific questions to ask"],
    "suggested_flow_description": "brief description",
    "suggested_variables": {{
        "env": ["API_KEY", "etc"],
        "local": ["counter", "results", "etc"],
        "constants": ["MAX_RETRIES", "etc"]
    }},
    "suggested_plugins": ["plugin names that might be useful"],
    "complexity": "simple|moderate|complex"
}}"""
        
        print("SYSTEM PROMPT:")
        print(system_prompt)
        print("\n" + "-"*80)
        print("USER PROMPT:")
        print(f"Analyze this workflow request: {request}")
        
    except Exception as e:
        print(f"❌ Error generating analysis prompt: {e}")


def main():
    """Run prompt tests."""
    print("="*80)
    print("🧪 LLM PROMPT TESTING TOOL (WITH DEBUG)")
    print("="*80)
    
    if ENTERPRISE_AVAILABLE:
        print("✅ Enterprise features available! Running tests...")
        show_plugin_capabilities()
        
        # Test with a simple request
        test_request = "send an email when a file is uploaded"
        show_analysis_prompt(test_request)
        
    else:
        print("❌ Enterprise features not available")
        print("See debug information above to fix the issue")


if __name__ == "__main__":
    main()