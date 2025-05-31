#!/usr/bin/env python3
"""Test script to show LLM prompts without making API calls."""

import sys
from pathlib import Path

# Add project root to path
# Fix: Detect if we're in tests directory and go up to project root
script_dir = Path(__file__).parent
if script_dir.name == "tests":
    project_root = script_dir.parent  # Go up one level
else:
    project_root = script_dir
sys.path.insert(0, str(project_root))

try:
    from enterprise.ai.plugin_discovery import PluginDiscovery
    from enterprise.ai.flow_copilot import FlowCopilot
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False


def show_plugin_capabilities():
    """Show the plugin capabilities prompt."""
    if not ENTERPRISE_AVAILABLE:
        print("❌ Enterprise features not available")
        return
    
    print("="*80)
    print("🔌 PLUGIN CAPABILITIES PROMPT")
    print("="*80)
    
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


def show_generation_prompt(request: str, clarifications: dict = None):
    """Show the workflow generation prompt."""
    if not ENTERPRISE_AVAILABLE:
        print("❌ Enterprise features not available")
        return
    
    print("="*80)
    print("🔨 WORKFLOW GENERATION PROMPT")
    print("="*80)
    print(f"User Request: {request}")
    if clarifications:
        print(f"Clarifications: {clarifications}")
    print("-"*80)
    
    # Get plugin capabilities
    discovery = PluginDiscovery()
    plugin_capabilities = discovery.get_plugin_capabilities_prompt()
    
    # Build context from clarifications
    context_text = f"Original request: {request}"
    if clarifications:
        context_text += "\n\nClarifications:"
        for key, value in clarifications.items():
            context_text += f"\n- Q: {key}"
            context_text += f"\n  A: {value}"
    
    system_prompt = f"""You are an expert at creating workflow definitions for the Workflow Orchestrator system.

AVAILABLE PLUGINS:
{plugin_capabilities if plugin_capabilities.strip() else "No plugins currently available"}

WORKFLOW SCHEMA:
```yaml
name: workflow_name
description: Brief description
version: 1.0.0
triggers:
  - type: manual|schedule|event
    config: {{}}
states:
  state_id:
    plugin: plugin_name
    action: action_name
    inputs:
      param: value_or_reference
    outputs:
      - name: output_name
        type: string|number|object
    dependencies:
      other_state_id:
        type: required|optional|parallel
        lifecycle: once|always|session
    resources:
      cpu: 1
      memory: 512
      priority: normal|high|critical
    retry:
      max_attempts: 3
      backoff: exponential
```

IMPORTANT RULES:
- Use state_id.output_name to reference outputs from other states
- Use context methods for variables:
  - context.get_secret("KEY") for secrets
  - context.get_constant("NAME") for constants
  - context.get_variable("name") for flow variables
- States are executed based on dependencies and priority
- Each state must have a unique ID
- Plugin actions are referenced as plugin.action

Generate a complete workflow definition in YAML format."""
    
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\n" + "-"*80)
    print("USER PROMPT:")
    print(context_text)


def main():
    """Run prompt tests."""
    print("🧪 LLM Prompt Testing Tool")
    print("This shows exactly what prompts would be sent to the AI")
    
    if not ENTERPRISE_AVAILABLE:
        print("\n❌ Enterprise AI features not available")
        print("Install enterprise dependencies to see actual prompts")
        return
    
    # Test requests
    test_requests = [
        "send an email when a file is uploaded",
        "process a CSV file and generate a summary report",
        "backup database every night at 2 AM",
        "monitor API health and alert if down for 5 minutes"
    ]
    
    # Show plugin capabilities first
    show_plugin_capabilities()
    
    # Test each request
    for i, request in enumerate(test_requests, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST CASE {i}: {request}")
        print('#'*80)
        
        # Show analysis prompt
        show_analysis_prompt(request)
        
        print(f"\n{'='*40}")
        print("⬇️  WOULD ANALYZE AND GET RESPONSE  ⬇️")
        print('='*40)
        
        # Show generation prompt with sample clarifications
        sample_clarifications = {
            "What email service?": "Gmail SMTP",
            "What file types?": "PDF and Excel files",
            "Where are files uploaded?": "/uploads folder"
        } if "email" in request else None
        
        show_generation_prompt(request, sample_clarifications)
    
    print(f"\n\n{'#'*80}")
    print("🏁 PROMPT TESTING COMPLETE")
    print("💡 Use these prompts to test with ChatGPT/Claude directly")
    print('#'*80)


def interactive_test():
    """Interactive prompt testing."""
    if not ENTERPRISE_AVAILABLE:
        print("❌ Enterprise features not available")
        return
    
    print("🔍 Interactive Prompt Testing")
    print("Enter your workflow request (or 'quit' to exit):")
    
    while True:
        request = input("\n> ").strip()
        if request.lower() in ['quit', 'exit', 'q']:
            break
        
        if not request:
            continue
        
        print(f"\n📝 Testing prompt for: {request}")
        show_analysis_prompt(request)
        
        print(f"\n{'='*40}")
        print("Continue with generation prompt? (y/n)")
        if input("> ").lower().startswith('y'):
            show_generation_prompt(request)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM prompts")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode")
    parser.add_argument("--request", "-r", type=str,
                       help="Test specific request")
    parser.add_argument("--plugins-only", action="store_true",
                       help="Show only plugin capabilities")
    
    args = parser.parse_args()
    
    if args.plugins_only:
        show_plugin_capabilities()
    elif args.request:
        show_analysis_prompt(args.request)
        print("\n" + "="*40)
        show_generation_prompt(args.request)
    elif args.interactive:
        interactive_test()
    else:
        main()