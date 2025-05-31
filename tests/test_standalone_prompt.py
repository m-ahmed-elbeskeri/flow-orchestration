#!/usr/bin/env python3
"""Standalone prompt testing - no dependencies required."""

def get_sample_plugin_capabilities():
    """Sample plugin capabilities that would be discovered."""
    return """Plugin: echo
  Description: A simple plugin that echoes a message.
  Version: 0.1.0
  States:
    - echo.execute: Execute echo command
      Inputs:
        - message (string, required): The message to echo
      Outputs:
        - echo_message (string): The echoed message

Plugin: file_watcher
  Description: Monitor file system changes and trigger actions
  Version: 1.0.0
  States:
    - file_watcher.monitor: Monitor a directory for file changes
      Inputs:
        - path (string, required): Directory path to monitor
        - pattern (string, optional, default: *): File pattern to match
        - recursive (boolean, optional, default: false): Monitor subdirectories
      Outputs:
        - file_path (string): Path of the changed file
        - event_type (string): Type of change (created, modified, deleted)
      Examples:
        - Monitor uploads folder for new files
        - Watch config directory for changes

Plugin: email
  Description: Send email notifications and messages
  Version: 1.2.0
  States:
    - email.send: Send an email message
      Inputs:
        - recipient (string, required): Email recipient address
        - subject (string, required): Email subject line
        - body (string, required): Email body content
        - sender (string, optional): Sender email address
        - smtp_server (string, optional): SMTP server hostname
      Outputs:
        - message_id (string): Unique message identifier
        - status (string): Delivery status
      Examples:
        - Send welcome email to new users
        - Send daily report notifications
        - Alert administrators of system issues

Plugin: csv_processor
  Description: Process and analyze CSV files
  Version: 2.0.0
  States:
    - csv_processor.read: Read and parse CSV file
      Inputs:
        - file_path (string, required): Path to CSV file
        - delimiter (string, optional, default: ,): Field delimiter
        - header (boolean, optional, default: true): First row contains headers
      Outputs:
        - data (object): Parsed CSV data
        - row_count (number): Number of rows processed
        - columns (array): List of column names
    - csv_processor.filter: Filter CSV data based on conditions
      Inputs:
        - data (object, required): CSV data from read operation
        - conditions (object, required): Filter conditions
      Outputs:
        - filtered_data (object): Filtered dataset
        - filtered_count (number): Number of rows after filtering
    - csv_processor.aggregate: Generate summary statistics
      Inputs:
        - data (object, required): CSV data to analyze
        - group_by (array, optional): Columns to group by
        - metrics (array, required): Metrics to calculate
      Outputs:
        - summary (object): Aggregated results
        - charts (array): Generated chart data

Plugin: database
  Description: Database operations and backup utilities
  Version: 1.5.0
  States:
    - database.backup: Create database backup
      Inputs:
        - connection_string (string, required): Database connection
        - backup_path (string, required): Where to store backup
        - compress (boolean, optional, default: true): Compress backup file
      Outputs:
        - backup_file (string): Path to created backup file
        - backup_size (number): Size of backup in bytes
    - database.query: Execute database query
      Inputs:
        - connection_string (string, required): Database connection
        - query (string, required): SQL query to execute
        - parameters (object, optional): Query parameters
      Outputs:
        - results (array): Query results
        - row_count (number): Number of rows returned

Plugin: http_monitor
  Description: Monitor HTTP endpoints and APIs
  Version: 1.0.0
  States:
    - http_monitor.check: Check HTTP endpoint status
      Inputs:
        - url (string, required): URL to monitor
        - timeout (number, optional, default: 30): Request timeout in seconds
        - expected_status (number, optional, default: 200): Expected HTTP status
        - check_interval (number, optional, default: 60): Check interval in seconds
      Outputs:
        - status_code (number): HTTP response status
        - response_time (number): Response time in milliseconds
        - is_healthy (boolean): Whether endpoint is healthy
      Examples:
        - Monitor API health
        - Check website uptime
        - Verify service availability"""


def show_analysis_prompt(request: str):
    """Show the analysis prompt for a given request."""
    plugin_capabilities = get_sample_plugin_capabilities()
    
    system_prompt = f"""You are an AI assistant analyzing requests for the Workflow Orchestrator system.

AVAILABLE PLUGINS AND CAPABILITIES:
{plugin_capabilities}

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
    
    print("="*80)
    print("🔍 ANALYSIS PROMPT")
    print("="*80)
    print(f"Request: {request}")
    print("-"*80)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\n" + "-"*80)
    print("USER PROMPT:")
    print(f"Analyze this workflow request: {request}")


def show_generation_prompt(request: str, clarifications: dict = None):
    """Show the workflow generation prompt."""
    plugin_capabilities = get_sample_plugin_capabilities()
    
    # Build context from clarifications
    context_text = f"Original request: {request}"
    if clarifications:
        context_text += "\n\nClarifications:"
        for question, answer in clarifications.items():
            context_text += f"\n- {question}"
            context_text += f"\n  Answer: {answer}"
    
    system_prompt = f"""You are an expert at creating workflow definitions for the Workflow Orchestrator system.

AVAILABLE PLUGINS:
{plugin_capabilities}

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
    
    print("="*80)
    print("🔨 WORKFLOW GENERATION PROMPT")
    print("="*80)
    print(f"Request: {request}")
    if clarifications:
        print(f"With clarifications: {list(clarifications.keys())}")
    print("-"*80)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\n" + "-"*80)
    print("USER PROMPT:")
    print(context_text)


def test_request(request: str, clarifications: dict = None):
    """Test a single request."""
    print(f"\n{'#'*80}")
    print(f"TESTING: {request}")
    print('#'*80)
    
    # Show analysis prompt
    show_analysis_prompt(request)
    
    print(f"\n{'='*40}")
    print("⬇️  AFTER ANALYSIS, WOULD GENERATE WORKFLOW  ⬇️")
    print('='*40)
    
    # Show generation prompt
    show_generation_prompt(request, clarifications)


def main():
    """Run prompt tests."""
    print("🧪 LLM Prompt Testing Tool (Standalone)")
    print("This shows exactly what prompts would be sent to the AI")
    print("No dependencies required - using sample plugin data")
    
    # Show available plugins first
    print("\n" + "="*80)
    print("🔌 AVAILABLE PLUGINS (Sample Data)")
    print("="*80)
    print(get_sample_plugin_capabilities())
    
    # Test requests with different complexity levels
    test_cases = [
        {
            "request": "send an email when a file is uploaded",
            "clarifications": {
                "What email service should be used?": "Gmail SMTP",
                "What file types should trigger the email?": "PDF and Excel files only",
                "Where are files uploaded to?": "/uploads/documents folder"
            }
        },
        {
            "request": "process a CSV file and generate a summary report",
            "clarifications": {
                "What kind of summary is needed?": "Count of rows, column statistics, and charts",
                "Where should the report be saved?": "/reports folder as PDF",
                "Should the original CSV be archived?": "Yes, move to /processed folder"
            }
        },
        {
            "request": "backup database every night at 2 AM",
            "clarifications": {
                "Which database type?": "PostgreSQL",
                "Where should backups be stored?": "AWS S3 bucket",
                "How long should backups be retained?": "30 days"
            }
        },
        {
            "request": "monitor API health and alert if down for 5 minutes",
            "clarifications": {
                "Which API endpoints to monitor?": "https://api.example.com/health",
                "How should alerts be sent?": "Email to admin team",
                "What recovery actions should be taken?": "Restart service automatically"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        test_request(test_case["request"], test_case["clarifications"])
    
    print(f"\n{'#'*80}")
    print("🏁 PROMPT TESTING COMPLETE")
    print("💡 Copy these prompts to test with ChatGPT, Claude, or other LLMs")
    print("📝 Use these to refine prompt engineering before making API calls")
    print('#'*80)


def interactive_mode():
    """Interactive prompt testing."""
    print("🔍 Interactive Prompt Testing")
    print("Enter workflow requests to see the prompts that would be generated")
    print("Commands: 'plugins' to see available plugins, 'quit' to exit")
    
    while True:
        print("\n" + "-"*40)
        request = input("Enter workflow request: ").strip()
        
        if request.lower() in ['quit', 'exit', 'q']:
            break
        elif request.lower() == 'plugins':
            print("\n🔌 Available Plugins:")
            print(get_sample_plugin_capabilities())
            continue
        elif not request:
            continue
        
        # Ask for clarifications
        print("\nAdd clarifications? (y/n): ", end="")
        if input().lower().startswith('y'):
            clarifications = {}
            print("Enter clarifications (empty line to finish):")
            while True:
                question = input("Question: ").strip()
                if not question:
                    break
                answer = input("Answer: ").strip()
                if answer:
                    clarifications[question] = answer
        else:
            clarifications = None
        
        test_request(request, clarifications)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_mode()
        elif sys.argv[1] == "--plugins":
            print("🔌 Available Plugins:")
            print(get_sample_plugin_capabilities())
        else:
            # Treat arguments as a request
            request = " ".join(sys.argv[1:])
            test_request(request)
    else:
        main()