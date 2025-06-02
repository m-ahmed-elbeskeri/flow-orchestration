"""
Process urgent emails and send notifications

Generated standalone workflow from YAML definition.
Author: Team Workflow <team@example.com>
Version: 1.0.0
Generated: 2025-06-01T10:59:49.537062

This is a standalone workflow that includes all necessary plugin code.
No external dependencies on the original project structure.
"""

import asyncio
import logging
import time
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Import standalone agent system
from .agent.agent import Agent, RetryPolicy, ResourceRequirements

# Import plugin base classes
from .plugins.base import Context

# Import local plugins
from .plugins.gmail import GmailPlugin
from .plugins.slack import SlackPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailProcessorWorkflow:
    """Process urgent emails and send notifications"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Load environment variables
        self._load_environment()
        
        # Initialize agent
        self.agent = Agent(
            name="email_processor",
            max_concurrent=3,
            state_timeout=60.0,
            retry_policy=RetryPolicy(
                max_retries=3,
                initial_delay=2.0,
                exponential_base=2.0,
                jitter=True
            )
        )
        
        # Initialize plugins
        self.plugins = {}
        self.plugins['gmail'] = GmailPlugin()
        self.plugins['slack'] = SlackPlugin()
        
        # Setup workflow states
        self._setup_states()
    
    def _load_environment(self):
        """Load environment variables and secrets."""
        # Load from .env file if it exists
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                logger.info("Loaded environment from .env file")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file")
        
        # Set secrets in agent context
        secret_value = os.getenv("GMAIL_CREDENTIALS")
        if secret_value:
            self.agent.context.set_secret("gmail_credentials", secret_value)
        secret_value = os.getenv("SLACK_TOKEN")
        if secret_value:
            self.agent.context.set_secret("slack_token", secret_value)
    
    def _setup_states(self):
        """Setup all workflow states."""
        self._add_state_start()
        self._add_state_fetch_emails()
        self._add_state_check_emails()
        self._add_state_send_notification()
        self._add_state_error_handler()
        self._add_state_end()
    
    def _add_state_start(self):
        """Add start state to workflow."""
        # Built-in state
        state_func = self._start_handler
        
        self.agent.add_state(
            "start",
            state_func,
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    async def _start_handler(self, context: Context) -> Any:
        """Handler for start state."""
        logger.info("Executing state: start")
        
        try:
            # Start state - workflow initialization
            context.set_constant("workflow_start_time", time.time())
            context.set_constant("workflow_name", "email_processor")
            context.set_constant("workflow_version", "1.0.0")
            logger.info("Starting workflow: email_processor")
            # Handle transitions
            return "fetch_emails"
            
        except Exception as e:
            logger.error(f"Error in state start: {str(e)}")
            context.set_state("last_error", str(e))
            
            # Handle error transitions
            
            raise
    
    def _add_state_fetch_emails(self):
        """Add fetch_emails state to workflow."""
        # Plugin state with transition handling
        state_func = self._create_plugin_state_wrapper(
            'gmail', 
            'read_emails',
            {"query": "is:unread label:urgent", "max_results": 10, "mark_as_read": False},
            {
                "on_success": "check_emails",
                "on_failure": "error_handler",
            }
        )
        
        self.agent.add_state(
            "fetch_emails",
            state_func,
            dependencies=[
                "start",
            ],
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=2.0,
                priority=1,
            ),
            max_retries=3
        )
    
    def _add_state_check_emails(self):
        """Add check_emails state to workflow."""
        # Built-in state
        state_func = self._check_emails_handler
        
        self.agent.add_state(
            "check_emails",
            state_func,
            dependencies=[
                "fetch_emails",
            ],
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    async def _check_emails_handler(self, context: Context) -> Any:
        """Handler for check_emails state."""
        logger.info("Executing state: check_emails")
        
        try:
            # Conditional state - branching logic
            # Evaluate condition
            condition = """len(context.get_state("fetched_emails", [])) > 0"""
            # Safely evaluate the condition
            fetched_emails = context.get_state("fetched_emails", [])
            condition_result = eval(condition, {
                "context": context, 
                "len": len, 
                "fetched_emails": fetched_emails,
                "__builtins__": {}  # Restrict built-ins for safety
            })
            logger.info(f"Condition evaluated to: {condition_result}")
            if condition_result:
                return "send_notification"
            else:
                return "end"
            # Handle transitions
            
        except Exception as e:
            logger.error(f"Error in state check_emails: {str(e)}")
            context.set_state("last_error", str(e))
            
            # Handle error transitions
            
            raise
    
    def _add_state_send_notification(self):
        """Add send_notification state to workflow."""
        # Plugin state with transition handling
        state_func = self._create_plugin_state_wrapper(
            'slack', 
            'send_message',
            {"channel": "#alerts", "message": "Found {{ fetched_emails|length }} urgent emails"},
            {
                "on_success": "end",
                "on_failure": "error_handler",
            }
        )
        
        self.agent.add_state(
            "send_notification",
            state_func,
            dependencies=[
                "check_emails",
            ],
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    def _add_state_error_handler(self):
        """Add error_handler state to workflow."""
        # Built-in state
        state_func = self._error_handler_handler
        
        self.agent.add_state(
            "error_handler",
            state_func,
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    async def _error_handler_handler(self, context: Context) -> Any:
        """Handler for error_handler state."""
        logger.info("Executing state: error_handler")
        
        try:
            # Error handler state
            error_info = context.get_state("last_error")
            if error_info:
                logger.error(f"Handling error: {error_info}")
                context.set_state("error_notification_sent", True)
            # Handle transitions
            return "end"
            
        except Exception as e:
            logger.error(f"Error in state error_handler: {str(e)}")
            context.set_state("last_error", str(e))
            
            # Handle error transitions
            
            raise
    
    def _add_state_end(self):
        """Add end state to workflow."""
        # Built-in state
        state_func = self._end_handler
        
        self.agent.add_state(
            "end",
            state_func,
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    async def _end_handler(self, context: Context) -> Any:
        """Handler for end state."""
        logger.info("Executing state: end")
        
        try:
            # End state - workflow completion
            start_time = context.get_constant("workflow_start_time", time.time())
            duration = time.time() - start_time
            logger.info(f"Workflow email_processor completed in {duration:.2f} seconds")
            # Handle transitions
            return None
            
        except Exception as e:
            logger.error(f"Error in state end: {str(e)}")
            context.set_state("last_error", str(e))
            
            # Handle error transitions
            
            raise
    
    def _create_plugin_state_wrapper(self, plugin_name: str, action_name: str, config: Dict[str, Any], transitions: Dict[str, str]):
        """Create a wrapper for plugin states that handles transitions."""
        async def plugin_wrapper(context: Context) -> Any:
            logger.info(f"Executing plugin state: {plugin_name}.{action_name}")
            
            try:
                # Get plugin
                plugin = self.plugins.get(plugin_name)
                if not plugin:
                    raise ValueError(f"Plugin '{plugin_name}' not available")
                
                # Resolve any template variables in config
                resolved_config = self._resolve_config_templates(config, context)
                
                # Get state function from plugin
                state_func = plugin.get_state_function(action_name, resolved_config)
                result = await state_func(context)
                
                logger.info(f"Plugin state {plugin_name}.{action_name} completed")
                
                # Handle transitions based on success
                if "on_success" in transitions:
                    return transitions["on_success"]
                else:
                    return result
                
            except Exception as e:
                logger.error(f"Error in plugin state {plugin_name}.{action_name}: {str(e)}")
                context.set_state("last_error", str(e))
                
                # Handle error transitions
                if "on_failure" in transitions:
                    return transitions["on_failure"]
                else:
                    raise
        
        return plugin_wrapper
    
    def _create_plugin_state_handler(self, plugin_name: str, action_name: str, config: Dict[str, Any]):
        """Create a handler for plugin states (legacy method for compatibility)."""
        return self._create_plugin_state_wrapper(plugin_name, action_name, config, {})
    
    def _resolve_config_templates(self, config: Dict[str, Any], context: Context) -> Dict[str, Any]:
        """Resolve template variables in configuration."""
        resolved_config = {}
        
        for key, value in config.items():
            if isinstance(value, str) and '{' + '{' in value and '}' + '}' in value:
                # Simple template resolution for common variables
                template_str = value
                
                # Extract variable names from template
                vars_in_template = re.findall(r'\{\{\s*(\w+)(?:\|(\w+))?\s*\}\}', template_str)
                
                for var_name, filter_name in vars_in_template:
                    var_value = context.get_state(var_name, [])
                    
                    # Apply filter if specified
                    if filter_name == 'length':
                        replacement = str(len(var_value))
                    elif filter_name == 'first':
                        replacement = str(var_value[0] if var_value else '')
                    elif filter_name == 'last':
                        replacement = str(var_value[-1] if var_value else '')
                    elif filter_name == 'join':
                        replacement = ', '.join(str(item) for item in var_value) if isinstance(var_value, list) else str(var_value)
                    else:
                        replacement = str(var_value)
                    
                    # Replace the template variable
                    template_pattern = r'\{\{\s*' + re.escape(var_name)
                    if filter_name:
                        template_pattern += r'\s*\|\s*' + re.escape(filter_name)
                    template_pattern += r'\s*\}\}'
                    
                    template_str = re.sub(template_pattern, replacement, template_str)
                
                resolved_config[key] = template_str
            else:
                resolved_config[key] = value
        
        return resolved_config
    
    async def run(self, timeout: Optional[float] = None) -> None:
        """Run the complete workflow."""
        logger.info(f"Starting workflow execution: email_processor")
        
        try:
            # Use configured timeout or default
            workflow_timeout = timeout or 600.0
            
            await self.agent.run(timeout=workflow_timeout)
            logger.info("Workflow execution completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    async def run_from_state(self, state_name: str) -> None:
        """Run workflow starting from a specific state."""
        logger.info(f"Starting workflow from state: {state_name}")
        
        try:
            await self.agent.run_state(state_name)
            logger.info(f"Workflow execution from {state_name} completed")
            
        except Exception as e:
            logger.error(f"Workflow execution from {state_name} failed: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "name": "email_processor",
            "version": "1.0.0",
            "agent_status": self.agent.status.value if hasattr(self.agent.status, 'value') else str(self.agent.status),
            "completed_states": list(self.agent.completed_states),
            "running_states": list(self.agent._running_states),
            "total_states": len(self.agent.states),
            "plugins_loaded": list(self.plugins.keys())
        }
    
    def get_context_data(self) -> Dict[str, Any]:
        """Get current context data for debugging."""
        return {
            "outputs": self.agent.context._outputs,
            "state": self.agent.context._state,
            "constants": self.agent.context._constants
        }
    
    def validate_workflow(self) -> Dict[str, Any]:
        """Validate workflow configuration and setup."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check plugins
        validation["info"]["plugins"] = {}
        for plugin_name, plugin in self.plugins.items():
            try:
                states = plugin.register_states()
                validation["info"]["plugins"][plugin_name] = {
                    "loaded": True,
                    "states": list(states.keys()),
                    "description": plugin.manifest.description
                }
            except Exception as e:
                validation["errors"].append(f"Plugin {plugin_name} failed to register states: {e}")
                validation["valid"] = False
        
        # Check state configuration
        validation["info"]["states"] = len(self.agent.states)
        validation["info"]["dependencies"] = self.agent.dependencies
        
        # Check for circular dependencies (basic check)
        try:
            self._check_circular_dependencies()
        except Exception as e:
            validation["errors"].append(f"Circular dependency detected: {e}")
            validation["valid"] = False
        
        return validation
    
    def _check_circular_dependencies(self):
        """Check for circular dependencies in the workflow."""
        def has_cycle(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in self.agent.dependencies.get(node, []):
                if not visited.get(neighbor, False):
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            
            rec_stack[node] = False
            return False
        
        visited = {}
        rec_stack = {}
        
        for state in self.agent.states:
            if not visited.get(state, False):
                if has_cycle(state, visited, rec_stack):
                    raise ValueError(f"Circular dependency detected involving state: {state}")


# Convenience function for external use
async def run_workflow(config: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to run the workflow."""
    workflow = EmailProcessorWorkflow(config)
    await workflow.run()


# Entry point for direct execution
async def main():
    """Main entry point."""
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_file = Path(__file__).parent.parent / "config" / "config.yaml"
    config = {}
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Override with environment variables
    env_vars = {
        "LOG_LEVEL": "INFO",
        "EMAIL_CHECK_QUERY": "is:unread label:urgent",
    }
    
    for var_name, default_value in env_vars.items():
        env_value = os.getenv(var_name, default_value)
        if env_value != default_value:
            config.setdefault('environment', {})[var_name] = env_value
    
    # Run the workflow
    try:
        await run_workflow(config)
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())