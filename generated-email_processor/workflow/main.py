"""
Process urgent emails and send notifications

Generated workflow from YAML definition.
Author: Team Workflow <team@example.com>
Version: 1.0.0
Generated: 2025-05-31T14:47:21.199647
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from core.agent.base import Agent, RetryPolicy
from core.agent.context import Context
from core.agent.state import StateResult
from core.resources.requirements import ResourceRequirements

# Import available integrations
from plugins.slack import SlackPlugin
from plugins.gmail import GmailPlugin

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


        self.plugins['slack'] = SlackPlugin()



        self.plugins['gmail'] = GmailPlugin()


        
        # Setup workflow states
        self._setup_states()
    
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
    
    async def _start_handler(self, context: Context) -> StateResult:
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
        # Plugin state
        state_func = self._create_plugin_state_handler(
            'gmail', 
            'read_emails',
            {"query": "is:unread label:urgent", "max_results": 10, "mark_as_read": False}
        )
        
        self.agent.add_state(
            "fetch_emails",
            state_func,
            dependencies={
                "start": "required",
            },
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
            dependencies={
                "fetch_emails": "required",
            },
            resources=ResourceRequirements(
                cpu_units=0.1,
                memory_mb=100,
                network_weight=1.0,
                priority=1,
            ),
            max_retries=3
        )
    
    async def _check_emails_handler(self, context: Context) -> StateResult:
        """Handler for check_emails state."""
        logger.info("Executing state: check_emails")
        
        try:
            # Conditional state - branching logic
            # Evaluate condition
            condition = """len(context.get_state("fetched_emails", [])) > 0"""
            # Safely evaluate the condition
            fetched_emails = context.get_state("fetched_emails", [])
            condition_result = eval(condition, {"context": context, "len": len, "fetched_emails": fetched_emails})
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
        # Plugin state
        state_func = self._create_plugin_state_handler(
            'slack', 
            'send_message',
            {"channel": "#alerts", "message": "Found {{ fetched_emails|length }} urgent emails"}
        )
        
        self.agent.add_state(
            "send_notification",
            state_func,
            dependencies={
                "check_emails": "required",
            },
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
    
    async def _error_handler_handler(self, context: Context) -> StateResult:
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
    
    async def _end_handler(self, context: Context) -> StateResult:
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
    
    def _create_plugin_state_handler(self, plugin_name: str, action_name: str, config: Dict[str, Any]):
        """Create a handler for plugin states."""
        async def plugin_handler(context: Context) -> StateResult:
            logger.info(f"Executing plugin state: {plugin_name}.{action_name}")
            
            try:
                # Get plugin
                plugin = self.plugins.get(plugin_name)
                if not plugin:
                    raise ValueError(f"Plugin '{plugin_name}' not available")
                
                # Resolve any template variables in config
                resolved_config = {}
                for key, value in config.items():
                    if isinstance(value, str) and '{' + '{' in value and '}' + '}' in value:
                        # Simple template resolution for common variables
                        template_str = value
                        
                        # Extract variable names from template
                        import re
                        vars_in_template = re.findall(r'\{\{\s*(\w+)(?:\|(\w+))?\s*\}\}', template_str)
                        
                        for var_name, filter_name in vars_in_template:
                            var_value = context.get_state(var_name, [])
                            
                            # Apply filter if specified
                            if filter_name == 'length':
                                replacement = str(len(var_value))
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
                
                # Get state function from plugin
                if hasattr(plugin, 'get_state_function'):
                    # Plugin uses get_state_function method
                    state_func = plugin.get_state_function(action_name, resolved_config)
                    result = await state_func(context)
                else:
                    # Plugin uses state classes directly
                    state_cls = plugin.register_states().get(action_name)
                    if not state_cls:
                        raise ValueError(f"Action '{action_name}' not found in plugin '{plugin_name}'")
                    
                    state_instance = state_cls(resolved_config)
                    result = await state_instance.execute(context)
                
                logger.info(f"Plugin state {plugin_name}.{action_name} completed")
                return result
                
            except Exception as e:
                logger.error(f"Error in plugin state {plugin_name}.{action_name}: {str(e)}")
                context.set_state("last_error", str(e))
                raise
        
        return plugin_handler
    
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


# Convenience function for external use
async def run_workflow(config: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to run the workflow."""
    workflow = EmailProcessorWorkflow(config)
    await workflow.run()


# Entry point for direct execution
async def main():
    """Main entry point."""
    import os
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_file = Path(__file__).parent.parent / "config" / "config.yaml"
    config = {}
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with environment variables
    for var_name, var_value in {
        "LOG_LEVEL": "INFO",
        "EMAIL_CHECK_QUERY": "is:unread label:urgent",
    }.items():
        env_value = os.getenv(var_name, var_value)
        if env_value != var_value:
            config.setdefault('environment', {})[var_name] = env_value
    
    # Run the workflow
    await run_workflow(config)


if __name__ == "__main__":
    asyncio.run(main())