"""DAG builder and code generation from YAML."""

from typing import Dict, List, Optional, Any, Type, Union, Tuple
from pathlib import Path
import black
from datetime import datetime
import structlog

from core.dag.graph import DAG, DAGNode, DAGEdge
from core.dag.parser import YAMLParser, WorkflowDefinition, StateDefinition
from core.agent.base import Agent # Assuming Context is also in core.agent.base or imported where needed
# If Context is separate: from core.agent.context import Context
from plugins.base import Plugin, PluginState


logger = structlog.get_logger(__name__)

# Default Jinja2 template strings
DEFAULT_WORKFLOW_TEMPLATE_STR = """
# Auto-generated workflow: {{ workflow.name }}
# Version: {{ workflow.version }} - Generated {{ generated_at }}

import asyncio
from typing import Dict, Any, Optional

{# Imports managed by CodeGenerator._get_required_imports #}
{% for import_stmt in imports -%}
{{ import_stmt }}
{% endfor %}

class {{ workflow.name|title|replace('_', '') }}Workflow:
    \"\"\"{{ workflow.description or (workflow.name + " workflow") }}\"\"\"

    def __init__(self):
        self.agent = Agent("{{ workflow.name }}")
        self._setup_states()

    def _setup_states(self):
        \"\"\"Setup workflow states for the agent.\"\"\"
{%- for state_name, state in workflow.states.items() %}
        # State: {{ state_name }}
        self.agent.add_state(
            name="{{ state_name }}",
            func=self.{{ state_name }}{% if state.dependencies or state.resources or state.retries is not none %},{% endif %}
{%- if state.dependencies %}
            dependencies={
{%- for dep in state.dependencies %}
                "{{ dep }}": "required",
{%- endfor %}
            },{% endif %}
{%- if state.resources %}
            resources=ResourceRequirements(
                cpu_units={{ state.resources.cpu_units | default(0.1) }},
                memory_mb={{ state.resources.memory_mb | default(128) }}{% if state.resources.timeout is not none %},
                timeout={{ state.resources.timeout }}{% endif %}{% if state.resources.io_weight is not none %},
                io_weight={{ state.resources.io_weight }}{% endif %}{% if state.resources.network_weight is not none %},
                network_weight={{ state.resources.network_weight }}{% endif %}{% if state.resources.gpu_units is not none %},
                gpu_units={{ state.resources.gpu_units }}{% endif %}
            ),{% endif %}
{%- if state.retries is not none %}
            max_retries={{ state.retries }}{% endif %}
        )
{%- endfor %}

{%- for state_name, state in workflow.states.items() %}
    async def {{ state_name }}(self, context: Context):
        \"\"\"{{ state.metadata.get('description', state_name + ' state logic.') }}\"\"\"
        logger.info(f"Executing state: {context.current_node.id}", state_config=context.current_node.data.config)

{% if state.config.get('message') %}
        # Example action: printing a message from config
        print({{ state.config.message | tojson }})
{% else %}
        # TODO: Implement custom logic for state '{{ state_name }}' using the 'context' object.
        # Access state configuration via: context.current_node.data.config
        # Set output for subsequent states via: context.set_output("my_output", value)
        pass
{% endif %}

{%- if state.transitions %}
    {%- if state.transitions|length == 1 and not state.transitions[0].condition %}
        # Single, unconditional transition
        return "{{ state.transitions[0].target }}"
    {%- else %}
        # Conditional or multiple transitions
        # TODO: Implement logic to evaluate conditions and determine the next state.
        # Example:
        # if context.get_output("some_previous_output") == "value_a":
        #     return "target_a"
        # elif {{ state.transitions[0].condition or "True" }}: # Placeholder for actual condition
        #     return "{{ state.transitions[0].target }}"
        # Fallback or default transition if any:
        {% for transition in state.transitions %}
{%- if transition.condition %}
        # if context.evaluate_condition("{{ transition.condition }}"): # If you have a condition evaluator
        if {{ transition.condition }}: # Assumes '{{ transition.condition }}' is valid Python evaluable with context
            logger.debug(f"Condition '{{ transition.condition }}' met, transitioning to '{{ transition.target }}'")
            return "{{ transition.target }}"
{%- elif loop.index == state.transitions|length %} {# Last transition, could be default #}
        logger.debug(f"Default transition to '{{ transition.target }}'")
        return "{{ transition.target }}"
{%- endif %}
        {%- endfor %}
        logger.warning(f"No transition condition met in state '{{ state_name }}', ending path.")
        return None # Or raise error if no transition found and it's not an end state
    {%- endif %}
{%- else %}
        # No explicit transitions defined, this state is a terminal point for this path.
        logger.info(f"State '{{ state_name }}' is an end state for this execution path.")
        return None
{%- endif %}
{%- endfor %}

    async def run(self, initial_context_data: Optional[Dict[str, Any]] = None):
        \"\"\"Run the workflow. Initial context data can be provided.\"\"\"
        logger.info(f"Running workflow '{{ workflow.name }}'")
        await self.agent.run(initial_context_data=initial_context_data)
        logger.info(f"Workflow '{{ workflow.name }}' finished.")

async def main():
    \"\"\"Example of how to run the workflow.\"\"\"
    workflow = {{ workflow.name|title|replace('_', '') }}Workflow()
    await workflow.run() # Add initial_context_data={"key": "value"} if needed

if __name__ == "__main__":
    asyncio.run(main())
"""

DEFAULT_STATE_FUNCTION_TEMPLATE_STR = """
async def {{ state.name }}(context: Context):
    \"\"\"
    State: {{ state.name }}
    Type: {{ state.type }}
    Description: {{ state.metadata.get('description', 'No description provided for this state.') }}
    \"\"\"
    logger.info(f"Executing state function: {{ state.name }}", state_config=context.current_node.data.config)

{%- if state.config.get('plugin') %}
    # This state is configured to use a plugin: {{ state.config.plugin }}
    # Plugin logic is typically handled by the agent's core execution loop if integrated.
    # If you need to manually invoke a plugin (less common for agent-based systems):
    # plugin_name = "{{ state.config.plugin }}"
    # plugin_config = context.current_node.data.config # Or specific parts of it
    # result = await context.call_plugin(plugin_name, plugin_config) # Hypothetical plugin call
    logger.info(f"State '{{ state.name }}' uses plugin '{{ state.config.plugin }}'. Agent handles execution.")
{%- else %}
    # TODO: Implement custom logic for state '{{ state.name }}'.
    # Access config via: context.current_node.data.config
    # Set results via: context.set_output("output_key", value)
    print(f"Implement logic for state {{ state.name }}.")
    pass
{%- endif %}

{%- if state.transitions %}
    # Transitions are determined by returning the name of the next state.
    {%- if state.transitions|length == 1 and not state.transitions[0].condition %}
    return "{{ state.transitions[0].target }}"
    {%- else %}
    # TODO: Implement logic to choose the next state based on conditions.
    # Example:
    # if context.get_output("some_data") == "expected_value":
    #    return "next_state_A"
    # else:
    #    return "next_state_B"
    {% for transition in state.transitions %}
{%- if transition.condition %}
    # if context.evaluate_condition("{{ transition.condition }}"): # If you have a condition evaluator
    if {{ transition.condition }}: # Assumes '{{ transition.condition }}' is valid Python evaluable with context
        return "{{ transition.target }}"
{%- elif loop.index == state.transitions|length %}
    return "{{ transition.target }}" # Default or last transition
{%- endif %}
    {%- endfor %}
    return None # Fallback if no condition matched
    {%- endif %}
{%- else %}
    # No explicit transitions; this state is a terminal point for this execution path.
    return None
{%- endif %}
"""

DEFAULT_WORKFLOW_TEST_TEMPLATE_STR = """
import pytest
import asyncio
from pathlib import Path
import sys
from typing import Dict, Any

# Ensure the generated workflow module can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
# Adjust PROJECT_ROOT if the tests are generated deeper in the structure
PROJECT_ROOT = SCRIPT_DIR.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from {{ workflow.name }}_auto import {{ workflow.name|title|replace('_','') }}Workflow
from core.agent import Context # Uncomment if you need to create specific Context objects for tests
# from unittest.mock import patch, AsyncMock # For advanced mocking scenarios

@pytest.mark.asyncio
async def test_workflow_instantiation():
    \"\"\"Test that the {{ workflow.name }} workflow can be instantiated.\"\"\"
    try:
        workflow = {{ workflow.name|title|replace('_','') }}Workflow()
        assert workflow is not None, "Workflow object should not be None."
        assert hasattr(workflow, "run"), "Workflow instance should have a 'run' method."
    except Exception as e:
        pytest.fail(f"Workflow instantiation failed: {e}")

{% for test_case in test_cases %}
@pytest.mark.asyncio
async def {{ test_case.name }}():
    \"\"\"{{ test_case.description }}\"\"\"
    workflow = {{ workflow.name|title|replace('_','') }}Workflow()

    initial_context_data: Dict[str, Any] = {}
    # Example: Setup initial context based on test_case.setup
    # if "{{ test_case.setup.get('condition') }}": # This is illustrative
    #     initial_context_data["some_key_for_condition"] = True

    try:
        await workflow.run(initial_context_data=initial_context_data)
    except Exception as e:
        pytest.fail(f"Test '{{ test_case.name }}' failed during workflow execution: {e}")

    # TODO: Add specific assertions for this test case.
    # This might involve:
    # - Checking the final state of the agent (if exposed).
    # - Verifying outputs stored in the context (if the context object is accessible post-run).
    # - Capturing and inspecting logs or stdout.
    # - Mocking external dependencies and verifying their interactions.
    
    # Example assertion (requires agent to store history or final state):
    # assert workflow.agent.get_final_state() == "{{ test_case.get('expected_final_state', '') }}"
    
    # For the demo_workflow, if it prints "Hello, world!", you might capture stdout:
    # from io import StringIO
    # import sys
    # old_stdout = sys.stdout
    # sys.stdout = captured_output = StringIO()
    # await workflow.run()
    # sys.stdout = old_stdout
    # assert "Hello, world!" in captured_output.getvalue()

{% if not loop.last %}

{% endif %}
{% endfor %}

# Add more specific or advanced tests below if needed.
# Example using mocking:
# @pytest.mark.asyncio
# async def test_state_with_mocked_dependency():
#     with patch("path.to.dependency.method", return_value="mocked_value") as mock_method:
#         workflow = {{ workflow.name|title|replace('_','') }}Workflow()
#         await workflow.run()
#         mock_method.assert_called_once()
"""


class DAGBuilder:
    """Build DAG from workflow definition."""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.manifest.name] = plugin
    
    def build_from_definition(self, workflow: WorkflowDefinition) -> DAG[StateDefinition]:
        """Build DAG from workflow definition."""
        dag = DAG(workflow.name)
        
        # Add nodes
        for state_name, state in workflow.states.items():
            node = DAGNode(
                id=state_name,
                data=state,
                metadata={
                    "label": state.metadata.get("label", state_name),
                    "type": state.type,
                    "plugin": state.config.get("plugin")
                }
            )
            dag.add_node(node)
        
        # Add edges from transitions
        for state_name, state in workflow.states.items():
            for transition in state.transitions:
                edge = DAGEdge(
                    source=state_name,
                    target=transition.target,
                    condition=transition.condition,
                    metadata=transition.metadata
                )
                dag.add_edge(edge)
        
        # Add edges from dependencies
        for state_name, state in workflow.states.items():
            for dep in state.dependencies:
                # Check if edge already exists
                existing_edges = dag.get_edges(source=dep, target=state_name)
                if not existing_edges:
                    edge = DAGEdge(
                        source=dep,
                        target=state_name,
                        metadata={"type": "dependency"}
                    )
                    dag.add_edge(edge)
        
        # Validate DAG
        dag.validate()
        
        return dag
    
    def build_agent(self, workflow: WorkflowDefinition, dag: DAG[StateDefinition]) -> Agent:
        """Build agent from workflow and DAG."""
        agent = Agent(name=workflow.name)
        
        # Get topological order
        topo_order = dag.topological_sort()
        
        # Add states in order
        for state_name in topo_order:
            state_def = workflow.states[state_name]
            
            # Create state function
            state_func = self._create_state_function(state_def)
            
            # Determine dependencies
            dependencies = {}
            for dep in state_def.dependencies:
                dependencies[dep] = "required"
            
            # Add state to agent
            agent.add_state(
                name=state_name,
                func=state_func,
                dependencies=dependencies,
                resources=self._convert_resources(state_def.resources),
                max_retries=state_def.retries
            )
        
        return agent
    
    def _create_state_function(self, state_def: StateDefinition):
        """Create state function from definition."""
        plugin_name = state_def.config.get("plugin")
        
        if plugin_name and plugin_name in self._plugins:
            # Use plugin
            plugin = self._plugins[plugin_name]
            return plugin.get_state_function(state_def.type, state_def.config)
        else:
            # Create generic function
            # NOTE: This generic function is basic. The code generator produces more complete
            # async functions in the generated workflow file.
            async def state_function(context): # Make sure 'context' is defined or imported
                # Generic implementation
                # Access context properties like: context.current_node.id, context.current_node.data.config
                logger.info(f"Executing generic state: {state_def.name}", config=state_def.config)
                context.set_state("state_name", state_def.name) # Example, context API might differ
                context.set_state("state_type", state_def.type)
                context.set_state("config", state_def.config)
                
                # Return transitions
                if state_def.transitions:
                    if len(state_def.transitions) == 1:
                        return state_def.transitions[0].target
                    else:
                        # Multiple transitions - would need condition evaluation
                        # This basic version just returns the first one or a list
                        logger.warning(f"Multiple transitions for {state_def.name} without plugin/custom logic. Returning first or list.")
                        return state_def.transitions[0].target # Or handle conditions
                
                return None
            
            return state_function
    
    def _convert_resources(self, resources):
        """Convert resource definition to agent format."""
        if not resources:
            return None
        
        # Ensure ResourceRequirements is imported or defined
        from core.resources.requirements import ResourceRequirements # Assuming this path
        
        return ResourceRequirements(
            cpu_units=resources.cpu_units,
            memory_mb=resources.memory_mb,
            io_weight=resources.io_weight,
            network_weight=resources.network_weight,
            gpu_units=resources.gpu_units,
            timeout=resources.timeout
        )


class YAMLDAGBuilder(DAGBuilder):
    """Build DAG directly from YAML."""
    
    def build_from_yaml(self, yaml_path: Union[str, Path]) -> Tuple[WorkflowDefinition, DAG[StateDefinition]]:
        """Build workflow and DAG from YAML file."""
        workflow = YAMLParser.parse_file(yaml_path)
        dag = self.build_from_definition(workflow)
        return workflow, dag
    
    def build_from_yaml_string(self, yaml_string: str) -> Tuple[WorkflowDefinition, DAG[StateDefinition]]:
        """Build workflow and DAG from YAML string."""
        workflow = YAMLParser.parse_string(yaml_string)
        dag = self.build_from_definition(workflow)
        return workflow, dag


class CodeGenerator:
    """Generate Python code from workflow definition."""
    
    DEFAULT_TEMPLATES = {
        "workflow.py.j2": DEFAULT_WORKFLOW_TEMPLATE_STR,
        "state_function.py.j2": DEFAULT_STATE_FUNCTION_TEMPLATE_STR,
        "workflow_test.py.j2": DEFAULT_WORKFLOW_TEST_TEMPLATE_STR,
    }

    def __init__(self, template_dir: Optional[Path] = None):
        # self.template_dir_path points to user-specified dir or default core/dag/templates
        self.template_dir_path = template_dir or Path(__file__).parent / "templates"
        self.jinja_env = None
        # Jinja env is initialized lazily on first use by _ensure_jinja_env()
    
    def _ensure_jinja_env(self):
        """Initialize Jinja2 environment lazily to avoid circular imports."""
        if self.jinja_env:
            return

        try:
            import jinja2
        except ImportError:
            logger.error("jinja2_not_installed", error="Jinja2 library is not installed. Code generation unavailable.")
            raise

        loaders = []
        # Priority 1: Filesystem loader from self.template_dir_path if it's a valid directory
        if self.template_dir_path.is_dir():
            logger.debug(f"Attempting to use FileSystemLoader for templates from: {self.template_dir_path}")
            loaders.append(jinja2.FileSystemLoader(str(self.template_dir_path)))
        else:
            logger.debug(f"Template directory not found or not a directory: {self.template_dir_path}. Skipping FileSystemLoader.")

        # Priority 2: Built-in dictionary loader (always available as a fallback)
        logger.debug("Adding DictLoader with default built-in templates as fallback.")
        loaders.append(jinja2.DictLoader(self.DEFAULT_TEMPLATES))
        
        final_loader = jinja2.ChoiceLoader(loaders)
        self.jinja_env = jinja2.Environment(loader=final_loader, trim_blocks=True, lstrip_blocks=True)
        
        # Make sure Context is available in templates if not explicitly passed
        # but it's better to pass it via render context
        # self.jinja_env.globals['Context'] = Context 
        
        # Add 'tojson' filter if not present (Jinja2 usually has it via 'json' extension or by default)
        # For safety, ensure it's there
        try:
            import json
            from markupsafe import Markup
            self.jinja_env.filters['tojson'] = lambda value: Markup(json.dumps(value))
        except ImportError:
            logger.warning("json module not found, 'tojson' filter might not work.")
        except Exception as e:
            logger.error("Failed to set up jinja environment", error=str(e), path=self.template_dir_path)
            raise
    
    def generate_workflow_code(
        self,
        workflow: WorkflowDefinition,
        output_file: Optional[Path] = None
    ) -> str:
        """Generate Python code for workflow."""
        self._ensure_jinja_env()
        template = self.jinja_env.get_template("workflow.py.j2")
        
        # Ensure Context is available for the template
        from core.agent import Context # Import locally if needed for template
        
        code = template.render(
            workflow=workflow,
            generated_at=datetime.utcnow().isoformat(),
            imports=self._get_required_imports(workflow),
            Context=Context # Pass Context type to template if needed for type hints
        )
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.warning("code_formatting_failed", error=str(e))
        
        if output_file:
            output_file.write_text(code, encoding='utf-8')
        
        return code
    
    def generate_state_code(self, state: StateDefinition) -> str:
        """Generate code for a single state."""
        self._ensure_jinja_env()
        template = self.jinja_env.get_template("state_function.py.j2")
        from core.agent import Context # Import locally if needed for template
        
        return template.render(state=state, Context=Context)
    
    def generate_test_code(
        self,
        workflow: WorkflowDefinition,
        output_file: Optional[Path] = None
    ) -> str:
        """Generate test code for workflow."""
        self._ensure_jinja_env()
        template = self.jinja_env.get_template("workflow_test.py.j2")
        
        code = template.render(
            workflow=workflow,
            test_cases=self._generate_test_cases(workflow)
        )
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.warning("test_formatting_failed", error=str(e))
        
        if output_file:
            output_file.write_text(code, encoding='utf-8')
        
        return code
    
    def _get_required_imports(self, workflow: WorkflowDefinition) -> List[str]:
        """Determine required imports for workflow."""
        imports = [
            "import asyncio",
            "from core.agent import Agent, Context", # Make sure Context is exported from core.agent
            "from core.resources import ResourceRequirements", # Assuming this path is correct
            "import logging",
        ]
        imports.append("logger = logging.getLogger(__name__)") # Define logger instance
        imports.append("logging.basicConfig(level=logging.INFO) # Basic logging setup")
        
        # Check for specific plugin imports
        plugins = set()
        for state in workflow.states.values():
            if "plugin" in state.config:
                plugins.add(state.config["plugin"])
        
        for plugin_name in plugins:
            # Make plugin import more robust, assuming a standard plugin structure
            # e.g. from plugins.echo_plugin import EchoPlugin
            plugin_class_name = "".join(part.capitalize() for part in plugin_name.split('_')) + "Plugin"
            plugin_module_name = plugin_name.lower() 
            # Check if it's a snake_case plugin name e.g. my_plugin
            if '_' in plugin_name:
                plugin_module_name = plugin_name.lower()
            else: # assume camelCase or simple name like 'echo'
                plugin_module_name = plugin_name.lower() + "_plugin"


            imports.append(f"from plugins.{plugin_module_name} import {plugin_class_name} # Placeholder: Adjust if plugin naming/location differs")
        
        return imports
    
    def _generate_test_cases(self, workflow: WorkflowDefinition) -> List[Dict[str, Any]]:
        """Generate test cases for workflow."""
        test_cases = []
        
        # Test case for successful path
        test_cases.append({
            "name": "test_successful_execution",
            "description": "Test successful workflow execution with default path.",
            "setup": {}, # Can be used to pass initial context data
            "expected_states": list(workflow.states.keys()) # Illustrative, actual assertion might differ
        })
        
        # Test case for each branch (if identifiable)
        # This is a simplified approach; more complex branching might need specific test data
        for state_name, state in workflow.states.items():
            if len(state.transitions) > 1:
                for i, transition in enumerate(state.transitions):
                    test_cases.append({
                        "name": f"test_{state_name}_to_{transition.target.replace('.', '_')}_branch",
                        "description": f"Test transition from '{state_name}' to '{transition.target}' "
                                       f"{f'when condition {transition.condition} is met.' if transition.condition else ' (unconditional branch)'}",
                        "setup": {"condition_trigger": transition.condition or "default"}, # Info for test setup
                        "expected_next_state_from_" + state_name: transition.target
                    })
        
        return test_cases