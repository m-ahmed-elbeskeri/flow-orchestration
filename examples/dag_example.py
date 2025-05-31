#!/usr/bin/env python3
"""
examples/dag_example.py

End-to-end demo script for the DAG engine:
  1. Patches rogue Jinja2 loader if needed (though less critical now)
  2. Writes a simple YAML workflow
  3. Parses → DAGBuilder → DAG
  4. Generates workflow + test code via CodeGenerator (using its internal/filesystem templates)
  5. Executes the DAG with DAGExecutor
  6. Renders & prints a Graphviz visualisation
"""
import os
import sys
from pathlib import Path
import asyncio
import uuid

# from textwrap import dedent # No longer needed for bootstrapping templates
from datetime import datetime, timezone
import logging # Added for better output control

# Configure basic logging for the example script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Comprehensive Jinja2 cleanup and environment setup
# ─────────────────────────────────────────────────────────────────────────────
def clean_jinja2_environment():
    """Clean any corrupted jinja2.loaders file and prepare clean environment."""
    try:
        # Clear any cached jinja2 modules first
        modules_to_clear = [key for key in sys.modules.keys() if key.startswith('jinja2')]
        for module_key in modules_to_clear:
            del sys.modules[module_key]
        
        # Now try to find and clean the jinja2 installation
        import jinja2
        jinja2_path = Path(jinja2.__file__).parent
        loaders_path = jinja2_path / "loaders.py"
        
        if loaders_path.exists():
            text = loaders_path.read_text(encoding='utf-8')
            original_text = text
            
            # Remove any problematic imports from our project (if known)
            # This section might be less relevant if CodeGenerator is robust
            patterns_to_remove = [
                r'^from\s+core\.dag\.builder\s+import.*CodeGenerator.*$', # More specific
                r'^cg\s*=\s*CodeGenerator\(\).*$',
                r'^print\(\"\[DEBUG\].*$', # General debug prints that might have been added
                # r'.*DAGBuilder.*CodeGenerator.*' # This was too broad
            ]
            
            modified = False
            for pattern in patterns_to_remove:
                new_text = re.sub(pattern, '', text, flags=re.MULTILINE)
                if new_text != text:
                    modified = True
                    text = new_text
            
            # Clean up extra whitespace
            new_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text) # Multiple blank lines
            new_text = re.sub(r'\n\s*$', '\n', new_text) # Trailing whitespace on last line
            if new_text != text:
                modified = True
                text = new_text

            if modified:
                # Backup the original first
                backup_path = loaders_path.with_suffix('.py.backup')
                if not backup_path.exists(): # Only backup if one doesn't exist or create versioned
                    backup_path.write_text(original_text, encoding='utf-8')
                    logger.info(f"  Backed up original Jinja2 loaders.py to: {backup_path}")
                else:
                    logger.info(f"  Backup Jinja2 loaders.py already exists: {backup_path}")

                # Write cleaned version
                loaders_path.write_text(text, encoding='utf-8')
                logger.info(f"✔ Cleaned Jinja2 loaders.py: {loaders_path}")
                
                # Clear modules again after fixing
                modules_to_clear = [key for key in sys.modules.keys() if key.startswith('jinja2')]
                for module_key in modules_to_clear:
                    del sys.modules[module_key]
                    
                return True # Indicates a change was made
        
        return False # No changes made or file not found
        
    except Exception as e:
        logger.warning(f"Could not clean jinja2 environment: {e}")
        logger.warning("Recommendation: Run 'pip uninstall jinja2 && pip install jinja2 --force-reinstall'")
        return False

def setup_safe_jinja2():
    """Setup a safe jinja2 environment for our use."""
    try:
        # Try to import jinja2 safely
        import jinja2
        logger.info(f"✔ Jinja2 {jinja2.__version__} loaded successfully from: {jinja2.__file__}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load jinja2: {e}")
        return False

# Run the cleanup and setup
logger.info("🔧 Setting up clean Jinja2 environment...")
cleaned = clean_jinja2_environment()
if cleaned:
    logger.info("✔ Jinja2 environment potentially cleaned. Re-importing.")

if not setup_safe_jinja2():
    logger.error("❌ Cannot proceed without working Jinja2. Please reinstall jinja2.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap templates (ensure UTF-8 encoding) - NO LONGER NEEDED
# ─────────────────────────────────────────────────────────────────────────────
# The CodeGenerator now handles its own default templates internally
# and can optionally use templates from core/dag/templates if they exist.
# This example script no longer needs to create them.
logger.info("ℹ️ CodeGenerator will use its internal default templates or override from `core/dag/templates` if present.")

# For verification, you can check if the default template dir exists
# and CodeGenerator would use it:
# TEMPLATE_DIR_FOR_CODEGEN = Path(__file__).resolve().parent.parent / "core" / "dag" / "templates"
# if TEMPLATE_DIR_FOR_CODEGEN.exists() and TEMPLATE_DIR_FOR_CODEGEN.is_dir():
#    logger.info(f"ℹ️ Optional custom templates can be placed in: {TEMPLATE_DIR_FOR_CODEGEN}")
# else:
#    logger.info(f"ℹ️ Optional custom template directory not found at: {TEMPLATE_DIR_FOR_CODEGEN}. Using internal defaults.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Write the YAML definition
# ─────────────────────────────────────────────────────────────────────────────
yaml_text = """
name: demo_workflow
version: "1.0"
description: "A simple demonstration workflow."
start_state: say_hello
end_states: [done] # Optional, but good practice

states:
  say_hello:
    type: task
    description: "Greets the world."
    config:
      plugin: echo # Assumes an 'echo' plugin exists
      message: "Hello, world from the YAML!"
    transitions:
      - target: done
    # Example of resources (will be used by generated code if ResourceRequirements exist)
    # resources:
    #   cpu_units: 0.5
    #   memory_mb: 256
    #   timeout: 300

  done:
    type: task
    description: "Final state of the workflow."
    config:
      message: "Workflow finished." # Could be used by a logging or notification plugin
    # No transitions means it's an end state for this path
"""

# Define output directory for generated files (within the example's directory)
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_workflow_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

yaml_file = OUTPUT_DIR / "demo_workflow.yaml"
yaml_file.write_text(yaml_text.strip() + "\n", encoding='utf-8')
logger.info(f"✔ YAML workflow written to: {yaml_file}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Parse & build the DAG
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n🏗️  Building DAG from YAML...")

# Add core directory to sys.path to find modules if running script directly from examples
# This is often needed if your project structure isn't installed as a package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added project root to sys.path: {project_root}")


try:
    from core.dag.parser import YAMLParser
    from core.dag.builder import DAGBuilder, CodeGenerator
    from core.dag.executor import DAGExecutor
    from core.dag.graph import DAGVisualizer
    # Ensure Context is available if used directly or by generated code
    from core.agent import Agent, Context 
    from core.resources import ResourceRequirements # For generated code if resources are defined
    
    # Example: Create a dummy echo plugin for the demo if it doesn't exist
    # This is for making the generated code runnable out-of-the-box for this demo
    plugins_dir = project_root / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    echo_plugin_file = plugins_dir / "echo_plugin.py"
    if not echo_plugin_file.exists():
        echo_plugin_code = """
from plugins.base import Plugin, PluginManifest, PluginState
from core.agent.context import Context # Adjust import if Context location differs

class EchoPlugin(Plugin):
    manifest = PluginManifest(
        name="echo",
        version="0.1.0",
        description="A simple plugin that echoes a message."
    )
    async def execute(self, context: Context, config: dict) -> PluginState:
        message = config.get("message", "Default echo message.")
        print(f"[EchoPlugin]: {message}")
        context.set_output("echo_message", message)
        return PluginState.COMPLETED
"""
        echo_plugin_file.write_text(echo_plugin_code.strip() + "\n", encoding='utf-8')
        logger.info(f"✔ Created dummy echo_plugin.py at: {echo_plugin_file}")
        # Create __init__.py if it doesn't exist for plugins package
        (plugins_dir / "__init__.py").touch(exist_ok=True)

    logger.info("✔ All DAG modules imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import DAG modules: {e}")
    logger.error("Make sure all required modules (core.dag, core.agent, core.resources, plugins.base) are available in PYTHONPATH.")
    sys.exit(1)

try:
    workflow_def = YAMLParser.parse_file(yaml_file)
    builder = DAGBuilder()
    
    # If you have plugins to register with the builder (for `build_agent` method, not `CodeGenerator`):
    # from plugins.echo_plugin import EchoPlugin # Assuming it exists
    # builder.register_plugin(EchoPlugin())

    dag = builder.build_from_definition(workflow_def)
    logger.info("✔ DAG built successfully")
    logger.info(f"  Nodes: {', '.join(dag._nodes.keys())}")
    logger.info(f"  Edges: {len(dag._edges)}")

    
except Exception as e:
    logger.error(f"❌ Failed to build DAG: {e}", exc_info=True)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Code-generation
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n📝 Generating code...")

try:
    # CodeGenerator will use its internal templates, or templates from
    # core/dag/templates if that directory exists and contains relevant .j2 files.
    cg = CodeGenerator() 
    
    generated_py_file = OUTPUT_DIR / f"{workflow_def.name}_auto.py"
    generated_test_file = OUTPUT_DIR / f"test_{workflow_def.name}_auto.py"

    # Generate workflow code
    workflow_code = cg.generate_workflow_code(workflow_def, output_file=generated_py_file)
    
    # Generate test code
    test_code = cg.generate_test_code(workflow_def, output_file=generated_test_file)

    logger.info(f"✔ Code generation complete")
    logger.info(f"  • Workflow → {generated_py_file}")
    logger.info(f"  • Tests    → {generated_test_file}")
    
except Exception as e:
    logger.error(f"❌ Code generation failed: {e}", exc_info=True)
    logger.warning("Continuing without attempting to run generated code...")
    generated_py_file = None # Ensure it's None if generation failed

# ─────────────────────────────────────────────────────────────────────────────
# 5. Execute the DAG (using DAGExecutor, not the generated code directly yet)
#    For executing the generated code, you would typically run it as a separate Python script.
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n⚡ Executing DAG with DAGExecutor...")

async def run_dag_execution_with_executor():
    try:
        # The DAGExecutor uses the state functions created by DAGBuilder._create_state_function
        # which might be very generic if no plugins are registered with the DAGBuilder.
        # For this demo, we rely on the fact that YAML has `plugin: echo` and we provided a dummy plugin.
        # If plugins are involved, they should be registered with the DAGBuilder instance
        # *before* `build_agent` is called (which is implicitly used by some executors or manually).
        # However, this specific DAGExecutor might use the DAG structure directly.
        # Let's assume DAGExecutor can work with the raw DAG or requires an Agent.
        # If it needs an Agent, we'd do:
        # agent = builder.build_agent(workflow_def, dag)
        # executor = DAGExecutor(agent, max_concurrent=2)
        # ctx = await executor.execute() # If agent based

        # Simpler execution model if DAGExecutor works on DAG directly (hypothetical)
        # This part depends heavily on DAGExecutor's design.
        # For now, let's assume a simple DAG traversal if no agent logic is deeply integrated here.
        # The `dag_example.py` seemed to use a context for node executions.

        executor = DAGExecutor(max_concurrent=2) # Assuming it can take the DAG in execute()
        
        # The `execute` method of DAGExecutor likely needs the DAG and possibly an initial context.
        # The previous example passed `dag` to `executor.execute(dag)`.
        # Let's create a basic initial context.
        from core.dag.executor import ExecutionContext  # make sure this import exists

        execution_context = ExecutionContext(
            workflow_id=workflow_def.name,
            execution_id=str(uuid.uuid4()),
            dag=dag,
            shared_state={"start_time": datetime.now(timezone.utc).isoformat()}
        )
        final_context = await executor.execute(dag, context=execution_context)

        
        logger.info("✔ DAG execution (via DAGExecutor) completed")
        logger.info("\n--- DAGExecutor Execution Summary ---")
        if hasattr(final_context, 'node_executions') and final_context.node_executions:
            for name, exec_info in final_context.node_executions.items():
                status = exec_info.status.value if hasattr(exec_info.status, 'value') else str(exec_info.status)
                duration = exec_info.duration if exec_info.duration is not None else "N/A"
                logger.info(f"  Node: {name:<15} Status: {status:<12} Duration: {duration}")
        else:
            logger.info("  No detailed node execution information available in the final context.")
            
    except Exception as e:
        logger.error(f"❌ DAG execution (via DAGExecutor) failed: {e}", exc_info=True)

try:
    asyncio.run(run_dag_execution_with_executor())
except Exception as e:
    logger.error(f"❌ asyncio.run error during DAGExecutor execution: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5b. (Optional) Attempt to run the generated workflow code
# ─────────────────────────────────────────────────────────────────────────────
if generated_py_file and generated_py_file.exists():
    logger.info(f"\n🚀 Attempting to run the generated workflow: {generated_py_file}...")
    import subprocess
    try:
        # Ensure the generated code can find 'core' and 'plugins'
        # We modify PYTHONPATH for the subprocess.
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        # Prepend project_root to PYTHONPATH
        env["PYTHONPATH"] = str(project_root) + os.pathsep + current_pythonpath
        
        process = subprocess.run(
            [sys.executable, str(generated_py_file)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=30 # Add a timeout
        )
        logger.info(f"✔ Generated workflow executed successfully.")
        if process.stdout:
            logger.info("--- Generated Workflow STDOUT ---")
            for line in process.stdout.strip().split('\n'):
                logger.info(f"  {line}")
            logger.info("-------------------------------")
        if process.stderr:
            logger.warning("--- Generated Workflow STDERR ---")
            for line in process.stderr.strip().split('\n'):
                logger.warning(f"  {line}")
            logger.warning("--------------------------------")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Generated workflow execution failed with return code {e.returncode}.")
        if e.stdout:
            logger.error("--- STDOUT from failed execution ---")
            logger.error(e.stdout)
            logger.error("------------------------------------")
        if e.stderr:
            logger.error("--- STDERR from failed execution ---")
            logger.error(e.stderr)
            logger.error("------------------------------------")
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Generated workflow execution timed out.")
    except Exception as e:
        logger.error(f"❌ Error attempting to run generated workflow: {e}", exc_info=True)
else:
    logger.warning("\n⏩ Skipping execution of generated workflow code as it was not created.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualise with Graphviz
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n📊 Generating visualization...")

try:
    viz_output_basename = OUTPUT_DIR / "demo_workflow_graph"
    dot_src = DAGVisualizer.to_dot(dag)
    logger.info("✔ Graphviz visualization generation attempt complete.")
    
    dot_file_path = viz_output_basename.with_suffix(".dot")
    png_file_path = viz_output_basename.with_suffix(".png")

    if dot_file_path.exists():
        logger.info(f"  ✔ DOT source written to: {dot_file_path}")
        logger.info("\n--- Graphviz DOT ---")
        # Print only a snippet if too long
        dot_lines = dot_src.splitlines()
        if len(dot_lines) > 15:
            for line in dot_lines[:10]: logger.info(line)
            logger.info("... (DOT source truncated)")
            for line in dot_lines[-5:]: logger.info(line)
        else:
            logger.info(dot_src)
        logger.info("--------------------")
    else:
        logger.warning(f"  ⚠️ DOT source file not found at expected location: {dot_file_path}")


    if png_file_path.exists():
        logger.info(f"\n✔ PNG image written to: {png_file_path}")
    else:
        logger.warning(f"\n⚠️  PNG image not found at: {png_file_path}")
        logger.warning("   This usually means Graphviz 'dot' command-line tool is not installed or not in PATH.")
        logger.warning("   To install Graphviz, visit: https://graphviz.org/download/")
        
except Exception as e:
    logger.error(f"❌ Visualization failed: {e}", exc_info=True)

logger.info(f"\n🎉 Demo completed at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
logger.info(f"ℹ️  All generated outputs are in: {OUTPUT_DIR}")

# Need to import os for subprocess environment modification
import os