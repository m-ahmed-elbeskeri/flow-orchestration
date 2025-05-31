# cli/commands/workflow.py
"""Workflow management commands with integrated code generation."""

import click
import yaml
import asyncio
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil


@click.group()
def workflow():
    """Manage workflows - run, list, generate code, validate, and visualize."""
    pass


# ============================================================================
# Workflow Commands
# ============================================================================

@workflow.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--timeout', '-t', default=300, help='Execution timeout in seconds')
def run(workflow_file: Path, verbose: bool, timeout: int):
    """Run a workflow from a YAML file."""
    if verbose:
        click.echo(f"📁 Loading workflow from: {workflow_file}")
    
    try:
        # Setup Python path for imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        with open(workflow_file, 'r') as f:
            workflow_def = yaml.safe_load(f)
        
        workflow_name = workflow_def.get('name', 'unnamed')
        click.echo(f"🚀 Running workflow: {workflow_name}")
        
        # Try to import and use the workflow system
        try:
            from core.dag.parser import YAMLParser
            from core.dag.builder import DAGBuilder, DAGRuntime
            from core.dag.executor import DAGExecutor
            
            if verbose:
                click.echo("✅ DAG modules loaded successfully")
            
            # Parse workflow
            workflow_definition = YAMLParser.parse_file(workflow_file)
            
            # Build DAG
            builder = DAGBuilder()
            dag = builder.build_from_definition(workflow_definition)
            
            if verbose:
                click.echo(f"📊 DAG built: {len(dag._nodes)} nodes, {len(dag._edges)} edges")
            
            # Execute workflow
            async def execute_workflow():
                runtime = DAGRuntime()
                context = await runtime.execute_workflow(workflow_definition, dag)
                return context
            
            # Run with timeout
            try:
                context = asyncio.wait_for(execute_workflow(), timeout=timeout)
                asyncio.run(context)
                click.echo("✅ Workflow execution completed successfully!")
                
            except asyncio.TimeoutError:
                click.echo(f"❌ Workflow execution timed out after {timeout} seconds", err=True)
                sys.exit(1)
                
        except ImportError as e:
            click.echo(f"❌ Could not import workflow modules: {e}", err=True)
            click.echo("💡 Try generating code first: workflow codegen generate ...", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error running workflow: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@workflow.command()
@click.option('--directory', '-d', type=click.Path(path_type=Path), 
              default=Path('workflows'), help='Directory to search for workflows')
@click.option('--details', is_flag=True, help='Show workflow details')
def list(directory: Path, details: bool):
    """List all available workflows."""
    if not directory.exists():
        click.echo(f"📁 Workflows directory '{directory}' not found.")
        click.echo(f"💡 Create it with: mkdir {directory}")
        return
    
    yaml_files = list(directory.glob('*.yaml')) + list(directory.glob('*.yml'))
    
    if not yaml_files:
        click.echo(f"📭 No workflows found in {directory}")
        click.echo("💡 Create a workflow with: workflow init")
        return
    
    click.echo(f"📚 Available workflows in {directory}:")
    
    for workflow_file in sorted(yaml_files):
        if details:
            try:
                with open(workflow_file, 'r') as f:
                    workflow_def = yaml.safe_load(f)
                
                name = workflow_def.get('name', workflow_file.stem)
                description = workflow_def.get('description', 'No description')
                version = workflow_def.get('version', 'Unknown')
                states_count = len(workflow_def.get('states', {}))
                
                click.echo(f"  📄 {workflow_file.name}")
                click.echo(f"     Name: {name}")
                click.echo(f"     Version: {version}")
                click.echo(f"     Description: {description}")
                click.echo(f"     States: {states_count}")
                click.echo()
            except Exception as e:
                click.echo(f"  ❌ {workflow_file.name} (error reading: {e})")
        else:
            click.echo(f"  📄 {workflow_file.name}")


@workflow.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def info(workflow_file: Path, verbose: bool):
    """Show detailed workflow information."""
    try:
        with open(workflow_file, 'r') as f:
            workflow_def = yaml.safe_load(f)
        
        click.echo(f"📋 Workflow Information: {workflow_file.name}")
        click.echo("=" * 60)
        click.echo(f"Name: {workflow_def.get('name', 'Unnamed')}")
        click.echo(f"Version: {workflow_def.get('version', 'Not specified')}")
        click.echo(f"Description: {workflow_def.get('description', 'No description')}")
        
        states = workflow_def.get('states', {})
        click.echo(f"States: {len(states)}")
        click.echo(f"Start State: {workflow_def.get('start_state', 'Not specified')}")
        
        end_states = workflow_def.get('end_states', [])
        click.echo(f"End States: {', '.join(end_states) if end_states else 'Not specified'}")
        
        metadata = workflow_def.get('metadata', {})
        if metadata and verbose:
            click.echo(f"\nMetadata:")
            for key, value in metadata.items():
                click.echo(f"  {key}: {value}")
        
        if verbose and states:
            click.echo(f"\n🔧 State Details:")
            for name, state in states.items():
                click.echo(f"  • {name}")
                click.echo(f"    Type: {state.get('type', 'task')}")
                
                deps = state.get('dependencies', [])
                if deps:
                    click.echo(f"    Dependencies: {', '.join(deps)}")
                
                transitions = state.get('transitions', [])
                if transitions:
                    if isinstance(transitions[0], str):
                        targets = transitions
                    else:
                        targets = [t.get('target', '') for t in transitions]
                    click.echo(f"    Transitions: {', '.join(targets)}")
                
                resources = state.get('resources', {})
                if resources:
                    click.echo(f"    Resources: {resources}")
                
                description = state.get('description') or state.get('metadata', {}).get('description')
                if description:
                    click.echo(f"    Description: {description}")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Error reading workflow: {e}", err=True)


@workflow.command()
@click.option('--name', '-n', prompt='Workflow name', help='Name of the workflow')
@click.option('--description', '-d', help='Workflow description')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output YAML file')
@click.option('--template', '-t', type=click.Choice(['basic', 'etl', 'ml', 'data_processing']),
              default='basic', help='Workflow template to use')
def init(name: str, description: Optional[str], output: Optional[Path], template: str):
    """Initialize a new workflow YAML file."""
    
    if not output:
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        output = Path(f"{safe_name}.yaml")
    
    if output.exists():
        if not click.confirm(f"File {output} exists. Overwrite?"):
            click.echo("❌ Initialization cancelled.")
            return
    
    # Create workflow based on template
    if template == 'basic':
        workflow_yaml = f"""name: {name}
version: "1.0"
description: "{description or 'A new workflow'}"
start_state: start
end_states: [end]

metadata:
  author: "{click.get_current_context().info_name or 'Unknown'}"
  created: "{datetime.now().isoformat()}"
  template: "basic"

states:
  start:
    type: task
    description: "Starting state"
    config:
      message: "Workflow started: {name}"
    transitions:
      - process
    
  process:
    type: task
    description: "Main processing state"
    config:
      operation: "process_data"
    dependencies: [start]
    transitions:
      - end
    resources:
      cpu_units: 1.0
      memory_mb: 512
    
  end:
    type: task
    description: "Final state"
    config:
      message: "Workflow completed: {name}"
    dependencies: [process]
"""
    
    elif template == 'etl':
        workflow_yaml = f"""name: {name}
version: "1.0"
description: "{description or 'ETL data processing workflow'}"
start_state: extract
end_states: [load_complete]

metadata:
  author: "{click.get_current_context().info_name or 'Unknown'}"
  created: "{datetime.now().isoformat()}"
  template: "etl"

states:
  extract:
    type: task
    description: "Extract data from source"
    config:
      source_type: "database"
      connection_string: "postgresql://localhost/source_db"
      query: "SELECT * FROM source_table"
    transitions:
      - transform
    resources:
      cpu_units: 2.0
      memory_mb: 1024

  transform:
    type: task
    description: "Transform the extracted data"
    config:
      operations:
        - "clean_data"
        - "normalize_values"
        - "validate_schema"
    dependencies: [extract]
    transitions:
      - load
    resources:
      cpu_units: 4.0
      memory_mb: 2048

  load:
    type: task
    description: "Load data to destination"
    config:
      destination_type: "database"
      connection_string: "postgresql://localhost/target_db"
      table: "target_table"
    dependencies: [transform]
    transitions:
      - load_complete
    resources:
      cpu_units: 2.0
      memory_mb: 1024

  load_complete:
    type: task
    description: "ETL process completed"
    config:
      notification: true
      message: "ETL workflow completed successfully"
    dependencies: [load]
"""

    elif template == 'ml':
        workflow_yaml = f"""name: {name}
version: "1.0"
description: "{description or 'Machine learning training workflow'}"
start_state: load_data
end_states: [model_deployed]

metadata:
  author: "{click.get_current_context().info_name or 'Unknown'}"
  created: "{datetime.now().isoformat()}"
  template: "ml"

states:
  load_data:
    type: task
    description: "Load training data"
    config:
      data_source: "s3://ml-bucket/training-data.csv"
      split_ratio: 0.8
    transitions:
      - preprocess
    resources:
      cpu_units: 2.0
      memory_mb: 4096

  preprocess:
    type: task
    description: "Preprocess the data"
    config:
      operations:
        - "handle_missing_values"
        - "feature_scaling"
        - "feature_engineering"
    dependencies: [load_data]
    transitions:
      - train_model
    resources:
      cpu_units: 4.0
      memory_mb: 8192

  train_model:
    type: task
    description: "Train ML model"
    config:
      algorithm: "random_forest"
      hyperparameters:
        n_estimators: 100
        max_depth: 10
    dependencies: [preprocess]
    transitions:
      - evaluate_model
    resources:
      cpu_units: 8.0
      memory_mb: 16384
      gpu_units: 1.0

  evaluate_model:
    type: task
    description: "Evaluate model performance"
    config:
      metrics: ["accuracy", "precision", "recall", "f1_score"]
      threshold: 0.85
    dependencies: [train_model]
    transitions:
      - target: deploy_model
        condition: "metrics.accuracy >= 0.85"
      - target: retrain_model
        condition: "metrics.accuracy < 0.85"

  deploy_model:
    type: task
    description: "Deploy model to production"
    config:
      deployment_target: "kubernetes"
      model_endpoint: "/api/v1/predict"
    transitions:
      - model_deployed

  retrain_model:
    type: task
    description: "Retrain with different parameters"
    config:
      hyperparameters:
        n_estimators: 200
        max_depth: 15
    transitions:
      - evaluate_model

  model_deployed:
    type: task
    description: "Model deployment completed"
    config:
      notification: true
      message: "ML model deployed successfully"
    dependencies: [deploy_model]
"""

    else:  # data_processing
        workflow_yaml = f"""name: {name}
version: "1.0"
description: "{description or 'Data processing workflow'}"
start_state: read_data
end_states: [write_complete]

metadata:
  author: "{click.get_current_context().info_name or 'Unknown'}"
  created: "{datetime.now().isoformat()}"
  template: "data_processing"

states:
  read_data:
    type: task
    description: "Read input data"
    config:
      input_path: "/data/input"
      format: "json"
    transitions:
      - validate_data
    resources:
      cpu_units: 1.0
      memory_mb: 1024

  validate_data:
    type: task
    description: "Validate input data"
    config:
      schema_file: "schemas/input_schema.json"
      strict_mode: true
    dependencies: [read_data]
    transitions:
      - target: process_data
        condition: "validation.passed"
      - target: handle_errors
        condition: "not validation.passed"

  process_data:
    type: parallel
    description: "Process data in parallel"
    config:
      tasks:
        - filter_data
        - transform_data
        - enrich_data
    dependencies: [validate_data]
    transitions:
      - aggregate_results

  filter_data:
    type: task
    description: "Filter data based on criteria"
    config:
      filters:
        - "remove_duplicates"
        - "filter_by_date"

  transform_data:
    type: task
    description: "Transform data format"
    config:
      transformations:
        - "normalize_fields"
        - "convert_types"

  enrich_data:
    type: task
    description: "Enrich with additional data"
    config:
      enrichment_sources:
        - "lookup_tables"
        - "external_apis"

  aggregate_results:
    type: task
    description: "Aggregate parallel processing results"
    config:
      aggregation_method: "merge"
    dependencies: [filter_data, transform_data, enrich_data]
    transitions:
      - write_data

  write_data:
    type: task
    description: "Write processed data"
    config:
      output_path: "/data/output"
      format: "parquet"
    dependencies: [aggregate_results]
    transitions:
      - write_complete

  handle_errors:
    type: task
    description: "Handle validation errors"
    config:
      error_action: "quarantine"
      quarantine_path: "/data/quarantine"
    transitions:
      - write_complete

  write_complete:
    type: task
    description: "Data processing completed"
    config:
      notification: true
      message: "Data processing workflow completed"
    dependencies: [write_data, handle_errors]
"""
    
    output.write_text(workflow_yaml)
    click.echo(f"✅ Workflow initialized: {output}")
    click.echo(f"📝 Template: {template}")
    click.echo(f"💡 Next steps:")
    click.echo(f"   1. Edit the workflow: {output}")
    click.echo(f"   2. Validate: workflow validate {output}")
    click.echo(f"   3. Generate code: workflow codegen generate {output}")
    click.echo(f"   4. Run: workflow run {output}")


# ============================================================================
# Code Generation Commands
# ============================================================================

@workflow.group()
def codegen():
    """Code generation commands."""
    pass


@codegen.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), 
              help='Output directory for generated files')
@click.option('--workflow-name', '-w', help='Override workflow name')
@click.option('--template-dir', '-t', type=click.Path(path_type=Path),
              help='Custom template directory')
@click.option('--format-code/--no-format', default=True,
              help='Format generated code with black')
@click.option('--generate-tests/--no-tests', default=True,
              help='Generate test files')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def generate(workflow_file: Path, output_dir: Optional[Path], workflow_name: Optional[str],
             template_dir: Optional[Path], format_code: bool, generate_tests: bool,
             overwrite: bool, verbose: bool):
    """Generate Python workflow code from YAML definition."""
    
    try:
        # Setup paths
        if not output_dir:
            output_dir = workflow_file.parent / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python path for imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core.dag.parser import YAMLParser
        from core.dag.builder import CodeGenerator
        
        if verbose:
            click.echo(f"📁 Input file: {workflow_file}")
            click.echo(f"📁 Output directory: {output_dir}")
        
        # Parse YAML
        click.echo("🔍 Parsing YAML workflow...")
        workflow_def = YAMLParser.parse_file(workflow_file)
        
        if workflow_name:
            workflow_def.name = workflow_name
        
        if verbose:
            click.echo(f"✅ Parsed workflow: {workflow_def.name}")
            click.echo(f"   States: {len(workflow_def.states)}")
            click.echo(f"   Version: {workflow_def.version}")
        
        # Setup code generator
        generator = CodeGenerator(template_dir=template_dir)
        
        # Generate workflow code
        click.echo("🐍 Generating Python workflow code...")
        workflow_file_out = output_dir / f"{workflow_def.name}_auto.py"
        
        if workflow_file_out.exists() and not overwrite:
            if not click.confirm(f"File {workflow_file_out} exists. Overwrite?"):
                click.echo("❌ Code generation cancelled.")
                return
        
        workflow_code = generator.generate_workflow_code(workflow_def, workflow_file_out)
        
        # Generate test code
        if generate_tests:
            click.echo("🧪 Generating test code...")
            test_file = output_dir / f"test_{workflow_def.name}_auto.py"
            
            if test_file.exists() and not overwrite:
                if click.confirm(f"Test file {test_file} exists. Overwrite?"):
                    generator.generate_test_code(workflow_def, test_file)
            else:
                generator.generate_test_code(workflow_def, test_file)
        
        # Create supporting files
        _create_supporting_files(output_dir, workflow_def, verbose)
        
        click.echo(f"✅ Code generation complete!")
        click.echo(f"📁 Generated files in: {output_dir}")
        click.echo(f"🐍 Workflow file: {workflow_file_out}")
        if generate_tests:
            click.echo(f"🧪 Test file: {test_file}")
        
        click.echo(f"\n💡 Next steps:")
        click.echo(f"   cd {output_dir}")
        click.echo(f"   pip install -r requirements.txt")
        click.echo(f"   python {workflow_file_out.name}")
            
    except Exception as e:
        click.echo(f"❌ Error generating code: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def _create_supporting_files(output_dir: Path, workflow_def, verbose: bool):
    """Create supporting files for generated workflow."""
    
    # Create requirements.txt
    requirements_file = output_dir / "requirements.txt"
    if not requirements_file.exists():
        requirements = [
            "asyncio",
            "structlog",
            "pydantic>=2.0.0",
            "pyyaml",
            "black",
            "pytest",
            "pytest-asyncio"
        ]
        requirements_file.write_text("\n".join(requirements) + "\n")
        if verbose:
            click.echo(f"📦 Created requirements.txt")
    
    # Create README
    readme_file = output_dir / "README.md"
    if not readme_file.exists():
        readme_content = f"""# Generated Workflow: {workflow_def.name}

## Description
{workflow_def.description or 'Auto-generated workflow'}

## Files
- `{workflow_def.name}_auto.py` - Main workflow implementation
- `test_{workflow_def.name}_auto.py` - Unit tests
- `requirements.txt` - Python dependencies

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Workflow
```bash
python {workflow_def.name}_auto.py
```

### Run Tests
```bash
pytest test_{workflow_def.name}_auto.py -v
```

## Workflow States
{chr(10).join(f"- **{name}**: {state.metadata.get('description', state.get('description', 'No description'))}" for name, state in workflow_def.states.items())}

Generated on: {datetime.now().isoformat()}
"""
        readme_file.write_text(readme_content)
        if verbose:
            click.echo(f"📖 Created README.md")


# ============================================================================
# Validation Commands
# ============================================================================

@workflow.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Verbose validation output')
def validate(workflow_file: Path, verbose: bool):
    """Validate a YAML workflow definition."""
    
    try:
        # Setup Python path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core.dag.parser import YAMLParser
        from core.dag.builder import DAGBuilder
        
        click.echo(f"🔍 Validating workflow: {workflow_file}")
        
        # Parse YAML
        workflow_def = YAMLParser.parse_file(workflow_file)
        
        if verbose:
            click.echo(f"✅ YAML parsing successful")
            click.echo(f"   Workflow: {workflow_def.name}")
            click.echo(f"   States: {len(workflow_def.states)}")
        
        # Build and validate DAG
        builder = DAGBuilder()
        dag = builder.build_from_definition(workflow_def)
        
        if verbose:
            click.echo(f"✅ DAG construction successful")
            click.echo(f"   Nodes: {len(dag._nodes)}")
            click.echo(f"   Edges: {len(dag._edges)}")
        
        # Validate DAG structure
        dag.validate()
        
        click.echo("✅ Workflow validation passed!")
        
        if verbose:
            # Show execution order
            topo_order = dag.topological_sort()
            click.echo(f"📋 Execution order: {' → '.join(topo_order)}")
            
            # Show levels for parallel execution
            levels = dag.get_levels()
            click.echo(f"🔀 Parallel execution levels:")
            for i, level in enumerate(levels):
                click.echo(f"   Level {i}: {', '.join(level)}")
                
    except Exception as e:
        click.echo(f"❌ Workflow validation failed: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Visualization Commands
# ============================================================================

@workflow.command()
@click.argument('workflow_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file for visualization')
@click.option('--format', '-f', type=click.Choice(['dot', 'png', 'svg', 'mermaid']),
              default='mermaid', help='Output format')
@click.option('--show-metadata/--no-metadata', default=True,
              help='Include metadata in visualization')
def visualize(workflow_file: Path, output: Optional[Path], format: str, show_metadata: bool):
    """Generate workflow visualization."""
    
    try:
        # Setup Python path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core.dag.parser import YAMLParser
        from core.dag.builder import DAGBuilder
        from core.dag.graph import DAGVisualizer
        
        # Parse and build DAG
        workflow_def = YAMLParser.parse_file(workflow_file)
        builder = DAGBuilder()
        dag = builder.build_from_definition(workflow_def)
        
        # Generate visualization
        if format == 'mermaid':
            content = DAGVisualizer.to_mermaid(dag)
            extension = '.mmd'
        elif format in ['png', 'svg']:
            if not output:
                output = workflow_file.with_suffix(f'.{format}')
            DAGVisualizer.to_dot(dag, str(output.with_suffix('')))
            click.echo(f"✅ {format.upper()} visualization saved to: {output}")
            return
        else:  # dot format
            content = DAGVisualizer.to_dot(dag)
            extension = '.dot'
        
        if output:
            if not output.suffix:
                output = output.with_suffix(extension)
            output.write_text(content)
            click.echo(f"✅ Visualization saved to: {output}")
        else:
            click.echo(f"📊 {format.upper()} visualization:")
            click.echo("=" * 50)
            click.echo(content)
            
    except Exception as e:
        click.echo(f"❌ Error generating visualization: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Template Commands
# ============================================================================

@workflow.group()
def template():
    """Template management commands."""
    pass


@template.command()
def list():
    """List available workflow templates."""
    
    try:
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core.dag.templates import TemplateLibrary
        
        library = TemplateLibrary()
        templates = library.list_templates()
        
        if not templates:
            click.echo("No templates found.")
            return
        
        click.echo("📚 Available templates:")
        for template_name in templates:
            template = library.get_template(template_name)
            click.echo(f"  • {template_name}")
            click.echo(f"    Description: {template.description}")
            click.echo(f"    Version: {template.version}")
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Error listing templates: {e}", err=True)


@template.command()
@click.argument('template_name')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output YAML file')
@click.option('--variables', '-v', multiple=True,
              help='Template variables in key=value format')
def render(template_name: str, output: Optional[Path], variables: tuple):
    """Render a template to YAML."""
    
    try:
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core.dag.templates import TemplateLibrary
        
        # Parse variables
        template_vars = {}
        for var in variables:
            if '=' not in var:
                click.echo(f"❌ Invalid variable format: {var} (use key=value)", err=True)
                sys.exit(1)
            key, value = var.split('=', 1)
            # Try to parse as JSON for complex values
            try:
                template_vars[key] = json.loads(value)
            except json.JSONDecodeError:
                template_vars[key] = value
        
        # Render template
        library = TemplateLibrary()
        if template_name not in library.list_templates():
            click.echo(f"❌ Template '{template_name}' not found", err=True)
            click.echo(f"Available templates: {', '.join(library.list_templates())}")
            sys.exit(1)
        
        workflow_def = library.render_template(template_name, **template_vars)
        
        # Convert to YAML
        from core.dag.parser import YAMLParser
        yaml_content = YAMLParser.to_yaml(workflow_def)
        
        if output:
            output.write_text(yaml_content)
            click.echo(f"✅ Template rendered to: {output}")
        else:
            click.echo("📄 Rendered YAML:")
            click.echo("=" * 50)
            click.echo(yaml_content)
            
    except Exception as e:
        click.echo(f"❌ Error rendering template: {e}", err=True)


if __name__ == '__main__':
    workflow()