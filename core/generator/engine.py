# core/generator/engine.py
"""Main code generation engine."""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2
from datetime import datetime
import json

from .parser import WorkflowSpec, parse_workflow_file, parse_workflow_string
from plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generate standalone workflow code from YAML specifications."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = templates_dir
        self._setup_jinja()
        
        # Load available plugins for code generation
        plugin_registry.load_plugins()
        self.available_plugins = {
            plugin['name']: plugin 
            for plugin in plugin_registry.list_plugins()
        }
    
    def _setup_jinja(self):
            """Setup Jinja2 environment."""
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir),
                trim_blocks=False,  # ← keep the newline after a block tag
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            
            # Add custom filters
            self.jinja_env.filters['snake_case'] = self._snake_case
            self.jinja_env.filters['camel_case'] = self._camel_case
            self.jinja_env.filters['pascal_case'] = self._pascal_case
            self.jinja_env.filters['indent'] = self._indent
            self.jinja_env.filters['quote'] = self._quote
            self.jinja_env.filters['python_value'] = self._python_value
            
            # Add global functions
            self.jinja_env.globals['now'] = datetime.now
            self.jinja_env.globals['datetime'] = datetime
    
    def _python_value(self, value: Any) -> str:
        """Convert JSON-like values to Python syntax."""
        if isinstance(value, bool):
            return 'True' if value else 'False'
        elif value is None:
            return 'None'
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, (list, dict)):
            # Convert to string and replace JSON booleans/null
            json_str = json.dumps(value)
            json_str = json_str.replace('true', 'True')
            json_str = json_str.replace('false', 'False')
            json_str = json_str.replace('null', 'None')
            return json_str
        else:
            return str(value)
    
    def generate_from_file(self, yaml_file: Path, output_dir: Path) -> None:
        """Generate code from YAML file."""
        spec = parse_workflow_file(yaml_file)
        self.generate_from_spec(spec, output_dir)
    
    def generate_from_string(self, yaml_content: str, output_dir: Path) -> None:
        """Generate code from YAML string."""
        spec = parse_workflow_string(yaml_content)
        self.generate_from_spec(spec, output_dir)
    
    def generate_from_spec(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate code from workflow specification."""
        logger.info(f"Generating workflow code for '{spec.name}' in {output_dir}")
        
        # Create output directory
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        self._create_directory_structure(output_dir)
        
        # Generate core files
        self._generate_main_workflow(spec, output_dir)
        self._generate_requirements(spec, output_dir)
        self._generate_config_files(spec, output_dir)
        self._generate_integration_files(spec, output_dir)
        self._generate_utility_files(spec, output_dir)
        self._generate_docker_files(spec, output_dir)
        self._generate_documentation(spec, output_dir)
        self._generate_tests(spec, output_dir)
        
        logger.info(f"✅ Successfully generated workflow project in {output_dir}")
    
    def _create_directory_structure(self, output_dir: Path) -> None:
        """Create project directory structure."""
        directories = [
            "workflow",
            "integrations", 
            "config",
            "scripts",
            "tests",
            "docs",
            ".github/workflows"  # For CI/CD
        ]
        
        for dir_name in directories:
            (output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_dirs = ["workflow", "integrations", "tests"]
        for dir_name in init_dirs:
            init_file = output_dir / dir_name / "__init__.py"
            init_file.write_text('"""Generated workflow package."""\n')
    
    def _generate_main_workflow(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate main workflow file."""
        template = self.jinja_env.get_template('workflow.py.j2')
        content = template.render(spec=spec, plugins=self.available_plugins)
        
        workflow_file = output_dir / "workflow" / "main.py"
        workflow_file.write_text(content, encoding='utf-8')
        
        # Generate workflow runner
        template = self.jinja_env.get_template('run.py.j2')
        content = template.render(spec=spec)
        
        run_file = output_dir / "run.py"
        run_file.write_text(content)
        run_file.chmod(0o755)  # Make executable
    
    def _generate_requirements(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate requirements.txt with proper deduplication and version handling."""
        # Base requirements
        base_requirements = [
            "pyyaml>=6.0",
            "pydantic>=1.8.0",
            "structlog>=21.0.0",
            "click>=8.0.0",
            "python-dotenv>=0.19.0",
            "aiofiles>=0.8.0",
        ]
        
        # Collect all requirements
        all_requirements = set()
        
        # Add base requirements
        for req in base_requirements:
            all_requirements.add(req)
        
        # Add plugin dependencies
        for integration in spec.integrations:
            plugin = plugin_registry.get_plugin(integration.name)
            if not plugin:
                continue
            
            # Add manifest dependencies
            for dep in plugin.manifest.dependencies:
                if dep and not dep.startswith("#"):
                    all_requirements.add(dep.strip())
            
            # Check for custom requirements.txt
            plugin_dir = plugin.manifest_path.parent
            custom_req_file = plugin_dir / "requirements.txt"
            if custom_req_file.is_file():
                try:
                    for line in custom_req_file.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            all_requirements.add(line)
                except Exception as e:
                    logger.warning(f"Could not read requirements for {integration.name}: {e}")
        
        # Sort requirements
        def get_package_name(requirement):
            """Extract package name from requirement string."""
            import re
            match = re.match(r'^([a-zA-Z0-9_\-\[\]]+)', requirement)
            return match.group(1).lower() if match else requirement.lower()
        
        sorted_requirements = sorted(all_requirements, key=get_package_name)
        
        # Write requirements.txt
        req_file = output_dir / "requirements.txt"
        with req_file.open('w', encoding='utf-8') as f:
            f.write("# Auto-generated requirements for " + spec.name + "\n")
            f.write("# Generated on " + datetime.now().isoformat() + "\n\n")
            
            # Core dependencies
            f.write("# Core dependencies\n")
            for req in sorted_requirements:
                if any(req.startswith(core) for core in ['pyyaml', 'pydantic', 'structlog', 'click', 'python-dotenv', 'aiofiles']):
                    f.write(f"{req}\n")
            
            f.write("\n# Plugin dependencies\n")
            for req in sorted_requirements:
                if not any(req.startswith(core) for core in ['pyyaml', 'pydantic', 'structlog', 'click', 'python-dotenv', 'aiofiles']):
                    f.write(f"{req}\n")
    
    def _generate_config_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate configuration files."""
        # Main config.yaml
        template = self.jinja_env.get_template('config.yaml.j2')
        content = template.render(spec=spec)
        
        config_file = output_dir / "config" / "config.yaml"
        config_file.write_text(content)
        
        # Environment template
        template = self.jinja_env.get_template('env.template.j2')
        content = template.render(spec=spec)
        
        env_file = output_dir / ".env.template"
        env_file.write_text(content)
    
    def _generate_integration_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate integration wrapper files."""
        for integration in spec.integrations:
            self._generate_integration_file(integration, spec, output_dir)
    
    def _generate_integration_file(self, integration, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate individual integration file."""
        integration_name = integration.name
        
        # Check if we have a specific template for this integration
        template_name = f"integrations/{integration_name}.py.j2"
        
        try:
            template = self.jinja_env.get_template(template_name)
        except jinja2.TemplateNotFound:
            # Use generic template
            template = self.jinja_env.get_template('integrations/generic.py.j2')
        
        # Get plugin information
        plugin_info = self.available_plugins.get(integration_name, {})
        
        content = template.render(
            spec=spec,
            integration=integration,
            plugin_info=plugin_info
        )
        
        integration_file = output_dir / "integrations" / f"{integration_name}.py"
        integration_file.write_text(content)
    
    def _generate_utility_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate utility and helper files."""
        # Generate logging configuration
        template = self.jinja_env.get_template('utils/logging.py.j2')
        content = template.render(spec=spec)
        
        utils_dir = output_dir / "workflow" / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        (utils_dir / "__init__.py").write_text("")
        (utils_dir / "logging.py").write_text(content)
        
        # Generate helpers
        template = self.jinja_env.get_template('utils/helpers.py.j2')
        content = template.render(spec=spec)
        (utils_dir / "helpers.py").write_text(content)
    
    def _generate_docker_files(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate Docker configuration."""
        # Dockerfile
        template = self.jinja_env.get_template('Dockerfile.j2')
        content = template.render(spec=spec)
        (output_dir / "Dockerfile").write_text(content)
        
        # docker-compose.yml
        template = self.jinja_env.get_template('docker-compose.yml.j2')
        content = template.render(spec=spec)
        (output_dir / "docker-compose.yml").write_text(content)
        
        # .dockerignore
        dockerignore_content = """
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.env
"""
        (output_dir / ".dockerignore").write_text(dockerignore_content.strip())
    
    def _generate_documentation(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate documentation files."""
        # README.md
        template = self.jinja_env.get_template('README.md.j2')
        content = template.render(spec=spec, plugins=self.available_plugins)
        (output_dir / "README.md").write_text(content)
        
        # API documentation
        template = self.jinja_env.get_template('docs/api.md.j2')
        content = template.render(spec=spec)
        (output_dir / "docs" / "api.md").write_text(content)
        
        # Deployment guide
        template = self.jinja_env.get_template('docs/deployment.md.j2')
        content = template.render(spec=spec)
        (output_dir / "docs" / "deployment.md").write_text(content)
    
    def _generate_tests(self, spec: WorkflowSpec, output_dir: Path) -> None:
        """Generate test files."""
        # Main test file
        template = self.jinja_env.get_template('tests/test_workflow.py.j2')
        content = template.render(spec=spec)
        (output_dir / "tests" / "test_workflow.py").write_text(content)
        
        # Integration tests
        for integration in spec.integrations:
            template_name = f"tests/test_{integration.name}.py.j2"
            try:
                template = self.jinja_env.get_template(template_name)
                content = template.render(spec=spec, integration=integration)
                (output_dir / "tests" / f"test_{integration.name}.py").write_text(content)
            except jinja2.TemplateNotFound:
                # Skip if no specific test template
                pass
        
        # pytest configuration
        pytest_ini_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
"""
        (output_dir / "pytest.ini").write_text(pytest_ini_content)
    
    # Utility methods for Jinja filters
    def _snake_case(self, text: str) -> str:
        """Convert to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _camel_case(self, text: str) -> str:
        """Convert to camelCase."""
        components = text.replace('-', '_').split('_')
        return components[0].lower() + ''.join(x.capitalize() for x in components[1:])
    
    def _pascal_case(self, text: str) -> str:
        """Convert to PascalCase."""
        components = text.replace('-', '_').split('_')
        return ''.join(x.capitalize() for x in components)
    
    def _indent(self, text: str, width: int = 4) -> str:
        """Indent text by specified width."""
        lines = text.splitlines()
        indent = ' ' * width
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    def _quote(self, text: str) -> str:
        """Quote text for Python strings."""
        return repr(text)