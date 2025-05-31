# plugins/registry.py
"""Plugin registry with automatic dependency installation."""

import importlib
import importlib.util
import subprocess
import sys
from typing import Dict, Type, Optional, List, Any, Tuple
from pathlib import Path
import yaml
import logging

from plugins.base import Plugin, PluginState

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugins with automatic dependency installation."""
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None, auto_install: bool = False):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_errors: Dict[str, str] = {}  # Track plugin load errors
        self._installed_dependencies: set = set()  # Track what we've installed
        self._failed_installations: set = set()  # Track failed installations
        self._loaded = False
        self._auto_install = auto_install
        self._plugin_dirs = plugin_dirs or self._discover_plugin_directories()
    
    def _discover_plugin_directories(self) -> List[Path]:
        """Discover plugin directories relative to project structure."""
        possible_dirs = []
        
        # Try relative to this file (plugins/registry.py)
        current_file = Path(__file__)
        plugins_dir = current_file.parent  # This should be the plugins/ directory
        if plugins_dir.name == 'plugins':
            possible_dirs.append(plugins_dir)
        
        # Try relative to project root
        project_root = current_file.parent.parent  # Should be Puffinflow/
        possible_dirs.extend([
            project_root / "plugins",
            project_root / "contrib" / "plugins",
            project_root / "extensions",
        ])
        
        # Try current working directory
        cwd = Path.cwd()
        possible_dirs.extend([
            cwd / "plugins",
            cwd / "contrib" / "plugins",
        ])
        
        # Filter to existing directories
        existing_dirs = [d for d in possible_dirs if d.exists() and d.is_dir()]
        
        if not existing_dirs:
            logger.warning(f"No plugin directories found. Searched: {possible_dirs}")
        
        return existing_dirs
    
    def set_auto_install(self, enabled: bool):
        """Enable or disable automatic dependency installation."""
        self._auto_install = enabled
        if enabled:
            logger.info("Auto-install enabled for plugin dependencies")
        else:
            logger.info("Auto-install disabled")
    
    def load_plugins(self, plugin_dirs: Optional[List[Path]] = None, auto_install: Optional[bool] = None):
        """Load all plugins from directories."""
        if self._loaded:
            return
        
        if plugin_dirs:
            self._plugin_dirs = plugin_dirs
        
        if auto_install is not None:
            self._auto_install = auto_install
        
        logger.debug(f"Loading plugins from directories: {self._plugin_dirs}")
        if self._auto_install:
            logger.info("Auto-install mode enabled - will install missing dependencies")
        
        loaded_count = 0
        error_count = 0
        installed_count = 0
        
        for plugin_dir in self._plugin_dirs:
            try:
                counts = self._load_plugins_from_directory(plugin_dir)
                loaded_count += counts[0]
                error_count += counts[1] 
                installed_count += counts[2]
            except Exception as e:
                logger.error(f"Failed to load plugins from {plugin_dir}: {e}")
                error_count += 1
        
        # Summary
        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} plugins")
        if error_count > 0:
            logger.warning(f"Failed to load {error_count} plugins")
        if installed_count > 0:
            logger.info(f"Auto-installed {installed_count} dependencies")
        
        self._loaded = True
    
    def _load_plugins_from_directory(self, plugin_dir: Path) -> Tuple[int, int, int]:
        """Load plugins from a specific directory."""
        loaded_count = 0
        error_count = 0
        installed_count = 0
        
        # Look for plugin directories (contain manifest.yaml)
        for path in plugin_dir.iterdir():
            if path.is_dir() and (path / "manifest.yaml").exists():
                try:
                    installed = self._load_plugin(path)
                    loaded_count += 1
                    if installed:
                        installed_count += installed
                except Exception as e:
                    plugin_name = path.name
                    self._plugin_errors[plugin_name] = str(e)
                    logger.debug(f"Plugin {plugin_name} failed to load: {e}")
                    error_count += 1
        
        return loaded_count, error_count, installed_count
    
    def _load_plugin(self, plugin_path: Path) -> int:
        """Load a single plugin with dependency management."""
        # Load manifest first
        try:
            with open(plugin_path / "manifest.yaml", 'r') as f:
                manifest = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Invalid manifest in {plugin_path}: {e}")
        
        plugin_name = manifest['name']
        dependencies = manifest.get('dependencies', [])
        
        # Check and install dependencies
        installed_count = 0
        if dependencies:
            missing_deps = self._check_dependencies(dependencies)
            if missing_deps:
                if self._auto_install:
                    logger.info(f"Installing missing dependencies for {plugin_name}: {missing_deps}")
                    installed_count = self._install_dependencies(missing_deps, plugin_name)
                    
                    # Re-check dependencies after installation
                    still_missing = self._check_dependencies(dependencies)
                    if still_missing:
                        raise ImportError(f"Still missing dependencies after installation: {', '.join(still_missing)}")
                else:
                    raise ImportError(f"Missing dependencies for {plugin_name}: {', '.join(missing_deps)}")
        
        logger.debug(f"Loading plugin: {plugin_name}")
        
        # Try different module loading strategies
        plugin_class = None
        
        # Strategy 1: Look for __init__.py with plugin class
        init_file = plugin_path / "__init__.py"
        if init_file.exists():
            plugin_class = self._load_plugin_class_from_file(init_file, plugin_name)
        
        # Strategy 2: Look for main.py
        if not plugin_class:
            main_file = plugin_path / "main.py"
            if main_file.exists():
                plugin_class = self._load_plugin_class_from_file(main_file, plugin_name)
        
        # Strategy 3: Create dynamic plugin from manifest
        if not plugin_class:
            plugin_class = self._create_dynamic_plugin_class(plugin_path, manifest)
        
        # Create plugin instance
        if plugin_class:
            try:
                plugin = plugin_class()
                self._plugins[plugin_name] = plugin
                logger.debug(f"Successfully loaded plugin: {plugin_name}")
                return installed_count
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate plugin {plugin_name}: {e}")
        else:
            raise RuntimeError(f"Could not find plugin class for {plugin_name}")
    
    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if plugin dependencies are available."""
        missing = []
        for dep in dependencies:
            # Skip if we know this installation failed
            if dep in self._failed_installations:
                missing.append(dep)
                continue
            
            # Try to import the dependency
            try:
                # Handle different dependency formats
                module_name = self._parse_dependency_name(dep)
                __import__(module_name)
            except ImportError:
                missing.append(dep)
        
        return missing
    
    def _parse_dependency_name(self, dep: str) -> str:
        """Parse dependency string to get module name."""
        # Handle different dependency formats
        if '>=' in dep:
            module_name = dep.split('>=')[0].strip()
        elif '==' in dep:
            module_name = dep.split('==')[0].strip()
        elif '>' in dep:
            module_name = dep.split('>')[0].strip()
        elif '<' in dep:
            module_name = dep.split('<')[0].strip()
        else:
            module_name = dep.strip()
        
        # Convert package names to module names for common cases
        module_name = module_name.replace('-', '_')
        
        # Handle special cases
        package_to_module = {
            'slack_sdk': 'slack_sdk',
            'google_auth': 'google.auth',
            'google_auth_oauthlib': 'google_auth_oauthlib',
            'google_auth_httplib2': 'google_auth_httplib2',
            'google_api_python_client': 'googleapiclient'
        }
        
        if module_name in package_to_module:
            return package_to_module[module_name]
        
        return module_name
    
    def _install_dependencies(self, dependencies: List[str], plugin_name: str) -> int:
        """Install missing dependencies."""
        installed_count = 0
        
        for dep in dependencies:
            # Skip if already installed or known to fail
            if dep in self._installed_dependencies or dep in self._failed_installations:
                continue
            
            try:
                logger.info(f"Installing {dep} for plugin {plugin_name}...")
                
                # Run pip install
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    logger.info(f"✅ Successfully installed {dep}")
                    self._installed_dependencies.add(dep)
                    installed_count += 1
                else:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    logger.error(f"❌ Failed to install {dep}: {error_msg}")
                    self._failed_installations.add(dep)
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Timeout installing {dep}")
                self._failed_installations.add(dep)
            except Exception as e:
                logger.error(f"❌ Error installing {dep}: {e}")
                self._failed_installations.add(dep)
        
        return installed_count
    
    def _load_plugin_class_from_file(self, file_path: Path, plugin_name: str) -> Optional[Type[Plugin]]:
        """Load plugin class from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}", 
                file_path
            )
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to enable relative imports
            sys.modules[spec.name] = module
            
            spec.loader.exec_module(module)
            
            # Find plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Plugin) and 
                    attr != Plugin):
                    return attr
            
            return None
            
        except Exception as e:
            raise ImportError(f"Error loading plugin class from {file_path}: {e}")
    
    def _create_dynamic_plugin_class(self, plugin_path: Path, manifest: Dict[str, Any]) -> Type[Plugin]:
        """Create a dynamic plugin class from manifest."""
        
        class DynamicPlugin(Plugin):
            def __init__(self):
                super().__init__(plugin_path / "manifest.yaml")
            
            def register_states(self) -> Dict[str, Type[PluginState]]:
                """Register states from states.py if it exists."""
                states = {}
                
                states_file = plugin_path / "states.py"
                if states_file.exists():
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"plugins.{manifest['name']}.states", 
                            states_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find state classes
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (isinstance(attr, type) and 
                                    issubclass(attr, PluginState) and 
                                    attr != PluginState):
                                    # Map class name to state name
                                    state_name = self._class_name_to_state_name(attr_name)
                                    states[state_name] = attr
                    
                    except Exception as e:
                        logger.debug(f"Could not load states from {states_file}: {e}")
                        # Return empty states dict instead of failing
                
                return states
            
            def _class_name_to_state_name(self, class_name: str) -> str:
                """Convert class name to state name."""
                import re
                
                # Remove 'State' suffix
                if class_name.endswith('State'):
                    class_name = class_name[:-5]
                
                # Convert CamelCase to snake_case
                s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
                return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        return DynamicPlugin
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        if not self._loaded:
            self.load_plugins()
        return self._plugins.get(name)
    
    def get_state_function(
        self,
        plugin_name: str,
        state_name: str,
        config: Dict[str, Any]
    ):
        """Get a state function from a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            available = list(self._plugins.keys())
            error_info = ""
            if plugin_name in self._plugin_errors:
                error_info = f" (load error: {self._plugin_errors[plugin_name]})"
            raise ValueError(f"Plugin '{plugin_name}' not found{error_info}. Available: {available}")
        
        return plugin.get_state_function(state_name, config)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all available plugins."""
        if not self._loaded:
            self.load_plugins()
        
        plugins_list = []
        for plugin in self._plugins.values():
            try:
                states = list(plugin.register_states().keys()) if hasattr(plugin, 'register_states') else []
                plugins_list.append({
                    "name": plugin.manifest.name,
                    "version": plugin.manifest.version,
                    "description": plugin.manifest.description,
                    "author": getattr(plugin.manifest, 'author', ''),
                    "states": states,
                    "dependencies": plugin.manifest.dependencies,
                    "status": "loaded"
                })
            except Exception as e:
                logger.error(f"Error getting info for plugin {plugin.manifest.name}: {e}")
                plugins_list.append({
                    "name": getattr(plugin.manifest, 'name', 'unknown'),
                    "version": getattr(plugin.manifest, 'version', '0.0.0'),
                    "description": f"Error: {e}",
                    "author": "",
                    "states": [],
                    "dependencies": [],
                    "status": "error"
                })
        
        # Also include failed plugins in the list
        for plugin_name, error in self._plugin_errors.items():
            if not any(p['name'] == plugin_name for p in plugins_list):
                plugins_list.append({
                    "name": plugin_name,
                    "version": "unknown",
                    "description": f"Failed to load: {error}",
                    "author": "",
                    "states": [],
                    "dependencies": [],
                    "status": "failed"
                })
        
        return plugins_list
    
    def get_plugin_errors(self) -> Dict[str, str]:
        """Get dictionary of plugin load errors."""
        return self._plugin_errors.copy()
    
    def get_installation_stats(self) -> Dict[str, Any]:
        """Get statistics about dependency installations."""
        return {
            "installed_dependencies": list(self._installed_dependencies),
            "failed_installations": list(self._failed_installations),
            "auto_install_enabled": self._auto_install
        }
    
    def refresh(self, auto_install: Optional[bool] = None):
        """Refresh plugin registry."""
        self._plugins.clear()
        self._plugin_errors.clear()
        self._loaded = False
        
        if auto_install is not None:
            self._auto_install = auto_install
        
        self.load_plugins()
    
    def install_plugin_dependencies(self, plugin_name: str, force: bool = False) -> bool:
        """Install dependencies for a specific plugin."""
        # Find plugin directory
        for plugin_dir in self._plugin_dirs:
            plugin_path = plugin_dir / plugin_name
            if plugin_path.exists() and (plugin_path / "manifest.yaml").exists():
                try:
                    with open(plugin_path / "manifest.yaml", 'r') as f:
                        manifest = yaml.safe_load(f)
                    
                    dependencies = manifest.get('dependencies', [])
                    if not dependencies:
                        return True
                    
                    if force:
                        # Remove from failed list to retry
                        for dep in dependencies:
                            self._failed_installations.discard(dep)
                    
                    missing_deps = self._check_dependencies(dependencies)
                    if missing_deps:
                        installed = self._install_dependencies(missing_deps, plugin_name)
                        return installed > 0
                    else:
                        return True
                        
                except Exception as e:
                    logger.error(f"Error installing dependencies for {plugin_name}: {e}")
                    return False
        
        raise ValueError(f"Plugin {plugin_name} not found")


# Global registry instance
plugin_registry = PluginRegistry()