"""Determinism checking and SHA256 pinning for workflow execution."""

import ast
import asyncio
import hashlib
import inspect
import json
import sys
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from core.agent.base import Agent
from core.agent.state import StateFunction  # Fix: Import from state module
from core.agent.context import Context

logger = structlog.get_logger(__name__)

class NonDeterministicError(Exception):
    """Raised when non-deterministic behavior is detected"""
    pass

@dataclass
class StateFingerprint:
    """Fingerprint of a state function for determinism checking"""
    state_name: str
    function_hash: str
    source_hash: str
    dependencies: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    external_calls: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_name': self.state_name,
            'function_hash': self.function_hash,
            'source_hash': self.source_hash,
            'dependencies': list(self.dependencies),
            'imports': list(self.imports),
            'external_calls': list(self.external_calls),
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateFingerprint":
        return cls(
            state_name=data['state_name'],
            function_hash=data['function_hash'],
            source_hash=data['source_hash'],
            dependencies=set(data.get('dependencies', [])),
            imports=set(data.get('imports', [])),
            external_calls=set(data.get('external_calls', [])),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class FunctionAnalyzer:
    """Analyzes functions for deterministic properties"""
    
    NON_DETERMINISTIC_CALLS = {
        'random', 'time', 'datetime', 'uuid', 'os.urandom',
        'requests', 'httpx', 'aiohttp', 'input', 'open'
    }
    
    SAFE_BUILTINS = {
        'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set',
        'tuple', 'sorted', 'sum', 'min', 'max', 'abs', 'round'
    }

    def analyze_function(self, func: Callable) -> StateFingerprint:
        """Analyze a function and create its fingerprint"""
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = ""
        
        function_hash = self._hash_bytecode(func)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        
        # Parse AST for dependencies
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        
        dependencies = set()
        imports = set()
        external_calls = set()
        
        if tree:
            visitor = DeterminismVisitor()
            visitor.visit(tree)
            dependencies = visitor.dependencies
            imports = visitor.imports
            external_calls = visitor.external_calls
        
        return StateFingerprint(
            state_name=func.__name__,
            function_hash=function_hash,
            source_hash=source_hash,
            dependencies=dependencies,
            imports=imports,
            external_calls=external_calls
        )

    def _hash_bytecode(self, func: Callable) -> str:
        """Create hash of function bytecode"""
        try:
            code = func.__code__
        except AttributeError:
            return hashlib.sha256(str(func).encode()).hexdigest()
        
        code_data = {
            'co_code': code.co_code,
            'co_names': code.co_names,
            'co_varnames': code.co_varnames,
            'co_consts': code.co_consts
        }
        
        code_str = json.dumps(code_data, default=str, sort_keys=True)
        return hashlib.sha256(code_str.encode()).hexdigest()

    def check_determinism(self, func: Callable) -> List[str]:
        """Check function for non-deterministic patterns"""
        issues = []
        
        fingerprint = self.analyze_function(func)
        
        # Check for non-deterministic calls
        for call in fingerprint.external_calls:
            if any(nd in call for nd in self.NON_DETERMINISTIC_CALLS):
                issues.append(f"Non-deterministic call detected: {call}")
        
        # Check for time-based operations
        if any('time' in imp for imp in fingerprint.imports):
            issues.append("Time-based import detected")
        
        # Check for random operations
        if any('random' in imp for imp in fingerprint.imports):
            issues.append("Random import detected")
        
        return issues

class DeterminismVisitor(ast.NodeVisitor):
    """AST visitor to analyze deterministic properties"""
    
    def __init__(self):
        self.dependencies = set()
        self.imports = set()
        self.external_calls = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if hasattr(node.func, 'attr') and hasattr(node.func, 'value'):
            if hasattr(node.func.value, 'id'):
                call_name = f"{node.func.value.id}.{node.func.attr}"
                self.external_calls.add(call_name)
        elif hasattr(node.func, 'id'):
            self.external_calls.add(node.func.id)
        
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.dependencies.add(name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.dependencies.add(node.id)
        self.generic_visit(node)

class DeterminismChecker:
    """Main class for checking workflow determinism"""
    
    def __init__(self):
        self.analyzer = FunctionAnalyzer()
        self._fingerprints: Dict[str, StateFingerprint] = {}
        self._state_outputs: Dict[str, List[Any]] = {}

    def register_state(self, state_name: str, func: StateFunction) -> StateFingerprint:
        """Register a state function for determinism checking"""
        fingerprint = self.analyzer.analyze_function(func)
        fingerprint.state_name = state_name
        
        # Check for issues
        issues = self.analyzer.check_determinism(func)
        if issues:
            logger.warning(f"Non-deterministic patterns in state {state_name}: {issues}")
        
        # Check for changes if already registered
        if state_name in self._fingerprints:
            current = self._fingerprints[state_name]
            expected = fingerprint
            
            if current.function_hash != expected.function_hash:
                logger.warning(f"State {state_name} function changed")
            
            if current.external_calls != expected.external_calls:
                new_calls = current.external_calls - expected.external_calls
                if new_calls:
                    logger.warning(f"New external calls in {state_name}: {new_calls}")
        
        self._fingerprints[state_name] = fingerprint
        return fingerprint

    def record_execution(self, state_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Record state execution for comparison"""
        input_hash = self._hash_inputs(inputs)
        
        if state_name not in self._state_outputs:
            self._state_outputs[state_name] = []
        
        execution_record = {
            'input_hash': input_hash,
            'outputs': outputs,
            'timestamp': datetime.utcnow()
        }
        
        self._state_outputs[state_name].append(execution_record)
        
        # Check for consistency if we have previous executions
        if len(self._state_outputs[state_name]) > 1:
            previous = self._state_outputs[state_name][-2]
            if (previous['input_hash'] == input_hash and 
                not self._outputs_match(previous['outputs'], outputs)):
                logger.warning(f"Non-deterministic output detected in state {state_name}")

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Create consistent hash of inputs"""
        stable_json = json.dumps(
            inputs, 
            sort_keys=True, 
            default=str, 
            separators=(',', ':')
        )
        return hashlib.sha256(stable_json.encode()).hexdigest()

    def _outputs_match(self, out1: Dict[str, Any], out2: Dict[str, Any]) -> bool:
        """Compare outputs for equality"""
        return json.dumps(out1, sort_keys=True, default=str) == \
               json.dumps(out2, sort_keys=True, default=str)

    def validate_agent_state(self, agent: Agent) -> None:
        """Validate deterministic properties of an agent"""
        for state_name, func in agent.states.items():
            self.register_state(state_name, func)

    def create_execution_manifest(self, agent: Agent) -> Dict[str, Any]:
        """Create manifest of execution environment"""
        manifest = {
            'agent_name': agent.name,
            'timestamp': datetime.utcnow().isoformat(),
            'state_fingerprints': {},
            'python_version': str(sys.version_info),
            'dependencies': []
        }
        
        for state_name, func in agent.states.items():
            fingerprint = self.register_state(state_name, func)
            manifest['state_fingerprints'][state_name] = fingerprint.to_dict()
        
        return manifest

    def load_manifest(self, manifest: Dict[str, Any]) -> None:
        """Load execution manifest for comparison"""
        for state_name, fp_data in manifest.get('state_fingerprints', {}).items():
            self._fingerprints[state_name] = StateFingerprint.from_dict(fp_data)

def deterministic(func: Callable) -> Callable:
    """Decorator to mark and validate deterministic functions"""
    analyzer = FunctionAnalyzer()
    issues = analyzer.check_determinism(func)
    
    if issues:
        logger.warning(f"Determinism issues in {func.__name__}: {issues}")
    
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Record inputs for determinism checking
            result = await func(*args, **kwargs)
            # Record outputs
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return sync_wrapper

def capture_state(key: str) -> Callable:
    """Decorator to capture state for determinism verification"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Capture input state
                context = args[0] if args and isinstance(args[0], Context) else None
                
                input_state = {
                    'args': args[1:] if context else args,
                    'kwargs': kwargs,
                    'context_state': context.shared_state.copy() if context else {}
                }
                
                result = await func(*args, **kwargs)
                
                output_state = {
                    'result': result,
                    'context_state': context.shared_state.copy() if context else {}
                }
                
                # Store for later verification
                # Implementation depends on global state management
                
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result
            return sync_wrapper
    return decorator