# enterprise/ai/flow_copilot.py
"""AI-powered flow generation copilot."""

import json
import yaml
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import uuid

from enterprise.ai.openrouter_api import OpenRouterAPI
from enterprise.ai.plugin_discovery import PluginDiscovery
from core.agent.base import Agent
from core.agent.state import Priority, StateResult
from core.agent.dependencies import DependencyType, DependencyLifecycle
from core.agent.context import Context


@dataclass
class FlowAnalysis:
    """Analysis of a flow generation request."""
    clear_enough: bool
    missing_information: List[str]
    clarification_questions: List[str]
    suggested_flow_description: str
    suggested_variables: Dict[str, List[str]]
    suggested_plugins: List[str]
    complexity: str  # simple, moderate, complex


class FlowCopilot:
    """AI-powered copilot for generating workflow definitions."""
    
    def __init__(
        self,
        openrouter_api: Optional[OpenRouterAPI] = None,
        plugin_discovery: Optional[PluginDiscovery] = None,
        model: str = "anthropic/claude-3-sonnet"
    ):
        """Initialize the flow copilot.
        
        Args:
            openrouter_api: OpenRouter API client
            plugin_discovery: Plugin discovery system
            model: Model to use for generation
        """
        self.api = openrouter_api or OpenRouterAPI()
        self.discovery = plugin_discovery or PluginDiscovery()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.clarifications: Dict[str, Dict[str, str]] = {}
        self.detected_variables = {
            "env": set(),
            "local": set(),
            "constants": set()
        }
    
    async def analyze_request(self, request: str) -> FlowAnalysis:
        """Analyze a user request to identify missing information.
        
        Args:
            request: User's natural language request
            
        Returns:
            FlowAnalysis object
        """
        plugin_capabilities = self.discovery.get_plugin_capabilities_prompt()
        
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
        
        self.conversation_history.append({"role": "user", "content": request})
        
        try:
            response = await self.api.generate(
                prompt=f"Analyze this workflow request: {request}",
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1024
            )
            
            # Extract JSON from response
            analysis_dict = self._extract_json(response.content)
            
            # Track detected variables
            suggested_vars = analysis_dict.get("suggested_variables", {})
            self.detected_variables["env"].update(suggested_vars.get("env", []))
            self.detected_variables["local"].update(suggested_vars.get("local", []))
            self.detected_variables["constants"].update(suggested_vars.get("constants", []))
            
            return FlowAnalysis(**analysis_dict)
            
        except Exception as e:
            print(f"Error analyzing request: {e}")
            return FlowAnalysis(
                clear_enough=False,
                missing_information=["Unable to analyze request"],
                clarification_questions=["Could you please provide more details about your workflow?"],
                suggested_flow_description="Workflow based on user request",
                suggested_variables={"env": [], "local": [], "constants": []},
                suggested_plugins=[],
                complexity="moderate"
            )
    
    async def ask_clarifications(self, analysis: FlowAnalysis) -> Dict[str, str]:
        """Interactive clarification gathering (for CLI usage).
        
        Args:
            analysis: FlowAnalysis result
            
        Returns:
            Dictionary of answers
        """
        answers = {}
        
        if not analysis.clear_enough:
            print(f"\nSuggested workflow: {analysis.suggested_flow_description}")
            
            if analysis.suggested_plugins:
                print("\nSuggested plugins:")
                for plugin in analysis.suggested_plugins:
                    print(f"  - {plugin}")
            
            print("\nI need some clarification:")
            
            for i, question in enumerate(analysis.clarification_questions):
                print(f"\n{question}")
                answer = input("> ")
                
                question_key = f"clarification_{i+1}"
                answers[question_key] = answer
                self.clarifications[question_key] = {
                    "question": question,
                    "answer": answer
                }
                
                # Track variables mentioned in answers
                self._detect_variables_in_text(answer)
        
        return answers
    
    async def generate_workflow(
        self,
        request: str,
        clarifications: Optional[Dict[str, str]] = None,
        analysis: Optional[FlowAnalysis] = None
    ) -> Dict[str, Any]:
        """Generate a complete workflow definition.
        
        Args:
            request: Original user request
            clarifications: Optional clarification answers
            analysis: Optional pre-computed analysis
            
        Returns:
            Dictionary containing workflow definition and metadata
        """
        plugin_capabilities = self.discovery.get_plugin_capabilities_prompt()
        
        # Build context from clarifications
        context_text = f"Original request: {request}"
        if clarifications:
            context_text += "\n\nClarifications:"
            for key, value in clarifications.items():
                if key in self.clarifications:
                    context_text += f"\n- {self.clarifications[key]['question']}"
                    context_text += f"\n  Answer: {self.clarifications[key]['answer']}"
        
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
        
        try:
            response = await self.api.generate(
                prompt=context_text,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2048
            )
            
            # Extract YAML
            yaml_content = self._extract_yaml(response.content)
            
            if yaml_content:
                # Parse and validate
                workflow_dict = yaml.safe_load(yaml_content)
                
                # Generate additional artifacts
                result = {
                    "workflow_definition": workflow_dict,
                    "yaml_content": yaml_content,
                    "analysis": analysis,
                    "detected_variables": dict(self.detected_variables),
                    "conversation_history": self.conversation_history
                }
                
                # Generate Python code
                result["python_code"] = self._generate_python_code(workflow_dict)
                
                # Generate mermaid diagram
                result["mermaid_diagram"] = self._generate_mermaid_diagram(workflow_dict)
                
                # Generate environment template if needed
                if self.detected_variables["env"]:
                    result["env_template"] = self._generate_env_template()
                
                return result
            else:
                raise ValueError("Failed to generate valid YAML")
                
        except Exception as e:
            print(f"Error generating workflow: {e}")
            return {
                "error": str(e),
                "workflow_definition": None,
                "yaml_content": None
            }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
        # Remove markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            for block in text.split("```"):
                if "{" in block and "}" in block:
                    text = block
                    break
        
        # Try to parse
        try:
            return json.loads(text)
        except:
            # Try to find JSON pattern
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        
        # Return default
        return {
            "clear_enough": False,
            "missing_information": [],
            "clarification_questions": [],
            "suggested_flow_description": "",
            "suggested_variables": {"env": [], "local": [], "constants": []},
            "suggested_plugins": [],
            "complexity": "moderate"
        }

    def _extract_yaml(self, text: str) -> Optional[str]:
        """Extract YAML from text."""
        # Remove markdown blocks
        if "```yaml" in text:
            return text.split("```yaml")[1].split("```")[0].strip()
        elif "```" in text:
            for block in text.split("```"):
                if "name:" in block or "states:" in block:
                    return block.strip()
        
        # Look for YAML patterns
        if "name:" in text or "states:" in text:
            # Find start of YAML
            for line in text.split("\n"):
                if line.strip().startswith("name:") or line.strip().startswith("states:"):
                    start_idx = text.find(line)
                    return text[start_idx:].strip()
        
        return None

    def _detect_variables_in_text(self, text: str):
        """Detect variables mentioned in text."""
        # Environment variables (UPPER_CASE)
        env_pattern = r'\b([A-Z][A-Z0-9_]+)\b'
        env_matches = re.findall(env_pattern, text)
        
        # Filter common words
        non_vars = {"API", "URL", "HTTP", "JSON", "YAML", "TRUE", "FALSE"}
        self.detected_variables["env"].update(
            var for var in env_matches 
            if len(var) > 2 and var not in non_vars
        )
        
        # Local variables (camelCase or snake_case)
        local_pattern = r'\b([a-z][a-zA-Z0-9_]*)\b'
        local_matches = re.findall(local_pattern, text)
        
        # Filter common words
        common_words = {"the", "and", "or", "if", "then", "else", "when", "for", "with"}
        self.detected_variables["local"].update(
            var for var in local_matches 
            if len(var) > 2 and var not in common_words
        )

    def _generate_python_code(self, workflow: Dict[str, Any]) -> str:
        """Generate Python code for the workflow."""
        code_lines = [
            "\"\"\"Generated workflow code.\"\"\"",
            "import asyncio",
            "from core.agent.base import Agent",
            "from core.agent.dependencies import DependencyType, DependencyLifecycle",
            "from core.plugins.registry import plugin_registry",
            "",
            "",
            f"async def create_{workflow.get('name', 'workflow')}_agent():",
            f"    \"\"\"Create {workflow.get('description', 'workflow')} agent.\"\"\"",
            "    agent = Agent(",
            f"        name=\"{workflow.get('name', 'workflow')}\",",
            "        max_concurrent=10",
            "    )",
            "",
            "    # Load plugins",
            "    plugin_registry.load_plugins()",
            ""
        ]
        
        # Add states
        states = workflow.get("states", {})
        for state_id, state_config in states.items():
            plugin = state_config.get("plugin", "")
            action = state_config.get("action", "")
            
            code_lines.append(f"    # State: {state_id}")
            
            # Build dependencies
            deps = state_config.get("dependencies", {})
            if deps:
                dep_dict = {}
                for dep_id, dep_config in deps.items():
                    dep_type = dep_config.get("type", "required").upper()
                    lifecycle = dep_config.get("lifecycle", "always").upper()
                    dep_dict[dep_id] = f"(DependencyType.{dep_type}, DependencyLifecycle.{lifecycle})"
                
                deps_str = json.dumps(dep_dict).replace('"(', '(').replace(')"', ')')
            else:
                deps_str = "None"
            
            code_lines.extend([
                f"    agent.add_state(",
                f"        name=\"{state_id}\",",
                f"        func=plugin_registry.get_state_function(",
                f"            \"{plugin}\",",
                f"            \"{action}\",",
                f"            {json.dumps(state_config.get('inputs', {}), indent=12)}",
                f"        ),",
                f"        dependencies={deps_str}",
                f"    )",
                ""
            ])
        
        code_lines.extend([
            "    return agent",
            "",
            "",
            "async def main():",
            f"    agent = await create_{workflow.get('name', 'workflow')}_agent()",
            "    await agent.run()",
            "",
            "",
            "if __name__ == \"__main__\":",
            "    asyncio.run(main())"
        ])
        
        return "\n".join(code_lines)

    def _generate_mermaid_diagram(self, workflow: Dict[str, Any]) -> str:
        """Generate Mermaid diagram for the workflow."""
        lines = ["graph TD"]
        
        states = workflow.get("states", {})
        
        # Add nodes
        for state_id, state_config in states.items():
            label = state_config.get("action", state_id).replace(".", "_")
            lines.append(f"    {state_id}[{label}]")
        
        # Add edges based on dependencies
        for state_id, state_config in states.items():
            deps = state_config.get("dependencies", {})
            for dep_id in deps:
                if dep_id in states:
                    lines.append(f"    {dep_id} --> {state_id}")
        
        return "\n".join(lines)

    def _generate_env_template(self) -> str:
        """Generate environment variable template."""
        lines = [
            "# Environment variables for workflow",
            "# Copy to .env and fill in values",
            ""
        ]
        
        for var in sorted(self.detected_variables["env"]):
            lines.append(f"{var}=")
        
        return "\n".join(lines)