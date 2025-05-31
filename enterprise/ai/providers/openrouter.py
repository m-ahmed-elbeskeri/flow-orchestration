"""OpenRouter AI provider integration."""

import httpx
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3-sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    site_url: Optional[str] = None
    site_name: Optional[str] = None


class OpenRouterProvider:
    """OpenRouter API provider."""
    
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "HTTP-Referer": self.config.site_url or "https://workflow-orchestrator.io",
                "X-Title": self.config.site_name or "Workflow Orchestrator"
            },
            timeout=60.0
        )
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Get completion from OpenRouter."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs
        }
        
        try:
            response = await self.client.post(
                "/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(
                "openrouter_completion_error",
                error=str(e),
                model=model or self.config.default_model
            )
            raise
    
    async def stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from OpenRouter."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "stream": True,
            **kwargs
        }
        
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = await self.client.get("/models")
        response.raise_for_status()
        return response.json()["data"]
    
    async def close(self):
        """Close the client connection."""
        await self.client.aclose()