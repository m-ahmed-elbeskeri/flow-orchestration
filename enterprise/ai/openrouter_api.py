# enterprise/ai/openrouter_api.py
"""OpenRouter API client for AI model integration."""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
import aiohttp
from dataclasses import dataclass


@dataclass
class OpenRouterResponse:
    """Response from OpenRouter API."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]


class OpenRouterAPI:
    """Client for OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenRouter API client.
        
        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",  # Optional, for tracking
            "X-Title": "Workflow Orchestrator AI Copilot",  # Optional
        }
    
    async def generate(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-sonnet",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> OpenRouterResponse:
        """Generate text using OpenRouter.
        
        Args:
            prompt: User prompt
            model: Model identifier (e.g., "anthropic/claude-3-sonnet")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            OpenRouterResponse object
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"OpenRouter API error: {error_text}")
                
                data = await response.json()
                
                # Extract content from response
                content = data["choices"][0]["message"]["content"]
                
                return OpenRouterResponse(
                    content=content,
                    model=data.get("model", model),
                    usage=data.get("usage", {}),
                    raw_response=data
                )
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-sonnet",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ):
        """Stream text generation using OpenRouter.
        
        Yields:
            Chunks of generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"OpenRouter API error: {error_text}")
                
                # Read streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue