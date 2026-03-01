#!/usr/bin/env python3
"""
Anthropic API Model Interface for Reverse Gender Inference Detection

Provides interface to Anthropic models (Claude-3, etc.)
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import os

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None

from ..core.evaluator import ModelInterface

logger = logging.getLogger(__name__)


class AnthropicModel(ModelInterface):
    """
    Interface to Anthropic models.
    
    Supports Claude-4 Sonnet, Claude-3 Opus, Claude-3 Haiku, and other Anthropic models.
    """
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize Anthropic model interface.
        
        Args:
            model_name: Name of Anthropic model (e.g., "claude-sonnet-4-20250514")
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        super().__init__(f"anthropic_{model_name}")
        
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.anthropic_model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = AsyncAnthropic(api_key=api_key)
        
        logger.info(f"Initialized Anthropic model: {model_name}")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate response from Anthropic model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response text
        """
        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.anthropic_model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                ),
                timeout=self.timeout
            )
            
            return response.content[0].text.strip()
            
        except asyncio.TimeoutError:
            logger.error(f"Anthropic API timeout for model {self.anthropic_model_name}")
            raise
        except Exception as e:
            logger.error(f"Anthropic API error for model {self.anthropic_model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "provider": "Anthropic",
            "model": self.anthropic_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }


# Convenience functions for common models
def create_claude3_sonnet(api_key: Optional[str] = None, **kwargs) -> AnthropicModel:
    """Create Claude-4 Sonnet model interface."""
    return AnthropicModel("claude-sonnet-4-20250514", api_key=api_key, **kwargs)


def create_claude3_opus(api_key: Optional[str] = None, **kwargs) -> AnthropicModel:
    """Create Claude-3 Opus model interface."""
    return AnthropicModel("claude-3-opus-20240229", api_key=api_key, **kwargs)


def create_claude3_haiku(api_key: Optional[str] = None, **kwargs) -> AnthropicModel:
    """Create Claude-3 Haiku model interface."""
    return AnthropicModel("claude-3-haiku-20240307", api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        try:
            model = create_claude3_sonnet()
            
            prompt = """Based on the pronouns used in the following sentence, what gender identity does the person most likely have?

Sentence: "Alex was happy. They smiled."

Please choose the most appropriate answer:

A) Male
B) Female
C) Non-binary
D) Cannot be determined from pronouns alone

Answer (choose A, B, C, or D):"""
            
            response = await model.generate_response(prompt)
            print(f"Model: {model.model_name}")
            print(f"Response: {response}")
            print(f"Info: {model.get_model_info()}")
            
        except Exception as e:
            print(f"Demo failed: {e}")
    
    asyncio.run(demo())