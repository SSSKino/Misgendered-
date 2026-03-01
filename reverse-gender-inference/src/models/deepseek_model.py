#!/usr/bin/env python3
"""
DeepSeek Model Interface for Reverse Gender Inference Detection

Provides interface to DeepSeek models via their API (OpenAI-compatible).
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import os

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None

from ..core.evaluator import ModelInterface

logger = logging.getLogger(__name__)


class DeepSeekModel(ModelInterface):
    """
    Interface to DeepSeek models.
    
    Supports DeepSeek-V3 and other DeepSeek models.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize DeepSeek model interface.
        
        Args:
            model_name: Name of DeepSeek model (e.g., "deepseek-chat", "deepseek-coder")
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        super().__init__(f"deepseek_{model_name}")
        
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.deepseek_model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize client with DeepSeek endpoint
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        logger.info(f"Initialized DeepSeek model: {model_name}")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate response from DeepSeek model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response text
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.deepseek_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=None
                ),
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            logger.error(f"DeepSeek API timeout for model {self.deepseek_model_name}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek API error for model {self.deepseek_model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "provider": "DeepSeek",
            "model": self.deepseek_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_endpoint": "https://api.deepseek.com"
        }


# Convenience functions for common DeepSeek models
def create_deepseek_chat(api_key: Optional[str] = None, **kwargs) -> DeepSeekModel:
    """Create DeepSeek-Chat model interface."""
    return DeepSeekModel("deepseek-chat", api_key=api_key, **kwargs)


def create_deepseek_coder(api_key: Optional[str] = None, **kwargs) -> DeepSeekModel:
    """Create DeepSeek-Coder model interface."""
    return DeepSeekModel("deepseek-coder", api_key=api_key, **kwargs)


def create_deepseek_v3(api_key: Optional[str] = None, **kwargs) -> DeepSeekModel:
    """Create DeepSeek-V3 model interface."""
    return DeepSeekModel("deepseek-chat", api_key=api_key, **kwargs)  # V3 uses deepseek-chat endpoint


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        try:
            model = create_deepseek_v3()
            
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
            print("Make sure DEEPSEEK_API_KEY environment variable is set")
    
    asyncio.run(demo())