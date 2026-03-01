#!/usr/bin/env python3
"""
Qwen Model Interface for Reverse Gender Inference Detection

Provides interface to Qwen models via DashScope API (OpenAI-compatible).
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


class QwenModel(ModelInterface):
    """
    Interface to Qwen models via DashScope API.
    
    Supports Qwen-Turbo, Qwen2.5-72B, and other Qwen models.
    """
    
    def __init__(
        self,
        model_name: str = "qwen-turbo-latest",
        api_key: Optional[str] = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize Qwen model interface.
        
        Args:
            model_name: Name of Qwen model (e.g., "qwen-turbo-latest", "qwen2.5-72b-instruct")
            api_key: DashScope API key. If None, reads from DASHSCOPE_API_KEY env var
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        super().__init__(f"qwen_{model_name}")
        
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.qwen_model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize client with DashScope endpoint
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DashScope API key not provided. Set DASHSCOPE_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        logger.info(f"Initialized Qwen model: {model_name}")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate response from Qwen model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response text
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.qwen_model_name,
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
            logger.error(f"Qwen API timeout for model {self.qwen_model_name}")
            raise
        except Exception as e:
            logger.error(f"Qwen API error for model {self.qwen_model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "provider": "Qwen/DashScope",
            "model": self.qwen_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }


# Convenience functions for common Qwen models
def create_qwen_turbo(api_key: Optional[str] = None, **kwargs) -> QwenModel:
    """Create Qwen-Turbo model interface."""
    return QwenModel("qwen-turbo-latest", api_key=api_key, **kwargs)


def create_qwen25_72b(api_key: Optional[str] = None, **kwargs) -> QwenModel:
    """Create Qwen2.5-72B model interface."""
    return QwenModel("qwen2.5-72b-instruct", api_key=api_key, **kwargs)


def create_qwen_max(api_key: Optional[str] = None, **kwargs) -> QwenModel:
    """Create Qwen-Max model interface."""
    return QwenModel("qwen-max", api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        try:
            model = create_qwen_turbo()
            
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
            print("Make sure DASHSCOPE_API_KEY environment variable is set")
    
    asyncio.run(demo())