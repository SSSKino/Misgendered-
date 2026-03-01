#!/usr/bin/env python3
"""
OpenAI API Model Interface for Reverse Gender Inference Detection

Provides interface to OpenAI models (GPT-3.5, GPT-4, etc.)
"""

import asyncio
import logging
import time
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


class OpenAIModel(ModelInterface):
    """
    Interface to OpenAI models.
    
    Supports GPT-3.5-turbo, GPT-4, and other OpenAI models.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize OpenAI model interface.
        
        Args:
            model_name: Name of OpenAI model (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        super().__init__(f"openai_{model_name}")
        
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.openai_model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Initialized OpenAI model: {model_name}")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate response from OpenAI model with rate limit handling.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response text
        """
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.openai_model_name,
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
                logger.error(f"OpenAI API timeout for model {self.openai_model_name}")
                raise
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors (429)
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        # Try to parse wait time from error message
                        try:
                            if "Please try again in" in error_str:
                                import re
                                match = re.search(r'Please try again in (\d+(?:\.\d+)?)([ms])', error_str)
                                if match:
                                    wait_value = float(match.group(1))
                                    unit = match.group(2)
                                    if unit == 's':
                                        wait_time = wait_value
                                    elif unit == 'ms':
                                        wait_time = wait_value / 1000
                                    else:  # assume seconds
                                        wait_time = wait_value
                        except:
                            pass  # Use exponential backoff
                        
                        logger.warning(f"Rate limit hit for {self.openai_model_name}, waiting {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries for model {self.openai_model_name}")
                        raise
                
                # Handle other API errors
                else:
                    logger.error(f"OpenAI API error for model {self.openai_model_name}: {e}")
                    raise
        
        # This should never be reached due to the raise statements above
        raise Exception(f"Failed to get response after {max_retries} attempts")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "model": self.openai_model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }


# Convenience functions for common models
def create_gpt35_turbo(api_key: Optional[str] = None, **kwargs) -> OpenAIModel:
    """Create GPT-3.5-turbo model interface."""
    return OpenAIModel("gpt-3.5-turbo", api_key=api_key, **kwargs)


def create_gpt4(api_key: Optional[str] = None, **kwargs) -> OpenAIModel:
    """Create GPT-4 model interface."""
    return OpenAIModel("gpt-4", api_key=api_key, **kwargs)


def create_gpt4_turbo(api_key: Optional[str] = None, **kwargs) -> OpenAIModel:
    """Create GPT-4-turbo model interface."""
    return OpenAIModel("gpt-4-turbo-preview", api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        try:
            model = create_gpt35_turbo()
            
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