"""Model interfaces for various LLM APIs."""

from .openai_model import OpenAIModel, create_gpt35_turbo, create_gpt4, create_gpt4_turbo
from .anthropic_model import AnthropicModel, create_claude3_sonnet, create_claude3_opus, create_claude3_haiku
from .qwen_model import QwenModel, create_qwen_turbo, create_qwen25_72b, create_qwen_max
from .deepseek_model import DeepSeekModel, create_deepseek_chat, create_deepseek_v3

__all__ = [
    # Base model classes
    "OpenAIModel",
    "AnthropicModel",
    "QwenModel",
    "DeepSeekModel",

    # OpenAI convenience functions
    "create_gpt35_turbo",
    "create_gpt4",
    "create_gpt4_turbo",

    # Anthropic convenience functions
    "create_claude3_sonnet",
    "create_claude3_opus",
    "create_claude3_haiku",

    # Qwen convenience functions
    "create_qwen_turbo",
    "create_qwen25_72b",
    "create_qwen_max",

    # DeepSeek convenience functions
    "create_deepseek_chat",
    "create_deepseek_v3",
]