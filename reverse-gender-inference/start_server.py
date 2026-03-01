#!/usr/bin/env python3
"""
Startup script for Reverse Gender Inference Detection System

This script provides a convenient way to start the web server with
proper environment setup and error handling.
"""

import os
import sys
import logging
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths."""
    
    # Add src directory to Python path
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Create required directories
    for dir_name in ["data", "config", "results"]:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
    
    # Create subdirectories for data
    (project_root / "data" / "templates").mkdir(exist_ok=True)
    (project_root / "data" / "names").mkdir(exist_ok=True)
    
    print("[OK] Environment setup completed")


def check_api_keys():
    """Check which API keys are available."""
    
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"), 
        "DashScope (Qwen)": os.getenv("DASHSCOPE_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API_KEY")
    }
    
    available_keys = []
    missing_keys = []
    
    for provider, key in api_keys.items():
        if key:
            available_keys.append(provider)
        else:
            missing_keys.append(provider)
    
    print("\n[API] API Key Status:")
    if available_keys:
        print(f"[OK] Available: {', '.join(available_keys)}")
    if missing_keys:
        print(f"[MISSING] Missing: {', '.join(missing_keys)}")
        print(f"  (Demo models will be used for missing providers)")
    
    return len(available_keys) > 0


def start_server():
    """Start the web server."""
    
    print("\nStarting Reverse Gender Inference Detection System")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check API keys
    has_real_models = check_api_keys()
    
    if not has_real_models:
        print("\n[WARNING] No API keys detected. Only demo models will be available.")
        print("   To use real models, set the following environment variables:")
        print("   - OPENAI_API_KEY for GPT models")
        print("   - ANTHROPIC_API_KEY for Claude models") 
        print("   - DASHSCOPE_API_KEY for Qwen models")
        print("   - DEEPSEEK_API_KEY for DeepSeek models")
    
    print("\n[INFO] Server Information:")
    print("   Web Interface: http://localhost:8099")
    print("   API Documentation: http://localhost:8099/docs")
    print("   Interactive API: http://localhost:8099/redoc")
    print("=" * 60)
    
    # Import and start the app
    try:
        import uvicorn
        from src.web.app import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8099,  # Changed port to avoid conflict
            reload=False,  # Set to True for development
            log_level="info"
        )
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   or: poetry install")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)