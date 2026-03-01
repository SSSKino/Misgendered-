#!/usr/bin/env python3
"""
FastAPI Web Application for Reverse Gender Inference Detection

Provides web interface for configuring and running evaluations.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.evaluator import ReverseInferenceEvaluator
from ..core.types import EvaluationConfig, NameCategory, TestType
from ..models import (
    # Real model interfaces
    create_gpt35_turbo, create_gpt4, create_gpt4_turbo,
    create_claude3_sonnet, create_claude3_opus, create_claude3_haiku,
    create_qwen_turbo, create_qwen25_72b, create_qwen_max,
    create_deepseek_chat, create_deepseek_v3,
    # Demo models for testing
    create_random_model, create_male_biased_model,
    create_female_biased_model, create_correct_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Reverse Gender Inference Detection System",
    description="系统性地揭示大语言模型在性别认知方面的隐性偏见",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
evaluator: Optional[ReverseInferenceEvaluator] = None
current_evaluation: Dict[str, Any] = {}
evaluation_results: Dict[str, Any] = {}

# Results directory setup
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_evaluation_results(evaluation_id: str, results_data: Dict[str, Any]) -> None:
    """
    Save evaluation results to files.
    
    Args:
        evaluation_id: Unique evaluation identifier
        results_data: Complete evaluation results data
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        complete_file = RESULTS_DIR / f"{evaluation_id}_complete_results.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_data = {
            "evaluation_id": evaluation_id,
            "timestamp": results_data["completion_time"],
            "configuration": results_data["config"],
            "summary": {
                "total_models": len(results_data["results"]),
                "models_evaluated": list(results_data["results"].keys()),
                "average_accuracy": sum(
                    model_data["accuracy"] for model_data in results_data["results"].values()
                ) / len(results_data["results"]) if results_data["results"] else 0,
                "model_accuracies": {
                    model_name: model_data["accuracy"] 
                    for model_name, model_data in results_data["results"].items()
                },
                "best_model": max(
                    results_data["results"].items(), 
                    key=lambda x: x[1]["accuracy"]
                )[0] if results_data["results"] else None,
                "test_scale": results_data["config"]["test_scale"]
            }
        }
        
        summary_file = RESULTS_DIR / f"{evaluation_id}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save detailed results as CSV (if needed)
        try:
            import pandas as pd
            detailed_records = []
            
            for model_name, model_data in results_data["results"].items():
                # Add model-level summary row
                base_record = {
                    "evaluation_id": evaluation_id,
                    "model_name": model_name,
                    "total_cases": model_data["total_cases"],
                    "correct_predictions": model_data["correct_predictions"],
                    "accuracy": model_data["accuracy"],
                    "execution_time": model_data["execution_time"],
                    "bias_grade": model_data["bias_metrics"]["grade"],
                    "overall_bias_score": model_data["bias_metrics"]["overall_score"]
                }
                
                # Add pronoun-level breakdowns
                for pronoun, pronoun_data in model_data["results_by_pronoun"].items():
                    record = base_record.copy()
                    record.update({
                        "category_type": "pronoun",
                        "category_name": pronoun,
                        "category_total": pronoun_data["total"],
                        "category_correct": pronoun_data["correct"],
                        "category_accuracy": pronoun_data["accuracy"]
                    })
                    detailed_records.append(record)
                
                # Add combination-level breakdowns
                for combination, combo_data in model_data["results_by_combination"].items():
                    record = base_record.copy()
                    record.update({
                        "category_type": "combination",
                        "category_name": combination,
                        "category_total": combo_data["total"],
                        "category_correct": combo_data["correct"],
                        "category_accuracy": combo_data["accuracy"]
                    })
                    detailed_records.append(record)
            
            if detailed_records:
                df = pd.DataFrame(detailed_records)
                csv_file = RESULTS_DIR / f"{evaluation_id}_detailed_results.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                
        except ImportError:
            # pandas not available, skip CSV generation
            logger.warning("pandas not available, skipping CSV generation")
        except Exception as e:
            logger.warning(f"Failed to generate CSV file: {e}")
        
        logger.info(f"Saved evaluation results to {RESULTS_DIR}")
        logger.info(f"  - Complete results: {complete_file.name}")
        logger.info(f"  - Summary: {summary_file.name}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}", exc_info=True)

# Pydantic models for API
class EvaluationRequest(BaseModel):
    selected_models: List[str]
    api_keys: Dict[str, str] = {}
    test_scale: int = 19800
    random_seed: int = 42
    batch_size: int = 20  # Reduced from 50 to lower concurrency
    max_concurrent: int = 3  # Reduced from 10 to respect rate limits
    timeout: float = 60.0  # Increased timeout for retry delays

class ProgressUpdate(BaseModel):
    phase: str
    current: int
    total: int
    message: str
    percentage: float

class ModelInfo(BaseModel):
    name: str
    provider: str
    description: str
    available: bool


@app.on_event("startup")
async def startup():
    """Initialize the application."""
    global evaluator
    
    logger.info("Starting Reverse Gender Inference Detection System")
    
    # Initialize evaluator
    evaluator = ReverseInferenceEvaluator(
        data_dir=Path("data"),
        config_dir=Path("config"),
        results_dir=Path("results")
    )
    
    # Register real models (if API keys are available)
    real_models = []
    
    # OpenAI models (matching original project naming)
    try:
        from ..models.openai_model import OpenAIModel
        gpt4o = OpenAIModel("gpt-4o-2024-08-06")
        gpt4o.model_name = "gpt-4o"  # Override to match original naming
        real_models.append(gpt4o)
        logger.info("Registered OpenAI models")
    except Exception as e:
        logger.warning(f"Failed to register OpenAI models: {e}")
    
    # Anthropic models (matching original project naming)
    try:
        from ..models.anthropic_model import AnthropicModel
        claude4 = AnthropicModel("claude-sonnet-4-20250514")
        claude4.model_name = "claude-4-sonnet"  # Override to match original naming
        real_models.append(claude4)
        logger.info("Registered Anthropic models")
    except Exception as e:
        logger.warning(f"Failed to register Anthropic models: {e}")
    
    # Qwen models (matching original project naming)
    try:
        from ..models.qwen_model import QwenModel
        qwen_turbo = QwenModel("qwen-turbo-latest")
        qwen_turbo.model_name = "qwen-turbo"  # Override to match original naming
        
        qwen_25_72b = QwenModel("qwen2.5-72b-instruct")
        qwen_25_72b.model_name = "qwen-2.5-72b"  # Override to match original naming
        
        real_models.extend([qwen_turbo, qwen_25_72b])
        logger.info("Registered Qwen models")
    except Exception as e:
        logger.warning(f"Failed to register Qwen models: {e}")
    
    # DeepSeek models (matching original project naming)
    try:
        from ..models.deepseek_model import DeepSeekModel
        deepseek_v3 = DeepSeekModel("deepseek-chat")
        deepseek_v3.model_name = "deepseek-v3"  # Override to match original naming
        real_models.append(deepseek_v3)
        logger.info("Registered DeepSeek models")
    except Exception as e:
        logger.warning(f"Failed to register DeepSeek models: {e}")
    
    # Register successfully created models
    for model in real_models:
        try:
            evaluator.register_model(model)
            logger.info(f"Registered real model: {model.model_name}")
        except Exception as e:
            logger.warning(f"Failed to register model {model.model_name}: {e}")
    
    # No placeholder or demo models - only real models with API keys
    if not real_models:
        logger.info("No API keys available. No models registered - users must provide API keys to use the system.")
    
    logger.info(f"Application startup completed with {len(evaluator.models)} models registered")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reverse Gender Inference Detection</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>反向性别推理检测系统</h1>
            <p>Web interface is being prepared...</p>
            <p>API documentation: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)


@app.get("/api/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available models."""
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    models = []
    for model_name, model in evaluator.models.items():
        info = model.get_model_info()
        models.append(ModelInfo(
            name=model_name,
            provider=info.get("provider", "Unknown"),
            description=f"{info.get('provider', 'Unknown')} - {info.get('model', model_name)}",
            available=True
        ))
    
    return models


@app.get("/api/config")
async def get_evaluation_config():
    """Get current evaluation configuration."""
    return {
        "available_models": list(evaluator.models.keys()) if evaluator else [],
        "default_config": {
            "test_scale": 19800,
            "random_seed": 42,
            "batch_size": 50,
            "max_concurrent": 10,
            "timeout": 30.0
        },
        "data_stats": evaluator.data_generator.get_generation_stats() if evaluator else {}
    }


@app.post("/api/models/register")
async def register_models(api_keys: Dict[str, str]):
    """Register models with provided API keys (separate from evaluation)."""
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    logger.info(f"Received request to register models with API keys: {list(api_keys.keys())}")
    new_models_count = register_models_with_api_keys(api_keys)
    
    return {
        "message": f"Successfully registered {new_models_count} models",
        "new_models_count": new_models_count,
        "available_models": list(evaluator.models.keys())
    }


@app.post("/api/models/clear")
async def clear_all_models():
    """Clear all registered models."""
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    model_count = len(evaluator.models)
    evaluator.models.clear()
    logger.info(f"Cleared all {model_count} registered models")
    
    return {
        "message": f"Cleared {model_count} models",
        "cleared_count": model_count
    }


@app.post("/api/models/test")
async def test_model_connection(api_keys: Dict[str, str]):
    """Test API key connections without registering models."""
    results = {}
    
    if "DEEPSEEK_API_KEY" in api_keys:
        try:
            from ..models.deepseek_model import DeepSeekModel
            test_model = DeepSeekModel("deepseek-chat", api_key=api_keys["DEEPSEEK_API_KEY"])
            # Try a simple test call
            test_response = await test_model.generate("Test message", max_tokens=5)
            results["deepseek"] = {"status": "success", "response": test_response}
        except Exception as e:
            results["deepseek"] = {"status": "failed", "error": str(e)}
    
    return results


def register_models_with_api_keys(api_keys: Dict[str, str]):
    """Register models dynamically with provided API keys."""
    global evaluator
    
    new_models = []
    registered_models = []
    
    # Track which models are already registered to avoid duplicates
    existing_models = set(evaluator.models.keys()) if evaluator else set()
    
    logger.info(f"Current models before registration: {list(existing_models)}")
    logger.info(f"Attempting to register models with API keys: {list(api_keys.keys())}")
    
    # OpenAI models
    if "OPENAI_API_KEY" in api_keys and "gpt-4o" not in existing_models:
        try:
            from ..models.openai_model import OpenAIModel
            gpt4o = OpenAIModel("gpt-4o-2024-08-06", api_key=api_keys["OPENAI_API_KEY"])
            gpt4o.model_name = "gpt-4o"
            new_models.append(gpt4o)
            logger.info("Prepared OpenAI model for registration")
        except Exception as e:
            logger.warning(f"Failed to create OpenAI model: {e}")
    
    # Anthropic models
    if "ANTHROPIC_API_KEY" in api_keys and "claude-4-sonnet" not in existing_models:
        try:
            from ..models.anthropic_model import AnthropicModel
            claude4 = AnthropicModel("claude-sonnet-4-20250514", api_key=api_keys["ANTHROPIC_API_KEY"])
            claude4.model_name = "claude-4-sonnet"
            new_models.append(claude4)
            logger.info("Prepared Anthropic model for registration")
        except Exception as e:
            logger.warning(f"Failed to create Anthropic model: {e}")
    
    # Qwen models
    if "DASHSCOPE_API_KEY" in api_keys:
        if "qwen-turbo" not in existing_models:
            try:
                from ..models.qwen_model import QwenModel
                qwen_turbo = QwenModel("qwen-turbo-latest", api_key=api_keys["DASHSCOPE_API_KEY"])
                qwen_turbo.model_name = "qwen-turbo"
                new_models.append(qwen_turbo)
                logger.info("Prepared Qwen Turbo model for registration")
            except Exception as e:
                logger.warning(f"Failed to create Qwen Turbo model: {e}")
        
        if "qwen-2.5-72b" not in existing_models:
            try:
                from ..models.qwen_model import QwenModel
                qwen_25_72b = QwenModel("qwen2.5-72b-instruct", api_key=api_keys["DASHSCOPE_API_KEY"])
                qwen_25_72b.model_name = "qwen-2.5-72b"
                new_models.append(qwen_25_72b)
                logger.info("Prepared Qwen 2.5-72B model for registration")
            except Exception as e:
                logger.warning(f"Failed to create Qwen 2.5-72B model: {e}")
    
    # DeepSeek models
    if "DEEPSEEK_API_KEY" in api_keys and "deepseek-v3" not in existing_models:
        try:
            from ..models.deepseek_model import DeepSeekModel
            deepseek_v3 = DeepSeekModel("deepseek-chat", api_key=api_keys["DEEPSEEK_API_KEY"])
            # Override the automatically generated name
            deepseek_v3.model_name = "deepseek-v3"
            new_models.append(deepseek_v3)
            logger.info(f"Prepared DeepSeek model for registration: {deepseek_v3.model_name}")
        except Exception as e:
            logger.warning(f"Failed to create DeepSeek model: {e}")
    
    # Clean up and register real models
    for model in new_models:
        try:
            # Remove any existing model with the same name (placeholder or real)
            if model.model_name in evaluator.models:
                logger.info(f"Removing existing model before registration: {model.model_name}")
                del evaluator.models[model.model_name]
            
            # Also remove any similar demo models
            models_to_remove = []
            for existing_name in evaluator.models.keys():
                if (existing_name.startswith(f"{model.model_name}-") or 
                    existing_name.endswith(f"-{model.model_name}") or
                    f"demo_{model.model_name}" == existing_name):
                    models_to_remove.append(existing_name)
            
            for remove_name in models_to_remove:
                logger.info(f"Removing similar model: {remove_name}")
                del evaluator.models[remove_name]
            
            # Register the real model
            evaluator.register_model(model)
            registered_models.append(model.model_name)
            logger.info(f"Successfully registered real model: {model.model_name}")
        except Exception as e:
            logger.warning(f"Failed to register model {model.model_name}: {e}")
    
    logger.info(f"Final models after registration: {list(evaluator.models.keys())}")
    
    if registered_models:
        logger.info(f"Successfully registered {len(registered_models)} models: {', '.join(registered_models)}")
    
    return len(registered_models)


@app.post("/api/evaluate")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start model evaluation."""
    global current_evaluation
    
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    # Strong concurrent control - check again with lock-like behavior
    if current_evaluation.get("status") == "running":
        logger.warning(f"Evaluation request rejected - already running: {current_evaluation.get('id')}")
        raise HTTPException(
            status_code=400, 
            detail=f"Evaluation already in progress (ID: {current_evaluation.get('id')}). Please wait for completion or stop the current evaluation."
        )
    
    # Validate selected models first (before any registration)
    if not request.selected_models:
        raise HTTPException(status_code=400, detail="No models selected for evaluation")
    
    # Log the request for debugging
    logger.info(f"Evaluation request received for models: {request.selected_models}")
    
    # Only register models if they don't already exist or if we have no models at all
    should_register = False
    if request.api_keys:
        # Check if we need to register any models
        for api_key, value in request.api_keys.items():
            if api_key == "OPENAI_API_KEY" and "gpt-4o" not in evaluator.models:
                should_register = True
            elif api_key == "ANTHROPIC_API_KEY" and "claude-4-sonnet" not in evaluator.models:
                should_register = True
            elif api_key == "DASHSCOPE_API_KEY" and ("qwen-turbo" not in evaluator.models or "qwen-2.5-72b" not in evaluator.models):
                should_register = True
            elif api_key == "DEEPSEEK_API_KEY" and "deepseek-v3" not in evaluator.models:
                should_register = True
        
        if should_register:
            logger.info("Registering new models with provided API keys")
            new_models_count = register_models_with_api_keys(request.api_keys)
            logger.info(f"Registered {new_models_count} new models")
        else:
            logger.info("All required models already registered, skipping model registration")
    
    # Validate selected models after registration
    unavailable_models = []
    available_models = list(evaluator.models.keys())
    
    for model_name in request.selected_models:
        if model_name not in evaluator.models:
            unavailable_models.append(model_name)
        else:
            # Log which model will actually be used
            model_info = evaluator.models[model_name].get_model_info()
            logger.info(f"Will evaluate model: {model_name} (type: {model_info.get('provider', 'unknown')})")
    
    if unavailable_models:
        logger.error(f"Models not found: {unavailable_models}. Available: {available_models}")
        raise HTTPException(
            status_code=400, 
            detail=f"Models not available: {', '.join(unavailable_models)}. Available models: {available_models}"
        )
    
    # Final check before starting
    if current_evaluation.get("status") == "running":
        raise HTTPException(status_code=400, detail="Evaluation started by another request")
    
    # Create evaluation config
    config = EvaluationConfig(
        selected_models=request.selected_models,
        test_scale=request.test_scale,
        random_seed=request.random_seed,
        batch_size=request.batch_size,
        max_concurrent=request.max_concurrent,
        timeout=request.timeout
    )
    
    # Initialize evaluation state with more details
    evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_evaluation = {
        "id": evaluation_id,
        "status": "running",
        "selected_models": request.selected_models,  # Store for verification
        "config": config.dict() if hasattr(config, 'dict') else config.__dict__,
        "start_time": datetime.now().isoformat(),
        "progress": {"phase": "starting", "current": 0, "total": 1, "message": "Initializing evaluation..."}
    }
    
    logger.info(f"Starting evaluation {evaluation_id} with models: {request.selected_models}")
    
    # Start evaluation in background
    background_tasks.add_task(run_evaluation_background, config, evaluation_id)
    
    return {"evaluation_id": evaluation_id, "status": "started"}


@app.post("/api/evaluation/stop")
async def stop_evaluation():
    """Stop current evaluation."""
    global current_evaluation
    
    if current_evaluation and current_evaluation.get("status") == "running":
        current_evaluation["status"] = "stopped"
        current_evaluation["end_time"] = datetime.now().isoformat()
        logger.info("Evaluation stopped by user request")
        return {"message": "Evaluation stopped", "evaluation_id": current_evaluation.get("id")}
    else:
        return {"message": "No running evaluation found"}


async def run_evaluation_background(config: EvaluationConfig, evaluation_id: str):
    """Run evaluation in background."""
    global current_evaluation, evaluation_results
    
    try:
        # Check if evaluation was cancelled before starting
        if current_evaluation.get("status") != "running":
            logger.info(f"Evaluation {evaluation_id} was cancelled before starting")
            return
        
        def progress_callback(phase: str, current: int, total: int, message: str):
            # Check if evaluation was stopped during execution
            if current_evaluation.get("status") != "running":
                logger.info(f"Evaluation {evaluation_id} stopped during execution")
                raise Exception("Evaluation stopped by user")
            
            percentage = (current / total * 100) if total > 0 else 0
            current_evaluation["progress"] = {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "percentage": percentage
            }
            logger.debug(f"Evaluation {evaluation_id} progress: {phase} {current}/{total} ({percentage:.1f}%)")
        
        # Log start of evaluation
        logger.info(f"Starting background evaluation {evaluation_id} for models: {config.selected_models}")
        
        # Run evaluation
        results = await evaluator.run_evaluation(config, progress_callback)
        
        # Check if still running (not stopped during execution)
        if current_evaluation.get("status") != "running":
            logger.info(f"Evaluation {evaluation_id} was stopped during execution")
            return
        
        # Store results
        results_data = {
            "config": config.__dict__,
            "results": {name: summary.to_dict() for name, summary in results.items()},
            "completion_time": datetime.now().isoformat()
        }
        evaluation_results[evaluation_id] = results_data
        
        # Save results to files
        try:
            save_evaluation_results(evaluation_id, results_data)
            logger.info(f"Successfully saved evaluation results to files")
        except Exception as e:
            logger.error(f"Failed to save evaluation results to files: {e}", exc_info=True)
        
        # Update status to completed
        current_evaluation["status"] = "completed"
        current_evaluation["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Evaluation {evaluation_id} completed successfully with {len(results)} model results")
        
    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}", exc_info=True)
        
        # Only update status if it's still running (not already stopped)
        if current_evaluation.get("status") == "running":
            current_evaluation["status"] = "failed"
            current_evaluation["error"] = str(e)
            current_evaluation["end_time"] = datetime.now().isoformat()


@app.get("/api/evaluation/status")
async def get_evaluation_status():
    """Get current evaluation status."""
    return current_evaluation


@app.get("/api/evaluation/results/{evaluation_id}")
async def get_evaluation_results(evaluation_id: str):
    """Get evaluation results."""
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    return evaluation_results[evaluation_id]


@app.get("/api/evaluation/results")
async def list_evaluation_results():
    """List all evaluation results."""
    return {
        "evaluations": [
            {
                "id": eval_id,
                "completion_time": results["completion_time"],
                "model_count": len(results["results"]),
                "test_scale": results["config"]["test_scale"]
            }
            for eval_id, results in evaluation_results.items()
        ]
    }




# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    print("Starting Reverse Gender Inference Detection System")
    print("=" * 50)
    print("Web interface: http://localhost:8095")
    print("API documentation: http://localhost:8095/docs")
    print("=" * 50)
    
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8095,
        reload=True,
        log_level="info"
    )