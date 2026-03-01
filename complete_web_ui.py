#!/usr/bin/env python3
"""
Complete Web UI for Pronoun Evaluation
Full-featured interface with all required functionality
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

from complete_pronoun_eval import (
    PronounEvaluator, PromptStrategy, MODEL_CONFIGS, create_model,
    EvaluationResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_ui.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Complete Pronoun Evaluation System")

# Setup static files and templates
templates = Jinja2Templates(directory="web_templates")

# Global storage for running tasks
running_tasks: Dict[str, Dict[str, Any]] = {}

class EvaluationRequest(BaseModel):
    selected_models: List[str]
    api_keys: Dict[str, str]
    test_limit: int = 11000
    strategies: List[str] = ["zero_shot", "in_context_learning"]

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: int  # 0-100
    current_model: Optional[str] = None
    current_strategy: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": MODEL_CONFIGS,
        "strategies": [
            {"value": "zero_shot", "name": "Zero-Shot Evaluation (Original MISGENDERED Method)"},
            {"value": "in_context_learning", "name": "In-Context Learning"},
        ]
    })

@app.post("/api/start_evaluation")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start evaluation task"""
    logger.info(f"Received evaluation request: models={request.selected_models}, test_limit={request.test_limit}")
    
    # Validate input
    if not request.selected_models:
        raise HTTPException(status_code=400, detail="Please select at least one model")
    
    # Check API keys
    missing_keys = []
    for model_name in request.selected_models:
        if model_name not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        required_key = MODEL_CONFIGS[model_name]["api_key_name"]
        if required_key not in request.api_keys or not request.api_keys[required_key]:
            missing_keys.append(required_key)
    
    if missing_keys:
        raise HTTPException(status_code=400, detail=f"Missing API keys: {', '.join(missing_keys)}")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    logger.info(f"Generated task ID: {task_id}")
    
    # Create task status
    running_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "current_model": None,
        "current_strategy": None,
        "results": None,
        "error": None,
        "started_at": datetime.now(),
        "completed_at": None,
        "total_evaluations": len(request.selected_models) * len(request.strategies)
    }
    
    # Start background task
    background_tasks.add_task(
        run_evaluation_task,
        task_id,
        request.selected_models,
        request.api_keys,
        request.strategies,
        request.test_limit
    )
    
    return {"task_id": task_id, "message": "Evaluation task started"}

@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return running_tasks[task_id]

@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Get evaluation results"""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = running_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")
    
    return task["results"]

@app.get("/api/detailed_analysis/{task_id}")
async def get_detailed_analysis(task_id: str):
    """Get detailed analysis results"""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = running_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")
    
    results = task.get("results", [])
    
    # Generate detailed analysis
    detailed_analysis = {
        "summary": _generate_summary_analysis(results),
        "bias_analysis": _generate_bias_analysis(results),
        "error_analysis": _generate_error_analysis(results),
        "comparative_analysis": _generate_comparative_analysis(results)
    }
    
    return detailed_analysis

@app.get("/api/export_results/{task_id}")
async def export_results(task_id: str, format: str = "json"):
    """Export results in different formats"""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = running_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")
    
    results = task.get("results", [])
    
    if format.lower() == "csv":
        csv_data = _convert_to_csv(results)
        return JSONResponse({
            "format": "csv",
            "data": csv_data,
            "filename": f"pronoun_eval_results_{task_id}.csv"
        })
    else:
        return {
            "format": "json",
            "data": results,
            "filename": f"pronoun_eval_results_{task_id}.json"
        }

@app.get("/api/history")
async def get_task_history():
    """Get task history"""
    tasks = []
    for task_id, task in running_tasks.items():
        if task["status"] in ["completed", "failed"]:
            tasks.append({
                "task_id": task_id,
                "timestamp": task["started_at"].isoformat(),
                "status": task["status"],
                "models": getattr(task, 'models', []),
                "strategies": getattr(task, 'strategies', [])
            })
    
    tasks.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"tasks": tasks}

async def run_evaluation_task(
    task_id: str,
    selected_models: List[str],
    api_keys: Dict[str, str],
    strategies: List[str],
    test_limit: int
):
    """Run evaluation task in background"""
    logger.info(f"Starting evaluation task {task_id}")
    
    try:
        task = running_tasks[task_id]
        evaluator = PronounEvaluator()
        
        results = []
        total_evaluations = len(selected_models) * len(strategies)
        current_evaluation = 0
        
        for model_name in selected_models:
            task["current_model"] = model_name
            logger.info(f"Processing model: {model_name}")
            
            # Create model instance
            try:
                model = create_model(model_name, api_keys)
            except Exception as e:
                task["status"] = "failed"
                task["error"] = f"Failed to create model {model_name}: {str(e)}"
                logger.error(f"Failed to create model {model_name}: {e}")
                return
            
            # Evaluate each strategy
            for strategy_name in strategies:
                task["current_strategy"] = strategy_name
                current_evaluation += 1
                
                try:
                    # Map strategy name
                    strategy = PromptStrategy.ZERO_SHOT if strategy_name == "zero_shot" else PromptStrategy.IN_CONTEXT_LEARNING
                    
                    logger.info(f"Evaluating {model_name} with {strategy.value}")
                    
                    # Define progress callback for individual model-strategy evaluation
                    def progress_callback(completed: int, total: int):
                        # Calculate overall progress including previous evaluations
                        # Each model-strategy pair gets equal weight in overall progress
                        strategy_progress = (completed / total) * 100 if total > 0 else 0
                        base_progress = ((current_evaluation - 1) / total_evaluations) * 100
                        current_strategy_weight = (1 / total_evaluations) * 100
                        overall_progress = base_progress + (strategy_progress / 100) * current_strategy_weight
                        
                        task["progress"] = int(overall_progress)
                        task["current_test_case"] = f"{completed}/{total}"
                        logger.info(f"Model {model_name} ({strategy_name}): {completed}/{total} cases, Overall: {overall_progress:.1f}%")
                    
                    # Run evaluation with progress callback
                    result = await evaluator.evaluate_model(
                        model=model,
                        strategy=strategy,
                        test_limit=test_limit,
                        progress_callback=progress_callback
                    )
                    
                    # Convert result to dict
                    result_data = {
                        "model_name": result.model_name,
                        "model_description": MODEL_CONFIGS[model_name]["description"],
                        "strategy": result.strategy.value,
                        "accuracy": result.accuracy,
                        "correct_predictions": result.correct_predictions,
                        "total_cases": result.total_cases,
                        "execution_time": result.execution_time,
                        "results_by_pronoun": result.results_by_pronoun,
                        "results_by_form": result.results_by_form,
                        "error_cases": result.error_cases,
                        "raw_responses": result.raw_responses  # Save all responses
                    }
                    
                    results.append(result_data)
                    
                    # Save individual result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"results_{model_name}_{strategy_name}_{timestamp}.json"
                    evaluator.save_results(result, filename)
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name} with {strategy_name}: {e}")
                    error_result = {
                        "model_name": model_name,
                        "strategy": strategy_name,
                        "error": str(e),
                        "accuracy": 0.0
                    }
                    results.append(error_result)
                
                # Update progress
                progress = int((current_evaluation / total_evaluations) * 100)
                task["progress"] = progress
                logger.info(f"Progress: {progress}% ({current_evaluation}/{total_evaluations})")
        
        # Task completed
        task["status"] = "completed"
        task["progress"] = 100
        task["completed_at"] = datetime.now()
        task["results"] = results
        task["models"] = selected_models
        task["strategies"] = strategies
        
        # Save complete task results
        await save_complete_task_results(task_id, task)
        
        logger.info(f"Task {task_id} completed successfully with {len(results)} evaluations")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task["status"] = "failed"
        task["error"] = str(e)
        task["completed_at"] = datetime.now()

async def save_complete_task_results(task_id: str, task: Dict[str, Any]):
    """Save complete task results to file"""
    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save main results
        main_file = results_dir / f"task_{task_id}_complete_results.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(task, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary
        summary_file = results_dir / f"task_{task_id}_summary.json"
        summary_data = {
            "task_id": task_id,
            "timestamp": task["started_at"].isoformat(),
            "status": task["status"],
            "models": task.get("models", []),
            "strategies": task.get("strategies", []),
            "total_evaluations": len(task.get("results", [])),
            "average_accuracy": _calculate_average_accuracy(task.get("results", []))
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete task results saved: {main_file}, {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save complete task results for {task_id}: {e}")

def _generate_summary_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary analysis"""
    if not results:
        return {}
    
    successful_results = [r for r in results if "error" not in r]
    
    return {
        "total_evaluations": len(results),
        "successful_evaluations": len(successful_results),
        "failed_evaluations": len(results) - len(successful_results),
        "average_accuracy": sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0,
        "best_performing_model": max(successful_results, key=lambda x: x["accuracy"])["model_name"] if successful_results else None,
        "accuracy_by_model": {r["model_name"]: r["accuracy"] for r in successful_results},
        "accuracy_by_strategy": _group_by_strategy(successful_results)
    }

def _generate_bias_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate bias analysis"""
    bias_patterns = {}
    
    for result in results:
        if "error" in result:
            continue
        
        # Analyze pronoun-specific performance
        for pronoun, stats in result.get("results_by_pronoun", {}).items():
            if pronoun not in bias_patterns:
                bias_patterns[pronoun] = []
            bias_patterns[pronoun].append(stats["accuracy"])
    
    # Calculate average performance per pronoun
    pronoun_averages = {}
    for pronoun, accuracies in bias_patterns.items():
        pronoun_averages[pronoun] = sum(accuracies) / len(accuracies) if accuracies else 0
    
    return {
        "pronoun_performance": pronoun_averages,
        "bias_score": max(pronoun_averages.values()) - min(pronoun_averages.values()) if pronoun_averages else 0,
        "most_difficult_pronouns": sorted(pronoun_averages.items(), key=lambda x: x[1])[:3],
        "easiest_pronouns": sorted(pronoun_averages.items(), key=lambda x: x[1], reverse=True)[:3]
    }

def _generate_error_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate error analysis"""
    all_errors = []
    
    for result in results:
        if "error_cases" in result:
            all_errors.extend(result["error_cases"])
    
    # Analyze error patterns
    error_patterns = {}
    for error in all_errors:
        pattern = f"{error['pronoun_type']}->{error['predicted']}"
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    
    return {
        "total_errors": len(all_errors),
        "common_error_patterns": sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10],
        "errors_by_pronoun": _group_errors_by_pronoun(all_errors),
        "errors_by_form": _group_errors_by_form(all_errors)
    }

def _generate_comparative_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparative analysis"""
    model_comparison = {}
    strategy_comparison = {}
    
    for result in results:
        if "error" in result:
            continue
        
        model_name = result["model_name"]
        strategy = result["strategy"]
        accuracy = result["accuracy"]
        
        if model_name not in model_comparison:
            model_comparison[model_name] = []
        model_comparison[model_name].append(accuracy)
        
        if strategy not in strategy_comparison:
            strategy_comparison[strategy] = []
        strategy_comparison[strategy].append(accuracy)
    
    # Calculate averages
    model_averages = {model: sum(accs)/len(accs) for model, accs in model_comparison.items()}
    strategy_averages = {strategy: sum(accs)/len(accs) for strategy, accs in strategy_comparison.items()}
    
    return {
        "model_ranking": sorted(model_averages.items(), key=lambda x: x[1], reverse=True),
        "strategy_effectiveness": strategy_averages,
        "performance_improvement": _calculate_improvement(results)
    }

def _convert_to_csv(results: List[Dict[str, Any]]) -> str:
    """Convert results to CSV format"""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "Model", "Strategy", "Accuracy", "Correct", "Total", 
        "Execution Time", "Status"
    ])
    
    # Write data
    for result in results:
        writer.writerow([
            result.get("model_name", ""),
            result.get("strategy", ""),
            result.get("accuracy", 0),
            result.get("correct_predictions", 0),
            result.get("total_cases", 0),
            result.get("execution_time", 0),
            "Error" if "error" in result else "Success"
        ])
    
    return output.getvalue()

def _group_by_strategy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Group results by strategy"""
    strategy_groups = {}
    for result in results:
        strategy = result["strategy"]
        if strategy not in strategy_groups:
            strategy_groups[strategy] = []
        strategy_groups[strategy].append(result["accuracy"])
    
    return {strategy: sum(accs)/len(accs) for strategy, accs in strategy_groups.items()}

def _group_errors_by_pronoun(errors: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group errors by pronoun type"""
    pronoun_errors = {}
    for error in errors:
        pronoun = error.get("pronoun_type", "unknown")
        pronoun_errors[pronoun] = pronoun_errors.get(pronoun, 0) + 1
    return pronoun_errors

def _group_errors_by_form(errors: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group errors by grammatical form"""
    form_errors = {}
    for error in errors:
        form = error.get("form", "unknown")
        form_errors[form] = form_errors.get(form, 0) + 1
    return form_errors

def _calculate_average_accuracy(results: List[Dict[str, Any]]) -> float:
    """Calculate average accuracy across all results"""
    successful_results = [r for r in results if "error" not in r]
    if not successful_results:
        return 0.0
    return sum(r["accuracy"] for r in successful_results) / len(successful_results)

def _calculate_improvement(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate improvement from zero-shot to in-context learning"""
    improvements = {}
    
    # Group by model
    model_results = {}
    for result in results:
        if "error" in result:
            continue
        model = result["model_name"]
        strategy = result["strategy"]
        if model not in model_results:
            model_results[model] = {}
        model_results[model][strategy] = result["accuracy"]
    
    # Calculate improvements
    for model, strategies in model_results.items():
        if "zero_shot" in strategies and "in_context_learning" in strategies:
            improvement = strategies["in_context_learning"] - strategies["zero_shot"]
            improvements[model] = improvement
    
    return improvements

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8094)