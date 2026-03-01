#!/usr/bin/env python3
"""
Core Evaluation Engine for Reverse Gender Inference Detection

Orchestrates data generation, model evaluation, and bias analysis.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .types import (
    ReverseTestCase, ModelResponse, EvaluationResult, BiasMetrics,
    ModelEvaluationSummary, EvaluationConfig, GenderChoice, TestType,
    NameCategory
)
from .seed_manager import SeedManager
from .prompt_builder import PromptBuilder
from ..data.generator import DataGenerator

logger = logging.getLogger(__name__)


class ModelInterface:
    """Base interface for language models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response from model."""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {"name": self.model_name}


class ReverseInferenceEvaluator:
    """
    Main evaluation engine for reverse gender inference testing.
    
    Coordinates data generation, model evaluation, and bias analysis.
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        config_dir: Path = Path("config"),
        results_dir: Path = Path("results")
    ):
        """
        Initialize evaluator.
        
        Args:
            data_dir: Directory containing templates and names
            config_dir: Directory for configuration files
            results_dir: Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        for directory in [self.data_dir, self.config_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.seed_manager = SeedManager(self.config_dir)
        self.data_generator = DataGenerator(
            templates_dir=self.data_dir / "templates",
            names_dir=self.data_dir / "names",
            seed_manager=self.seed_manager
        )
        self.prompt_builder = PromptBuilder()
        
        # State
        self.test_cases: List[ReverseTestCase] = []
        self.models: Dict[str, ModelInterface] = {}
        
        logger.info(f"Initialized evaluator with data_dir={data_dir}")
    
    def register_model(self, model: ModelInterface) -> None:
        """
        Register a model for evaluation.
        
        Args:
            model: Model interface instance
        """
        self.models[model.model_name] = model
        logger.info(f"Registered model: {model.model_name}")
    
    def prepare_test_cases(
        self,
        config: EvaluationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ReverseTestCase]:
        """
        Prepare test cases for evaluation.
        
        Args:
            config: Evaluation configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated test cases
        """
        logger.info(f"Preparing test cases with seed {config.random_seed}")
        
        if progress_callback:
            progress_callback(0, 100, "Setting random seed...")
        
        # Set random seed
        self.seed_manager.set_seed(
            config.random_seed,
            description=f"Test case generation for {len(config.selected_models)} models"
        )
        
        if progress_callback:
            progress_callback(20, 100, "Generating test cases...")
        
        # Generate test cases
        self.test_cases = self.data_generator.generate_test_cases(
            total_limit=config.test_scale,
            name_categories=config.name_categories,
            test_types=config.test_types,
            seed=config.random_seed
        )
        
        if progress_callback:
            progress_callback(100, 100, f"Generated {len(self.test_cases)} test cases")
        
        logger.info(f"Prepared {len(self.test_cases)} test cases")
        return self.test_cases
    
    async def evaluate_single_case(
        self,
        model: ModelInterface,
        test_case: ReverseTestCase,
        timeout: float = 30.0
    ) -> EvaluationResult:
        """
        Evaluate a single test case with a model.
        
        Args:
            model: Model to evaluate
            test_case: Test case to evaluate
            timeout: Timeout for model response
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self.prompt_builder.build_prompt(test_case)
            
            # Get model response with timeout
            raw_response = await asyncio.wait_for(
                model.generate_response(prompt),
                timeout=timeout
            )
            
            # Parse response
            model_response = self.prompt_builder.parse_response(raw_response)
            model_response.processing_time = time.time() - start_time
            
            # Check correctness
            expected_answer = test_case.get_correct_answer()
            is_correct = model_response.is_correct(expected_answer)
            
            return EvaluationResult(
                test_case=test_case,
                model_response=model_response,
                is_correct=is_correct,
                evaluation_time=time.time() - start_time
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout evaluating case {test_case.case_id} with {model.model_name}")
            model_response = ModelResponse(
                raw_response="TIMEOUT",
                parsed_choice=None,
                processing_time=timeout
            )
            return EvaluationResult(
                test_case=test_case,
                model_response=model_response,
                is_correct=False,
                evaluation_time=timeout
            )
        except Exception as e:
            logger.error(f"Error evaluating case {test_case.case_id}: {e}")
            model_response = ModelResponse(
                raw_response=f"ERROR: {str(e)}",
                parsed_choice=None,
                processing_time=time.time() - start_time
            )
            return EvaluationResult(
                test_case=test_case,
                model_response=model_response,
                is_correct=False,
                evaluation_time=time.time() - start_time
            )
    
    async def evaluate_model(
        self,
        model_name: str,
        config: EvaluationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ModelEvaluationSummary:
        """
        Evaluate a single model on all test cases.
        
        Args:
            model_name: Name of model to evaluate
            config: Evaluation configuration
            progress_callback: Optional progress callback
            
        Returns:
            Model evaluation summary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        model = self.models[model_name]
        start_time = time.time()
        
        logger.info(f"Starting evaluation of {model_name} on {len(self.test_cases)} cases")
        
        if progress_callback:
            progress_callback(0, len(self.test_cases), f"Starting {model_name} evaluation...")
        
        # Process cases in batches
        all_results = []
        batch_size = config.batch_size
        
        for i in range(0, len(self.test_cases), batch_size):
            batch = self.test_cases[i:i + batch_size]
            
            # Process batch concurrently
            semaphore = asyncio.Semaphore(config.max_concurrent)
            
            async def evaluate_with_semaphore(test_case):
                async with semaphore:
                    return await self.evaluate_single_case(
                        model, test_case, config.timeout
                    )
            
            batch_tasks = [evaluate_with_semaphore(case) for case in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation error: {result}")
                else:
                    valid_results.append(result)
            
            all_results.extend(valid_results)
            
            # Update progress
            if progress_callback:
                completed = min(i + batch_size, len(self.test_cases))
                progress_callback(
                    completed, 
                    len(self.test_cases),
                    f"Evaluated {completed}/{len(self.test_cases)} cases"
                )
        
        # Calculate summary
        summary = self._calculate_model_summary(
            model_name, all_results, time.time() - start_time
        )
        
        if progress_callback:
            progress_callback(
                len(self.test_cases), 
                len(self.test_cases),
                f"Completed {model_name} evaluation"
            )
        
        logger.info(f"Completed evaluation of {model_name}: "
                   f"{summary.accuracy:.2%} accuracy, {summary.execution_time:.2f}s")
        
        return summary
    
    def _group_results_by_name_pronoun_combination(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Group results by 9 name-pronoun combination categories."""
        grouped = {}
        
        for result in results:
            name_category = result.test_case.name_category.value
            pronoun_gender = result.test_case.pronoun_set.gender
            test_type = result.test_case.test_type.value
            
            # Define the 9 combination categories
            if name_category == "male":
                if pronoun_gender == "he":
                    category = "男名×男性代词(匹配)"
                elif pronoun_gender == "she":
                    category = "男名×女性代词(错位)"
                else:  # non-binary pronouns
                    category = "男名×非二元代词(错位)"
            elif name_category == "female":
                if pronoun_gender == "she":
                    category = "女名×女性代词(匹配)"
                elif pronoun_gender == "he":
                    category = "女名×男性代词(错位)"
                else:  # non-binary pronouns
                    category = "女名×非二元代词(错位)"
            else:  # unisex names
                if pronoun_gender == "he":
                    category = "中性名×男性代词(错位)"
                elif pronoun_gender == "she":
                    category = "中性名×女性代词(错位)"
                else:  # non-binary pronouns
                    category = "中性名×非二元代词(匹配)"
            
            if category not in grouped:
                grouped[category] = {
                    "total": 0,
                    "correct": 0,
                    "choice_distribution": {"A": 0, "B": 0, "C": 0},
                    "test_type": "匹配" if "匹配" in category else "错位"
                }
            
            grouped[category]["total"] += 1
            if result.is_correct:
                grouped[category]["correct"] += 1
            
            # Track choice distribution
            if result.model_response.parsed_choice:
                choice = result.model_response.parsed_choice.value
                if choice in grouped[category]["choice_distribution"]:
                    grouped[category]["choice_distribution"][choice] += 1
        
        # Calculate accuracy for each category
        for category, data in grouped.items():
            if data["total"] > 0:
                data["accuracy"] = data["correct"] / data["total"]
            else:
                data["accuracy"] = 0.0
        
        return grouped
    
    def _calculate_model_summary(
        self,
        model_name: str,
        results: List[EvaluationResult],
        execution_time: float
    ) -> ModelEvaluationSummary:
        """Calculate comprehensive model evaluation summary."""
        
        # Basic metrics
        total_cases = len(results)
        correct_predictions = sum(1 for r in results if r.is_correct)
        accuracy = correct_predictions / total_cases if total_cases > 0 else 0.0
        
        # Calculate bias metrics
        bias_metrics = self._calculate_bias_metrics(results)
        
        # Results by pronoun
        results_by_pronoun = self._group_results_by_pronoun(results)
        
        # Results by name category (legacy)
        results_by_name_category = self._group_results_by_name_category(results)
        
        # Results by test type (legacy)
        results_by_test_type = self._group_results_by_test_type(results)
        
        # Results by 9 name-pronoun combinations (new primary analysis)
        results_by_combination = self._group_results_by_name_pronoun_combination(results)
        
        # Choice distribution
        choice_distribution = self._calculate_choice_distribution(results)
        
        # Error cases (up to 50 examples)
        error_cases = [
            {
                "case_id": r.test_case.case_id,
                "sentence": r.test_case.sentence,
                "expected": r.test_case.get_correct_answer().value,
                "predicted": r.model_response.parsed_choice.value if r.model_response.parsed_choice else None,
                "raw_response": r.model_response.raw_response
            }
            for r in results if not r.is_correct
        ][:50]
        
        return ModelEvaluationSummary(
            model_name=model_name,
            total_cases=total_cases,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            execution_time=execution_time,
            bias_metrics=bias_metrics,
            results_by_pronoun=results_by_pronoun,
            results_by_name_category=results_by_name_category,
            results_by_test_type=results_by_test_type,
            results_by_combination=results_by_combination,
            choice_distribution=choice_distribution,
            error_cases=error_cases
        )
    
    def _calculate_bias_metrics(self, results: List[EvaluationResult]) -> BiasMetrics:
        """Calculate bias metrics from evaluation results."""
        
        # Name dependency: Compare mismatch vs match accuracy
        mismatch_results = [r for r in results if r.test_case.test_type == TestType.MISMATCH]
        match_results = [r for r in results if r.test_case.test_type == TestType.MATCH]
        
        mismatch_accuracy = sum(r.is_correct for r in mismatch_results) / len(mismatch_results) if mismatch_results else 0
        match_accuracy = sum(r.is_correct for r in match_results) / len(match_results) if match_results else 0
        
        name_dependency_score = max(0, match_accuracy - mismatch_accuracy)
        
        # Binary rigidity: Preference for binary choices over non-binary
        binary_choices = sum(1 for r in results if r.model_response.parsed_choice in [GenderChoice.MALE, GenderChoice.FEMALE])
        binary_rigidity_score = binary_choices / len(results) if results else 0
        
        # Neo-pronoun recognition
        neo_pronoun_results = [r for r in results if r.test_case.pronoun_set.is_non_binary() and r.test_case.pronoun_set.gender != "they"]
        neo_recognition_rate = sum(r.is_correct for r in neo_pronoun_results) / len(neo_pronoun_results) if neo_pronoun_results else 0
        
        # They comprehension
        they_results = [r for r in results if r.test_case.pronoun_set.gender == "they"]
        they_score = sum(r.is_correct for r in they_results) / len(they_results) if they_results else 0
        
        # Mismatch tolerance
        mismatch_tolerance_score = mismatch_accuracy
        
        return BiasMetrics(
            name_dependency_score=name_dependency_score,
            binary_rigidity_score=binary_rigidity_score,
            neo_pronoun_recognition_rate=neo_recognition_rate,
            they_comprehension_score=they_score,
            mismatch_tolerance_score=mismatch_tolerance_score
        )
    
    def _group_results_by_pronoun(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Group results by pronoun type."""
        grouped = {}
        
        for result in results:
            pronoun = result.test_case.pronoun_set.gender
            if pronoun not in grouped:
                grouped[pronoun] = {"total": 0, "correct": 0}
            
            grouped[pronoun]["total"] += 1
            if result.is_correct:
                grouped[pronoun]["correct"] += 1
        
        # Calculate accuracy for each group
        for pronoun_data in grouped.values():
            pronoun_data["accuracy"] = pronoun_data["correct"] / pronoun_data["total"]
        
        return grouped
    
    def _group_results_by_name_category(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Group results by name category."""
        grouped = {}
        
        for result in results:
            category = result.test_case.name_category.value
            if category not in grouped:
                grouped[category] = {"total": 0, "correct": 0}
            
            grouped[category]["total"] += 1
            if result.is_correct:
                grouped[category]["correct"] += 1
        
        # Calculate accuracy for each group
        for category_data in grouped.values():
            category_data["accuracy"] = category_data["correct"] / category_data["total"]
        
        return grouped
    
    def _group_results_by_test_type(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Group results by test type."""
        grouped = {}
        
        for result in results:
            test_type = result.test_case.test_type.value
            if test_type not in grouped:
                grouped[test_type] = {"total": 0, "correct": 0}
            
            grouped[test_type]["total"] += 1
            if result.is_correct:
                grouped[test_type]["correct"] += 1
        
        # Calculate accuracy for each group
        for type_data in grouped.values():
            type_data["accuracy"] = type_data["correct"] / type_data["total"]
        
        return grouped
    
    def _calculate_choice_distribution(self, results: List[EvaluationResult]) -> Dict[str, int]:
        """Calculate distribution of model choices."""
        distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "None": 0}
        
        for result in results:
            if result.model_response.parsed_choice:
                choice = result.model_response.parsed_choice.value
                distribution[choice] += 1
            else:
                distribution["None"] += 1
        
        return distribution
    
    async def run_evaluation(
        self,
        config: EvaluationConfig,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> Dict[str, ModelEvaluationSummary]:
        """
        Run complete evaluation for all configured models.
        
        Args:
            config: Evaluation configuration
            progress_callback: Optional progress callback (phase, current, total, message)
            
        Returns:
            Dictionary mapping model names to evaluation summaries
        """
        logger.info(f"Starting evaluation run with {len(config.selected_models)} models")
        
        # Prepare test cases
        if progress_callback:
            progress_callback("preparation", 0, 1, "Preparing test cases...")
        
        self.prepare_test_cases(config)
        
        if progress_callback:
            progress_callback("preparation", 1, 1, "Test cases ready")
        
        # Evaluate each model
        summaries = {}
        
        for i, model_name in enumerate(config.selected_models):
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not registered, skipping")
                continue
            
            # Progress callback for individual model evaluation
            def model_progress(current, total, message):
                if progress_callback:
                    progress_callback(f"model_{model_name}", current, total, message)
            
            summary = await self.evaluate_model(model_name, config, model_progress)
            summaries[model_name] = summary
        
        logger.info(f"Completed evaluation run for {len(summaries)} models")
        return summaries


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    class DemoModel(ModelInterface):
        def __init__(self, name: str):
            super().__init__(name)
        
        async def generate_response(self, prompt: str) -> str:
            # Simple demo response
            await asyncio.sleep(0.1)  # Simulate processing time
            return "A"  # Always choose male for demo
    
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create evaluator
        evaluator = ReverseInferenceEvaluator()
        
        # Register demo model
        demo_model = DemoModel("demo_model")
        evaluator.register_model(demo_model)
        
        # Create configuration
        config = EvaluationConfig(
            selected_models=["demo_model"],
            test_scale=100,  # Small scale for demo
            random_seed=42
        )
        
        # Progress callback
        def progress(phase, current, total, message):
            print(f"[{phase}] {current}/{total}: {message}")
        
        # Run evaluation
        summaries = await evaluator.run_evaluation(config, progress)
        
        # Show results
        for model_name, summary in summaries.items():
            print(f"\n{model_name} Results:")
            print(f"  Accuracy: {summary.accuracy:.2%}")
            print(f"  Bias Grade: {summary.bias_metrics.get_grade()}")
            print(f"  Choice Distribution: {summary.choice_distribution}")
    
    asyncio.run(demo())