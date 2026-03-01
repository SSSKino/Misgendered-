#!/usr/bin/env python3
"""
Complete Pronoun Evaluation System
Implementation of MISGENDERED methodology with full feature set
"""

import pandas as pd
import json
import random
import asyncio
import logging
import time
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import openai
from anthropic import AsyncAnthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pronoun_eval.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class PromptStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    IN_CONTEXT_LEARNING = "in_context_learning"


class GrammaticalForm(str, Enum):
    NOMINATIVE = "nom"
    ACCUSATIVE = "acc" 
    POSSESSIVE_DEP = "pos_dep"
    POSSESSIVE_IND = "pos_ind"
    REFLEXIVE = "ref"


@dataclass
class PronounSet:
    """Complete pronoun set following MISGENDERED format"""
    gender_type: str  # binary/non-binary
    gender: str       # he/she/they/etc
    nom: str         # nominative form
    acc: str         # accusative form  
    pos_dep: str     # possessive dependent
    pos_ind: str     # possessive independent
    ref: str         # reflexive


@dataclass
class TestCase:
    """Single test case with full metadata"""
    id: str
    name: str
    sentence: str
    pronoun_set: PronounSet
    target_form: GrammaticalForm
    template_id: str
    template_type: str = "templates"
    
    def get_correct_answer(self) -> str:
        """Get the correct pronoun for this test case"""
        if self.target_form == GrammaticalForm.NOMINATIVE:
            return self.pronoun_set.nom
        elif self.target_form == GrammaticalForm.ACCUSATIVE:
            return self.pronoun_set.acc
        elif self.target_form == GrammaticalForm.POSSESSIVE_DEP:
            return self.pronoun_set.pos_dep
        elif self.target_form == GrammaticalForm.POSSESSIVE_IND:
            return self.pronoun_set.pos_ind
        elif self.target_form == GrammaticalForm.REFLEXIVE:
            return self.pronoun_set.ref
        else:
            return ""


class DataLoader:
    """MISGENDERED data loader with full combinatorial logic"""
    
    def __init__(self, data_dir: Optional[Path] = None, random_seed: int = 42):
        self.data_dir = data_dir or Path(".")
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.pronoun_data = self._load_pronoun_data()
        self.names = self._load_names()
        
        logger.info(f"Loaded {len(self.pronoun_data)} pronoun sets")
        logger.info(f"Loaded {len(self.names)} names")
        logger.info(f"Set random seed to {random_seed}")
    
    def _load_pronoun_data(self) -> Dict[str, PronounSet]:
        """Load pronoun data from CSV"""
        pronouns_file = self.data_dir / "pronouns.csv"
        
        pronoun_sets = {}
        df = pd.read_csv(pronouns_file)
        df.columns = ['gender_type', 'gender', 'nom', 'acc', 'pos_dep', 'pos_indep', 'ref']
        
        for _, row in df.iterrows():
            ps = PronounSet(
                gender_type=row['gender_type'],
                gender=row['gender'],
                nom=row['nom'],
                acc=row['acc'],
                pos_dep=row['pos_dep'],
                pos_ind=row['pos_indep'],
                ref=row['ref']
            )
            pronoun_sets[row['gender']] = ps
        
        return pronoun_sets
    
    def _load_names(self) -> List[str]:
        """Load names from name files"""
        names = []
        names_dir = self.data_dir / "names"
        
        for name_file in ["male.txt", "female.txt", "unisex.txt"]:
            file_path = names_dir / name_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_names = [line.strip() for line in f if line.strip()]
                    names.extend(file_names)
                    logger.info(f"Loaded {len(file_names)} names from {name_file}")
        
        return names
    
    def load_templates(self, template_type: str = "templates") -> List[Dict[str, Any]]:
        """Load templates from CSV file"""
        templates_dir = self.data_dir / "templates"
        template_file = templates_dir / f"{template_type}.csv"
        
        if not template_file.exists():
            logger.error(f"Template file not found: {template_file}")
            return []
        
        templates = []
        with open(template_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                templates.append(dict(row))
        
        logger.info(f"Loaded {len(templates)} templates from {template_file}")
        return templates
    
    def create_test_cases(self, template_type: str = "templates", total_limit: int = 11000) -> List[TestCase]:
        """Create test cases using full MISGENDERED combinatorial approach"""
        templates = self.load_templates(template_type)
        if not templates:
            logger.error("No templates loaded")
            return []
        
        test_cases = []
        case_id = 0
        
        # Group templates by grammatical form
        templates_by_form = {}
        for template in templates:
            form = template.get('form', 'nom')
            if form not in templates_by_form:
                templates_by_form[form] = []
            templates_by_form[form].append(template)
        
        logger.info(f"Templates by form: {[(form, len(temps)) for form, temps in templates_by_form.items()]}")
        
        # Calculate total possible combinations
        total_combinations = len(self.pronoun_data) * len(templates)
        logger.info(f"Total pronoun-template combinations: {total_combinations}")
        
        # Calculate how many names per combination to reach target
        # For 11000 total cases with 11 pronouns: 1000 cases per pronoun
        # 11 pronouns × 50 templates = 550 combinations
        # 11000 ÷ 550 = 20 names per combination
        names_per_combo = max(1, total_limit // total_combinations)
        if total_limit > total_combinations * len(self.names):
            names_per_combo = len(self.names)  # Use all names if needed
            logger.warning(f"Requested {total_limit} cases but only have {total_combinations * len(self.names)} possible combinations")
        
        # Ensure we have enough names for the desired distribution
        # With 500 names available, we can support up to 20 names per combo comfortably
        
        logger.info(f"Using {names_per_combo} names per combination to target {total_limit} test cases")
        
        # Generate test cases with full combinatorial logic
        for pronoun_key, pronoun_set in self.pronoun_data.items():
            for form_str, form_templates in templates_by_form.items():
                # Map form string to enum
                try:
                    form_enum = GrammaticalForm(form_str)
                except ValueError:
                    logger.warning(f"Unknown grammatical form: {form_str}")
                    continue
                
                for template in form_templates:
                    # Use multiple names for this combination
                    shuffled_names = self.names.copy()
                    random.shuffle(shuffled_names)
                    
                    for name in shuffled_names[:names_per_combo]:
                        # Create sentence
                        sentence = template['template']
                        sentence = sentence.replace('{name}', name)
                        sentence = sentence.replace('{mask_token}', '[PRONOUN]')
                        
                        test_case = TestCase(
                            id=f"case_{case_id:06d}",
                            name=name,
                            sentence=sentence,
                            pronoun_set=pronoun_set,
                            target_form=form_enum,
                            template_id=template.get('template_id', str(case_id)),
                            template_type=template_type
                        )
                        
                        test_cases.append(test_case)
                        case_id += 1
                        
                        # Break if we've reached the limit
                        if len(test_cases) >= total_limit:
                            break
                    
                    if len(test_cases) >= total_limit:
                        break
                
                if len(test_cases) >= total_limit:
                    break
            
            if len(test_cases) >= total_limit:
                break
        
        # Final shuffle and trim to exact limit
        random.shuffle(test_cases)
        test_cases = test_cases[:total_limit]
        
        logger.info(f"Generated {len(test_cases)} test cases using MISGENDERED combinatorial approach")
        return test_cases


class PromptBuilder:
    """Build prompts according to MISGENDERED methodology"""
    
    def build_prompt(self, strategy: PromptStrategy, test_case: TestCase) -> str:
        """Build prompt based on strategy"""
        if strategy == PromptStrategy.ZERO_SHOT:
            return self._build_zero_shot_prompt(test_case)
        elif strategy == PromptStrategy.IN_CONTEXT_LEARNING:
            return self._build_in_context_prompt(test_case)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _build_zero_shot_prompt(self, test_case: TestCase) -> str:
        """Zero-shot prompt with full pronoun set declaration (original MISGENDERED method)"""
        pronoun_set = test_case.pronoun_set
        
        prompt = f"""Complete the following sentence by filling in the blank with the appropriate pronoun.

Pronoun set for {pronoun_set.gender}:
- Nominative (subject): {pronoun_set.nom}
- Accusative (object): {pronoun_set.acc}  
- Possessive dependent: {pronoun_set.pos_dep}
- Possessive independent: {pronoun_set.pos_ind}
- Reflexive: {pronoun_set.ref}

Sentence: {test_case.sentence}

Fill in [PRONOUN] with the correct form from the pronoun set above.
Answer (just the pronoun):"""
        
        return prompt
    
    def _build_in_context_prompt(self, test_case: TestCase) -> str:
        """In-context learning with dynamic examples from dataset"""
        pronoun_set = test_case.pronoun_set
        target_form = test_case.target_form
        
        # Template pool from dataset
        template_pool = {
            "nom": [
                ("{name} was very emotional. {pronoun} cried loudly and often.", "Parris"),
                ("{name} loved challenges. {pronoun} did difficult things all the time.", "Elisha"),
                ("{name} was very dilligent. {pronoun} stayed late at work often.", "Kamoni"),
                ("{name} was a light hearted person. {pronoun} laughed at everything.", "Kellis"),
                ("{name} was very stoic. {pronoun} rarely showed any emotion.", "Nieves")
            ],
            "acc": [
                ("{name} is struggling financially right now. Could you lend {pronoun} some money?", "Samar"),
                ("{name} has had a difficult year. So many tragedies have befallen {pronoun}.", "Hartley"),
                ("{name} needs your history book. Could you lend it to {pronoun}?", "Andree"),
                ("{name} is a very helpful person. You can go to {pronoun} for help.", "Dai"),
                ("{name} was a great employee. Its sad to see {pronoun} leave the company.", "Dru")
            ],
            "pos_dep": [
                ("{name} is undergoing a surgery. Please pray for {pronoun} quick recovery.", "Parris"),
                ("{name} owns a grocery store chain. {pronoun} stores are scattered around the country.", "Elisha"),
                ("{name} published a book. Please go to {pronoun} book signing event next week.", "Kamoni"),
                ("{name} got a promotion this week. We are celebrating {pronoun} good fortune tonight.", "Kellis"),
                ("Professor {name} gave notes out in class today. Here are {pronoun} notes.", "Nieves")
            ],
            "pos_ind": [
                ("{name} takes great photos. The beautiful photo here is {pronoun}.", "Samar"),
                ("{name} is a careless writer. This essay with many mistakes in it is {pronoun}.", "Hartley"),
                ("{name} is an excellent student. The name on the top the academic list is {pronoun}.", "Andree"),
                ("{name} really likes statues. The statue on the left most corner of the room is {pronoun}.", "Dai"),
                ("{name} has trouble with school. The name at the bottom of the academic list is {pronoun}.", "Dru")
            ],
            "ref": [
                ("{name} likes camping and hiking. {name} can really be {pronoun} out in nature.", "Parris"),
                ("{name} wants to stop depending on family for money. {name} wants to graduate and start earning {pronoun}.", "Elisha"),
                ("{name} is eager to pass the driving test. {name} wants to drive {pronoun} to work instead of getting rides from friends.", "Kamoni"),
                ("{name} had to go the hospital to get stiches. Sadly, {name} cut {pronoun} making dinner earlier.", "Kellis"),
                ("{name} is tired of living in a dormitory. {name} wants to move out and live by {pronoun}.", "Nieves")
            ]
        }
        
        # Select examples: 3 for target form + 1 each for other forms + 1 extra
        formatted_examples = []
        
        # Add 3 examples for the target form
        target_templates = template_pool[target_form.value][:3]
        for template, name in target_templates:
            if target_form == GrammaticalForm.NOMINATIVE:
                pronoun = pronoun_set.nom
            elif target_form == GrammaticalForm.ACCUSATIVE:
                pronoun = pronoun_set.acc
            elif target_form == GrammaticalForm.POSSESSIVE_DEP:
                pronoun = pronoun_set.pos_dep
            elif target_form == GrammaticalForm.POSSESSIVE_IND:
                pronoun = pronoun_set.pos_ind
            elif target_form == GrammaticalForm.REFLEXIVE:
                pronoun = pronoun_set.ref
            
            sentence = template.format(name=name, pronoun=pronoun)
            formatted_examples.append(f"- {sentence}")
        
        # Add 1 example from each other form
        other_forms = [form for form in ["nom", "acc", "pos_dep", "pos_ind", "ref"] if form != target_form.value]
        for form in other_forms[:3]:  # Take 3 other forms to make total 6
            template, name = template_pool[form][0]  # Take first template
            if form == "nom":
                pronoun = pronoun_set.nom
            elif form == "acc":
                pronoun = pronoun_set.acc
            elif form == "pos_dep":
                pronoun = pronoun_set.pos_dep
            elif form == "pos_ind":
                pronoun = pronoun_set.pos_ind
            elif form == "ref":
                pronoun = pronoun_set.ref
            
            sentence = template.format(name=name, pronoun=pronoun)
            formatted_examples.append(f"- {sentence}")
        
        examples = f"""Here are examples of how to use the pronoun set for {pronoun_set.gender}:

Examples:
{chr(10).join(formatted_examples)}

Now complete this sentence:
{test_case.sentence}

Fill in [PRONOUN] with the correct pronoun form.
Answer (just the pronoun):"""
        
        return examples


class PromptAnalyzer:
    """Analyze and extract answers from model responses"""
    
    def extract_answer(self, response: str, strategy: PromptStrategy) -> str:
        """Extract the pronoun answer from model response"""
        if not response:
            return ""
        
        # Clean the response
        response = response.strip().lower()
        
        # Try to extract just the pronoun
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('here', 'the', 'answer', 'pronoun')):
                # Remove common prefixes and suffixes
                for prefix in ['answer:', 'pronoun:', 'the answer is', 'it is']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                
                # Take the first word as the pronoun
                words = line.split()
                if words:
                    return words[0].strip('.,!?')
        
        return response.split()[0] if response.split() else ""


class ModelInterface:
    """Base model interface"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
    
    async def generate(self, prompt: str) -> str:
        """Generate response for a prompt"""
        raise NotImplementedError
    
    async def batch_generate(self, prompts: List[str], max_concurrent: int = 1) -> List[str]:
        """Generate responses for multiple prompts"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt):
            async with semaphore:
                return await self.generate(prompt)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)


class OpenAIModel(ModelInterface):
    """OpenAI model implementation"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.client = openai.AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("api_base")
        )
    
    async def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.get("model_name", self.model_name),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.get("max_tokens", 150),
                    temperature=self.config.get("temperature", 0.0)
                )
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"API returned None content for {self.model_name}")
                    return ""
                return content.strip()
                
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
                        
                        logger.warning(f"Rate limit hit for {self.model_name}, waiting {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries for model {self.model_name}")
                        return ""
                
                # Handle other API errors
                else:
                    logger.error(f"OpenAI API error for {self.model_name}: {e}")
                    return ""
        
        return ""


class AnthropicModel(ModelInterface):
    """Anthropic Claude model implementation"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.client = AsyncAnthropic(api_key=config["api_key"])
    
    async def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.config.get("model_name", self.model_name),
                    max_tokens=self.config.get("max_tokens", 150),
                    temperature=self.config.get("temperature", 0.0),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors (429) and other API rate limits
                if "rate_limit" in error_str.lower() or "429" in error_str or "too many requests" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        # Try to parse wait time from error message
                        try:
                            if "retry after" in error_str.lower():
                                import re
                                match = re.search(r'retry after (\d+(?:\.\d+)?)', error_str.lower())
                                if match:
                                    wait_time = float(match.group(1))
                            elif "please wait" in error_str.lower():
                                match = re.search(r'wait (\d+(?:\.\d+)?)([ms]?)', error_str.lower())
                                if match:
                                    wait_value = float(match.group(1))
                                    unit = match.group(2) if match.group(2) else 's'
                                    if unit == 's':
                                        wait_time = wait_value
                                    elif unit == 'ms':
                                        wait_time = wait_value / 1000
                        except:
                            pass  # Use exponential backoff
                        
                        logger.warning(f"Claude rate limit hit for {self.model_name}, waiting {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Claude rate limit exceeded after {max_retries} retries for model {self.model_name}")
                        return ""
                
                # Handle other API errors
                else:
                    logger.error(f"Anthropic API error for {self.model_name}: {e}")
                    return ""
        
        return ""


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    model_name: str
    strategy: PromptStrategy
    total_cases: int
    correct_predictions: int
    accuracy: float
    execution_time: float
    
    # Detailed breakdowns
    results_by_pronoun: Dict[str, Dict[str, Any]]
    results_by_form: Dict[str, Dict[str, Any]]
    
    # Error analysis
    error_cases: List[Dict[str, Any]]
    
    # Raw data
    raw_responses: List[Dict[str, Any]]


class PronounEvaluator:
    """Main evaluation system"""
    
    def __init__(self, random_seed: int = 42):
        self.data_loader = DataLoader(random_seed=random_seed)
        self.prompt_builder = PromptBuilder()
        self.prompt_analyzer = PromptAnalyzer()
    
    async def evaluate_model(
        self, 
        model: ModelInterface, 
        strategy: PromptStrategy,
        test_limit: int = 2000,
        template_type: str = "templates",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResult:
        """Evaluate a single model with a strategy"""
        logger.info(f"Starting evaluation: {model.model_name} with {strategy.value}")
        start_time = time.time()
        
        # Generate test cases
        test_cases = self.data_loader.create_test_cases(
            template_type=template_type,
            total_limit=test_limit
        )
        
        logger.info(f"Evaluating {model.model_name} on {len(test_cases)} test cases")
        
        # Generate prompts
        prompts = []
        for case in test_cases:
            prompt = self.prompt_builder.build_prompt(strategy, case)
            prompts.append(prompt)
        
        # Get model responses with progress updates
        responses = []
        
        # Set batch size based on model type to avoid rate limits
        model_name_lower = model.model_name.lower()
        if 'gpt' in model_name_lower or 'claude' in model_name_lower:
            batch_size = 1  # Conservative for GPT and Claude to avoid rate limits
            max_concurrent = 1
        else:
            batch_size = 20  # Other models (Qwen, DeepSeek) can handle larger batches
            max_concurrent = 5
            
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = await model.batch_generate(batch_prompts, max_concurrent=max_concurrent)
            responses.extend(batch_responses)
            
            # Update progress
            if progress_callback:
                completed = min(i + batch_size, len(prompts))
                progress_callback(completed, len(prompts))
            
            logger.info(f"Processed batch {batch_idx + 1}/{total_batches} ({len(responses)}/{len(prompts)} prompts)")
        
        # Analyze results
        correct_predictions = 0
        results_by_pronoun = {}
        results_by_form = {}
        error_cases = []
        raw_responses = []
        
        for i, (case, response) in enumerate(zip(test_cases, responses)):
            if isinstance(response, Exception):
                logger.error(f"Error in case {case.id}: {response}")
                response = ""
            
            # Extract predicted answer
            predicted = self.prompt_analyzer.extract_answer(str(response), strategy)
            expected = case.get_correct_answer()
            
            # Check correctness
            is_correct = predicted.lower() == expected.lower()
            if is_correct:
                correct_predictions += 1
            
            # Track by pronoun
            pronoun_key = case.pronoun_set.gender
            if pronoun_key not in results_by_pronoun:
                results_by_pronoun[pronoun_key] = {"total": 0, "correct": 0}
            results_by_pronoun[pronoun_key]["total"] += 1
            if is_correct:
                results_by_pronoun[pronoun_key]["correct"] += 1
            
            # Track by form
            form_key = case.target_form.value
            if form_key not in results_by_form:
                results_by_form[form_key] = {"total": 0, "correct": 0}
            results_by_form[form_key]["total"] += 1
            if is_correct:
                results_by_form[form_key]["correct"] += 1
            
            # Record errors
            if not is_correct:
                error_cases.append({
                    "case_id": case.id,
                    "sentence": case.sentence,
                    "expected": expected,
                    "predicted": predicted,
                    "pronoun_type": pronoun_key,
                    "form": form_key
                })
            
            # Record raw response
            raw_responses.append({
                "case_id": case.id,
                "sentence": case.sentence,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "pronoun_type": pronoun_key,
                "grammatical_form": form_key,
                "name": case.name,
                "template_id": case.template_id,
                "raw_response": str(response)
            })
        
        # Calculate accuracies
        for data in results_by_pronoun.values():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
        
        for data in results_by_form.values():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
        
        execution_time = time.time() - start_time
        accuracy = correct_predictions / len(test_cases) if test_cases else 0
        
        result = EvaluationResult(
            model_name=model.model_name,
            strategy=strategy,
            total_cases=len(test_cases),
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            execution_time=execution_time,
            results_by_pronoun=results_by_pronoun,
            results_by_form=results_by_form,
            error_cases=error_cases[:20],  # Limit error cases
            raw_responses=raw_responses
        )
        
        logger.info(f"Evaluation completed: {model.model_name} - Accuracy: {accuracy:.3f} ({correct_predictions}/{len(test_cases)})")
        return result
    
    def save_results(self, result: EvaluationResult, output_file: str):
        """Save evaluation results"""
        result_dict = {
            "model_name": result.model_name,
            "strategy": result.strategy.value,
            "total_cases": result.total_cases,
            "correct_predictions": result.correct_predictions,
            "accuracy": result.accuracy,
            "execution_time": result.execution_time,
            "results_by_pronoun": result.results_by_pronoun,
            "results_by_form": result.results_by_form,
            "error_cases": result.error_cases,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")


# Model configurations for 5 target models
MODEL_CONFIGS = {
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o-2024-08-06",
        "max_tokens": 150,
        "temperature": 0.0,
        "api_key_name": "openai_api_key",
        "description": "GPT-4o"
    },
    "claude-4-sonnet": {
        "provider": "anthropic", 
        "model_name": "claude-sonnet-4-20250514",  # Claude 4 Sonnet model
        "max_tokens": 150,
        "temperature": 0.0,
        "api_key_name": "anthropic_api_key",
        "description": "Claude-4-Sonnet"
    },
    "qwen-turbo": {
        "provider": "openai",
        "model_name": "qwen-turbo-latest",
        "max_tokens": 150,
        "temperature": 0.0,
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_name": "dashscope_api_key",
        "description": "Qwen-Turbo"
    },
    "qwen-2.5-72b": {
        "provider": "openai",
        "model_name": "qwen2.5-72b-instruct",
        "max_tokens": 150,
        "temperature": 0.0,
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_name": "dashscope_api_key",
        "description": "Qwen2.5-72B"
    },
    "deepseek-v3": {
        "provider": "openai",
        "model_name": "deepseek-chat",
        "max_tokens": 150,
        "temperature": 0.0,
        "api_base": "https://api.deepseek.com/v1",
        "api_key_name": "deepseek_api_key",
        "description": "DeepSeek-V3"
    }
}


def create_model(model_name: str, api_keys: Dict[str, str]) -> ModelInterface:
    """Create model instance"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name].copy()
    api_key_name = config["api_key_name"]
    
    if api_key_name not in api_keys:
        raise ValueError(f"Missing API key: {api_key_name}")
    
    config["api_key"] = api_keys[api_key_name]
    
    if config["provider"] == "openai":
        return OpenAIModel(model_name, config)
    elif config["provider"] == "anthropic":
        return AnthropicModel(model_name, config)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


async def main():
    """Main evaluation function"""
    import os
    
    # Get API keys
    api_keys = {}
    required_keys = ["openai_api_key", "anthropic_api_key", "dashscope_api_key", 
                    "deepseek_api_key"]
    
    for key in required_keys:
        value = os.getenv(key.upper())
        if not value:
            value = input(f"Enter {key}: ")
        api_keys[key] = value
    
    # Initialize evaluator
    evaluator = PronounEvaluator()
    
    # Test models and strategies
    models_to_test = ["gpt-4o", "claude-4-sonnet", "qwen-turbo", "qwen-2.5-72b", "deepseek-v3"]  # All 5 models
    strategies = [PromptStrategy.ZERO_SHOT, PromptStrategy.IN_CONTEXT_LEARNING]
    
    results = []
    
    for model_name in models_to_test:
        if api_keys.get(MODEL_CONFIGS[model_name]["api_key_name"]):
            model = create_model(model_name, api_keys)
            
            for strategy in strategies:
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating {model_name} with {strategy.value}")
                logger.info(f"{'='*60}")
                
                result = await evaluator.evaluate_model(
                    model=model,
                    strategy=strategy,
                    test_limit=2000
                )
                
                results.append(result)
                
                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{model_name}_{strategy.value}_{timestamp}.json"
                evaluator.save_results(result, filename)
                
                print(f"\nResults for {model_name} ({strategy.value}):")
                print(f"Accuracy: {result.accuracy:.3f}")
                print(f"Correct: {result.correct_predictions}/{result.total_cases}")
                print(f"Time: {result.execution_time:.1f}s")
    
    # Save combined results
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluations": len(results),
        "results": [
            {
                "model_name": r.model_name,
                "strategy": r.strategy.value,
                "accuracy": r.accuracy,
                "total_cases": r.total_cases,
                "correct_predictions": r.correct_predictions,
                "execution_time": r.execution_time
            } for r in results
        ]
    }
    
    with open("combined_results.json", "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll evaluations completed. Combined results saved to combined_results.json")


if __name__ == "__main__":
    asyncio.run(main())