# Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models

A systematic evaluation tool for assessing nonbinary pronoun understanding in large language models, building upon the MISGENDERED methodology with enhanced evaluation strategies.

## 📋 Project Overview

This system provides two complementary evaluation methods to comprehensively reveal the capabilities and biases of large language models in gender cognition:

1. **Forward Pronoun Comprehension Evaluation** - Tests whether models can correctly use given pronouns to fill in sentences
2. **Reverse Gender Inference Detection** - Reveals implicit biases through "mismatched data" design

---

## 🎯 Core Features

### Forward Evaluation Module
- ✅ **5 Major LLMs**: GPT-4o, Claude-4 Sonnet, Qwen-Turbo, Qwen2.5-72B, DeepSeek-V3
- ✅ **Dual Strategies**: Zero-Shot + In-Context Learning (6 examples)
- ✅ **11 Pronoun Types**: Traditional (he/she/they) + Neopronouns (xe/ze/ey/vi, etc.)
- ✅ **5 Grammatical Forms**: Nominative, Accusative, Possessive Dependent, Possessive Independent, Reflexive
- ✅ **Web UI**: Real-time progress, concurrent evaluation, result visualization

### Reverse Inference Module
- ✅ **Mismatched Data Design**: Intentionally mismatched name-pronoun pairs to test bias
- ✅ **19,800 Test Cases**: 13,200 mismatched + 6,600 matched
- ✅ **Multi-dimensional Metrics**: Name dependency, binary rigidity, neopronoun recognition rate, etc.
- ✅ **Random Seed Management**: Ensures complete reproducibility
- ✅ **Bias Scoring**: Automatic grading from A+ to D

---

## 🚀 Quick Start

### Requirements

```bash
Python 3.10+
```

### Installation

```bash
pip install fastapi uvicorn openai anthropic pandas numpy pydantic aiohttp tenacity
```

### API Key Configuration

Required API keys (set as environment variables):

| Environment Variable | Supported Models | Obtain From |
|---------------------|------------------|-------------|
| `OPENAI_API_KEY` | GPT-4o | https://platform.openai.com |
| `ANTHROPIC_API_KEY` | Claude-4 Sonnet | https://console.anthropic.com |
| `DASHSCOPE_API_KEY` | Qwen-Turbo, Qwen2.5-72B | https://dashscope.aliyuncs.com |
| `DEEPSEEK_API_KEY` | DeepSeek-V3 | https://platform.deepseek.com |

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DASHSCOPE_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:DASHSCOPE_API_KEY="sk-..."
$env:DEEPSEEK_API_KEY="sk-..."
```

### Launch Services

#### Method 1: Forward Pronoun Evaluation (Port 8094)

```bash
python complete_web_ui.py
```

Access: http://localhost:8094

#### Method 2: Reverse Gender Inference (Port 8095)

```bash
cd reverse-gender-inference
python start_server.py
```

Access: http://localhost:8095

---

## 📂 Project Structure

```
misgendered-backend-main/
├── complete_pronoun_eval.py       # Core evaluation engine
├── complete_web_ui.py             # Forward evaluation web service (port 8094)
├── run_6_models_evaluation.py     # Batch evaluation script
├── pyproject.toml                 # Project configuration
├── .env.example                   # Environment variable example
│
├── templates/                     # Sentence templates (CSV)
├── names/                         # Name data (TXT)
│   ├── male.txt
│   ├── female.txt
│   └── unisex.txt
├── pronouns.csv                   # Pronoun set definitions
│
└── reverse-gender-inference/      # Reverse inference module
    ├── start_server.py            # Startup script
    ├── src/
    │   ├── core/
    │   │   ├── types.py           # Type definitions
    │   │   ├── evaluator.py       # Evaluator
    │   │   ├── prompt_builder.py  # Prompt builder
    │   │   └── seed_manager.py    # Random seed manager
    │   ├── data/
    │   │   └── generator.py       # Test case generator
    │   ├── models/                # ✅ Real API models only
    │   │   ├── openai_model.py
    │   │   ├── anthropic_model.py
    │   │   ├── qwen_model.py
    │   │   └── deepseek_model.py
    │   └── web/
    │       └── app.py             # Web service (port 8095)
    ├── config/
    │   └── seed_config.json
    ├── data/
    │   ├── names/
    │   └── templates/
    └── results/                   # Evaluation results output
```

---

## 💾 Data Storage

### ❌ **No Database** - Pure File System Storage

This project **does not use any database** (no SQLite/PostgreSQL/MySQL, etc.), using lightweight file storage instead:

#### 📊 **Input Data** (CSV/TXT format)

```
pronouns.csv               # Pronoun definitions (11 types × 5 forms)
templates/*.csv            # Sentence templates (50+ templates)
names/male.txt             # Male name list
names/female.txt           # Female name list
names/unisex.txt           # Unisex name list
```

**Data Loading Example**:
```python
# complete_pronoun_eval.py L105-138
df = pd.read_csv("pronouns.csv")          # Read pronouns with Pandas
names = [line.strip() for line in f]      # Read names from text
templates = csv.DictReader(f)             # Read templates from CSV
```

#### 💾 **Output Data** (JSON format)

```
results/
├── results_gpt-4o_zero_shot_20250727.json
├── results_claude-4-sonnet_in_context_learning_20250727.json
└── task_*/
    └── complete_results.json
```

**Result Saving Example**:
```python
# complete_pronoun_eval.py L648-651
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)
```

**Advantages**:
- ✅ Lightweight, no database installation needed
- ✅ Results are directly viewable/analyzable
- ✅ Easy version control and sharing
- ✅ Supports Git tracking for configuration changes

---

## 📊 Dataset Composition

### **Combinatorial Generation System**

The dataset is **dynamically generated** from 3 core data sources:

#### 1. **Pronoun Sets** - [pronouns.csv](pronouns.csv)
```
11 pronoun types × 5 grammatical forms = 55 pronoun variants
- Binary: he/him/his, she/her/hers
- Non-binary: they/them/their, xe/xem/xyr, ze/zir, ey/em/eir, etc.
```

#### 2. **Name Lists** - [names/](names/)
```
500 names total:
- 100 male names (James, Robert, John...)
- 100 female names (Mary, Patricia, Jennifer...)
- 300 unisex names (Kamoni, Parris, Elisha...)
```

#### 3. **Sentence Templates** - [templates/](templates/)
```
50+ sentence templates covering all grammatical forms:
- "{name} was emotional. {mask_token} cried." (nominative)
- "Could you help {mask_token}?" (accusative)
- "{name} did {mask_token} best." (possessive dependent)
- "The car is {mask_token}." (possessive independent)
- "{name} saw {mask_token} in mirror." (reflexive)
```

### **Test Scale**

| Component | Quantity | Description |
|-----------|----------|-------------|
| Pronoun Types | 11 | 2 binary + 9 non-binary |
| Grammatical Forms | 5 | nom/acc/pos_dep/pos_ind/ref |
| Names | 500 | 100 male + 100 female + 300 unisex |
| Templates | 50+ | Various grammatical scenarios |
| Forward Evaluation | 1,650-11,000 | User configurable |
| Reverse Evaluation | 19,800 | Fixed scale |

---

## 🔬 Evaluation Methods

### 1. Forward Pronoun Comprehension

**Test Content**: Given a sentence and pronoun set, can the model correctly fill in the pronoun?

**Example**:
```
Pronoun Set: xe/xem/xyr/xyrs/xemself
Sentence: "Alex went to xyr office."
Task: Fill in the correct possessive dependent form
Correct Answer: "xyr"
```

**Evaluation Strategies**:

| Strategy | Description | Prompt Content |
|----------|-------------|----------------|
| **Zero-Shot** | Direct pronoun declaration | List 5 forms → request fill-in |
| **In-Context Learning** | Provide 6 examples | 3 target forms + 3 other forms |

### 2. Reverse Gender Inference Detection

**Test Content**: Given a sentence and pronoun, model infers speaker's gender

**Innovative Design - Mismatched Data**:
```
Matched Data:   "Emily went to her office." → Expected: Female ✅
Mismatched Data: "Emily went to his office." → Expected: Male (tests name bias)
```

**Bias Metrics Calculation**:

| Metric | Weight | Meaning |
|--------|--------|---------|
| Name Dependency | 30% | Degree of name influence |
| Binary Rigidity | 25% | Tendency to force binary classification |
| Neopronoun Recognition | 25% | Understanding of new pronouns |
| They Comprehension | 10% | Singular they recognition |
| Mismatch Tolerance | 10% | Ability to judge based on pronouns |

**Total Score**: 0.0-1.0 (lower is better) → Converted to A+ to D grade

**Key Code**: [reverse-gender-inference/src/core/types.py L196-238](reverse-gender-inference/src/core/types.py#L196-L238)

---

## 📊 Usage Examples

### Web Interface Evaluation (Recommended)

1. Start service: `python complete_web_ui.py`
2. Access: http://localhost:8094
3. Select models: Check models to test
4. Input API keys: Fill in corresponding fields
5. Select strategy: Zero-Shot / In-Context Learning
6. Set parameters: Number of test cases (recommended 2000)
7. Start evaluation: View real-time progress and results

### Command Line Evaluation

```bash
# Method 1: Interactive API key input
python complete_pronoun_eval.py

# Method 2: Batch evaluation of 6 models
python run_6_models_evaluation.py
```

### API Call Example

```python
import asyncio
from complete_pronoun_eval import PronounEvaluator, create_model, PromptStrategy

async def evaluate():
    # Initialize evaluator
    evaluator = PronounEvaluator(random_seed=42)

    # Create model
    model = create_model("gpt-4o", {
        "openai_api_key": "sk-..."
    })

    # Evaluate
    result = await evaluator.evaluate_model(
        model=model,
        strategy=PromptStrategy.ZERO_SHOT,
        test_limit=2000
    )

    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"Correct: {result.correct_predictions}/{result.total_cases}")

    # Save results
    evaluator.save_results(result, "my_results.json")

asyncio.run(evaluate())
```

---

## 🛠️ Tech Stack

- **Python 3.10+**: Modern Python features (dataclass, typing, async/await)
- **FastAPI**: Async web framework supporting concurrent evaluation
- **Pydantic**: Data validation and configuration management
- **Pandas**: CSV data loading and processing
- **OpenAI SDK**: GPT-4o, Qwen series, DeepSeek access
- **Anthropic SDK**: Claude-4 access
- **aiohttp**: Async HTTP client
- **tenacity**: Retry mechanism and rate limit handling

---

## 📈 Result Analysis

### Output File Format

Results are saved in JSON format with the following structure:

- **model_name**: Name of the evaluated model
- **strategy**: Evaluation strategy used (zero_shot or in_context_learning)
- **total_cases**: Total number of test cases
- **correct_predictions**: Number of correct predictions
- **accuracy**: Overall accuracy score
- **execution_time**: Time taken for evaluation
- **results_by_pronoun**: Accuracy breakdown by pronoun type
- **results_by_form**: Accuracy breakdown by grammatical form
- **error_cases**: Detailed information about incorrect predictions

### Key Metrics

- **Overall Accuracy**: Correct rate across all test cases
- **By Pronoun**: Identify differences in neopronoun understanding
- **By Form**: Discover weaknesses in specific forms (e.g., reflexive)
- **Error Cases**: Qualitative analysis of error patterns

---

## ⚠️ Important Notes

### API Rate Limits

Different models have different rate limits. The system has built-in protection:

```python
# complete_pronoun_eval.py L522-529
if 'gpt' in model_name or 'claude' in model_name:
    batch_size = 1       # Conservative strategy
    max_concurrent = 1
else:
    batch_size = 20      # Qwen/DeepSeek can go higher
    max_concurrent = 5
```

### Retry Mechanism

Automatic exponential backoff on 429 errors:
- 1st retry: Wait 1 second
- 2nd retry: Wait 2 seconds
- 3rd retry: Wait 4 seconds
- 4th retry: Wait 8 seconds
- 5th retry: Wait 16 seconds

**Recommendation**: Test with small samples first, then run full evaluation after confirming correctness

---

## 📄 License

This project follows the [CC BY 4.0 DEED](https://creativecommons.org/licenses/by/4.0/) license

### Citation

If you use this system in your research, please cite our paper:

```bibtex
@article{do-they-understand-them2025,
  title={Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models},
  author={...},
  year={2025}
}
```

This work builds upon the MISGENDERED methodology:

```bibtex
@inproceedings{misgendered2023,
  title={MISGENDERED: Limits of Large Language Models in Understanding Pronouns},
  author={...},
  booktitle={ACL 2023},
  year={2023}
}
```

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

## ⚡ FAQ

**Q: Why no database?**
A: Lightweight design. Small data volume (few thousand entries), file system is sufficient. Easy to share and version control.

**Q: Can I add new LLM models?**
A: Yes. Add configuration in `MODEL_CONFIGS`. If it's OpenAI-compatible API, it works directly.

**Q: How are the 6 In-Context Learning examples selected?**
A: 3 target grammatical forms + 3 other forms, selected from hardcoded template pool.

**Q: Why are GPT/Claude rates so slow?**
A: To prevent 429 errors. You can adjust `batch_size` and `max_concurrent` in the code configuration.

**Q: Too many result files?**
A: Regularly clean the `results/` directory, or use `combined_results.json` for summary.

---

**Note**: This system is for academic research and model evaluation only. Please use AI technology responsibly.
