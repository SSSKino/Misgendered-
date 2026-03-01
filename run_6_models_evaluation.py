#!/usr/bin/env python3
"""6个主流模型的代词理解评估脚本 - 闭源3个+开源3个"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import json

# 添加项目路径
sys.path.insert(0, '.')

from src.pronoun_eval.config.settings import Settings, MODEL_CONFIGS, ModelProvider
from src.pronoun_eval.models.factory import create_model
from src.pronoun_eval.core.evaluator import PronounEvaluator, EvaluationConfig
from src.pronoun_eval.core.prompts import PromptStrategy
from src.pronoun_eval.data.loader import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 6个目标模型配置 - 闭源3个+开源3个
TARGET_MODELS = [
    # 闭源模型
    "gpt-4o-2024-08-06",     # OpenAI GPT-4o
    "claude-4-sonnet",       # Anthropic Claude-4-Sonnet
    "qwen-max",              # Qwen-Max (阿里闭源版)
    
    # 开源模型  
    "qwen-2-5-open",         # Qwen2.5-72B (开源版)
    "deepseek-v3",           # DeepSeek-V3
    "llama-3-2-405b",        # Llama-3.2-405B
]

def get_api_key_for_model(model_name: str, settings: Settings) -> Optional[str]:
    """根据模型名称获取对应的API密钥"""
    api_key_mapping = {
        # 闭源模型
        "gpt-4o-2024-08-06": settings.openai_api_key,
        "claude-4-sonnet": settings.anthropic_api_key,
        "qwen-max": settings.dashscope_api_key,
        
        # 开源模型
        "qwen-2-5-open": settings.together_api_key,
        "deepseek-v3": settings.deepseek_api_key,
        "llama-3-2-405b": settings.together_api_key,
    }
    return api_key_mapping.get(model_name)


def check_available_models(settings: Settings) -> List[str]:
    """检查哪些模型的API密钥可用"""
    available_models = []
    
    for model_name in TARGET_MODELS:
        api_key = get_api_key_for_model(model_name, settings)
        if api_key:
            available_models.append(model_name)
            logger.info(f"✓ {model_name}: API密钥已配置")
        else:
            logger.warning(f"✗ {model_name}: API密钥未配置")
    
    return available_models


async def create_model_instance(model_name: str, settings: Settings):
    """创建模型实例"""
    try:
        model_config = MODEL_CONFIGS[model_name].copy()
        api_key = get_api_key_for_model(model_name, settings)
        
        if not api_key:
            raise ValueError(f"API key not found for {model_name}")
        
        model_config["api_key"] = api_key
        
        # 创建模型
        model = create_model(model_config["provider"], model_config)
        logger.info(f"✓ 成功创建模型: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"✗ 创建模型失败 {model_name}: {e}")
        return None


async def evaluate_single_model(
    model_name: str,
    model,
    strategy: PromptStrategy,
    test_limit: int = 50
) -> Dict[str, Any]:
    """评估单个模型"""
    logger.info(f"开始评估: {model_name} with {strategy.value}")
    
    try:
        # 配置评估
        config = EvaluationConfig(
            strategy=strategy,
            test_limit=test_limit,
            batch_size=10,
            save_results=True,
            output_dir=Path("results") / "6_models_comparison"
        )
        
        # 运行评估
        evaluator = PronounEvaluator(config)
        result = await evaluator.evaluate_model(model)
        
        logger.info(
            f"✓ {model_name} 完成: {result.accuracy:.3f} "
            f"({result.correct_predictions}/{result.total_cases})"
        )
        
        return {
            "model_name": model_name,
            "strategy": strategy.value,
            "accuracy": result.accuracy,
            "correct": result.correct_predictions,
            "total": result.total_cases,
            "execution_time": result.execution_time,
            "results_by_pronoun": result.results_by_pronoun,
            "results_by_form": result.results_by_form,
            "error_count": len(result.error_cases)
        }
        
    except Exception as e:
        logger.error(f"✗ 评估失败 {model_name}: {e}")
        return {
            "model_name": model_name,
            "strategy": strategy.value,
            "accuracy": 0.0,
            "error": str(e)
        }


async def run_comprehensive_evaluation(
    test_limit: int = 50,
    strategies: Optional[List[PromptStrategy]] = None
):
    """运行全面的8模型评估"""
    
    if strategies is None:
        strategies = [
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.IN_CONTEXT_LEARNING
        ]
    
    print("="*80)
    print("🚀 6个主流大模型代词理解能力评估 (闭源3个+开源3个)")
    print("="*80)
    
    # 加载设置
    settings = Settings()
    
    # 检查可用模型
    available_models = check_available_models(settings)
    
    if not available_models:
        print("❌ 没有可用的模型API密钥，请检查.env配置")
        return
    
    print(f"\n📊 将评估 {len(available_models)} 个模型，每个模型 {len(strategies)} 种策略")
    print(f"📝 每个模型测试 {test_limit} 个用例")
    
    # 创建输出目录
    output_dir = Path("results") / "6_models_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = []
    start_time = time.time()
    
    # 逐个模型评估
    for i, model_name in enumerate(available_models, 1):
        print(f"\n{'='*60}")
        print(f"📱 [{i}/{len(available_models)}] 评估模型: {model_name}")
        print(f"📝 描述: {MODEL_CONFIGS[model_name]['description']}")
        print(f"{'='*60}")
        
        # 创建模型实例
        model = await create_model_instance(model_name, settings)
        if not model:
            continue
        
        # 逐个策略评估
        for j, strategy in enumerate(strategies, 1):
            print(f"\n🎯 [{j}/{len(strategies)}] 策略: {strategy.value}")
            
            result = await evaluate_single_model(
                model_name, model, strategy, test_limit
            )
            all_results.append(result)
            
            # 显示即时结果
            if "error" not in result:
                print(f"   ✅ 准确率: {result['accuracy']:.3f} ({result['accuracy']*100:.1f}%)")
                print(f"   ⏱️  用时: {result['execution_time']:.2f}秒")
            else:
                print(f"   ❌ 失败: {result['error']}")
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 生成综合报告
    generate_comprehensive_report(all_results, total_time, output_dir)
    
    print(f"\n🎉 全部评估完成! 总耗时: {total_time:.2f}秒")
    print(f"📁 结果已保存到: {output_dir}")


def generate_comprehensive_report(results: List[Dict], total_time: float, output_dir: Path):
    """生成综合评估报告"""
    
    # 1. 保存原始结果
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 2. 生成汇总报告
    successful_results = [r for r in results if "error" not in r]
    
    if not successful_results:
        print("⚠️  没有成功的评估结果")
        return
    
    # 按模型分组
    results_by_model = {}
    for result in successful_results:
        model = result["model_name"]
        if model not in results_by_model:
            results_by_model[model] = []
        results_by_model[model].append(result)
    
    # 按策略分组  
    results_by_strategy = {}
    for result in successful_results:
        strategy = result["strategy"]
        if strategy not in results_by_strategy:
            results_by_strategy[strategy] = []
        results_by_strategy[strategy].append(result)
    
    # 生成报告内容
    report_lines = []
    report_lines.append("# 6个主流大模型代词理解能力评估报告\n")
    report_lines.append("## 模型分类")
    report_lines.append("**闭源模型**: GPT-4o, Claude-4.0, Qwen-Max")
    report_lines.append("**开源模型**: Qwen2.5-72B, DeepSeek-V3, Llama-3.2-405B\n")
    report_lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**总耗时**: {total_time:.2f}秒")
    report_lines.append(f"**成功评估**: {len(successful_results)}/{len(results)}")
    report_lines.append("")
    
    # 模型排行榜
    report_lines.append("## 📊 模型整体表现排行\n")
    model_averages = []
    for model, model_results in results_by_model.items():
        avg_accuracy = sum(r["accuracy"] for r in model_results) / len(model_results)
        model_averages.append((model, avg_accuracy, len(model_results)))
    
    model_averages.sort(key=lambda x: x[1], reverse=True)
    
    report_lines.append("| 排名 | 模型 | 平均准确率 | 评估次数 |")
    report_lines.append("|------|------|------------|----------|")
    
    for i, (model, accuracy, count) in enumerate(model_averages, 1):
        description = MODEL_CONFIGS.get(model, {}).get('description', model)
        report_lines.append(f"| {i} | {description} | {accuracy:.3f} ({accuracy*100:.1f}%) | {count} |")
    
    report_lines.append("")
    
    # 策略效果分析
    report_lines.append("## 🎯 提示策略效果分析\n")
    strategy_averages = []
    for strategy, strategy_results in results_by_strategy.items():
        avg_accuracy = sum(r["accuracy"] for r in strategy_results) / len(strategy_results)
        strategy_averages.append((strategy, avg_accuracy, len(strategy_results)))
    
    strategy_averages.sort(key=lambda x: x[1], reverse=True)
    
    report_lines.append("| 策略 | 平均准确率 | 评估次数 |")
    report_lines.append("|------|------------|----------|")
    
    for strategy, accuracy, count in strategy_averages:
        report_lines.append(f"| {strategy} | {accuracy:.3f} ({accuracy*100:.1f}%) | {count} |")
    
    report_lines.append("")
    
    # 详细结果矩阵
    report_lines.append("## 📋 详细结果矩阵\n")
    
    # 获取所有策略和模型
    all_strategies = sorted(set(r["strategy"] for r in successful_results))
    all_models = sorted(results_by_model.keys())
    
    # 创建矩阵表头
    header = "| 模型 | " + " | ".join(all_strategies) + " |"
    separator = "|" + "|".join(["---"] * (len(all_strategies) + 1)) + "|"
    report_lines.append(header)
    report_lines.append(separator)
    
    # 填充矩阵内容
    for model in all_models:
        row = f"| {MODEL_CONFIGS.get(model, {}).get('description', model)[:20]} |"
        for strategy in all_strategies:
            # 查找对应的结果
            result = next(
                (r for r in successful_results 
                 if r["model_name"] == model and r["strategy"] == strategy), 
                None
            )
            if result:
                accuracy = result["accuracy"]
                row += f" {accuracy:.3f} |"
            else:
                row += " - |"
        report_lines.append(row)
    
    # 保存报告
    report_content = "\n".join(report_lines)
    with open(output_dir / "evaluation_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 在控制台显示简要报告
    print("\n" + "="*80)
    print("📊 评估结果概览")
    print("="*80)
    
    print(f"\n🏆 最佳模型: {model_averages[0][0]} ({model_averages[0][1]:.3f})")
    print(f"🎯 最佳策略: {strategy_averages[0][0]} ({strategy_averages[0][1]:.3f})")
    
    print("\n📈 模型排行榜:")
    for i, (model, accuracy, count) in enumerate(model_averages[:3], 1):
        desc = MODEL_CONFIGS.get(model, {}).get('description', model)
        print(f"  {i}. {desc}: {accuracy:.3f}")
    
    print(f"\n📝 详细报告已保存: {output_dir / 'evaluation_report.md'}")


async def quick_test():
    """快速测试（少量用例）"""
    await run_comprehensive_evaluation(
        test_limit=10,
        strategies=[
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.IN_CONTEXT_LEARNING
        ]
    )


async def standard_test():
    """标准测试（中等用例）"""
    await run_comprehensive_evaluation(
        test_limit=50,
        strategies=[
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.IN_CONTEXT_LEARNING
        ]
    )


async def comprehensive_test():
    """全面测试（大量用例）"""
    await run_comprehensive_evaluation(
        test_limit=200,
        strategies=[
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.IN_CONTEXT_LEARNING
        ]
    )


async def main():
    """主函数"""
    print("🤖 6个主流大模型代词理解能力评估系统")
    print("📋 模型清单:")
    print("   闭源: GPT-4o, Claude-4.0, Qwen-Max")
    print("   开源: Qwen2.5-72B, DeepSeek-V3, Llama-3.2-405B")
    print("\n选择评估模式:")
    print("1. 快速测试 (每模型10个用例, 2种策略)")
    print("2. 标准测试 (每模型50个用例, 2种策略)")
    print("3. 全面测试 (每模型200个用例, 2种策略)")
    
    try:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🚀 启动快速测试...")
            await quick_test()
        elif choice == "2":
            print("\n🚀 启动标准测试...")
            await standard_test()
        elif choice == "3":
            print("\n🚀 启动全面测试...")
            await comprehensive_test()
        else:
            print("⚠️  无效选择，使用默认标准测试")
            await standard_test()
            
    except KeyboardInterrupt:
        print("\n⛔ 用户中断评估")
    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        logger.exception("Evaluation failed")


if __name__ == "__main__":
    asyncio.run(main())