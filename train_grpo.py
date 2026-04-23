#!/usr/bin/env python3
"""
GRPO Training Script for OmniTriageEnv
=======================================
Trains an LLM to perform email triage using Group Relative Policy Optimization.
Compatible with Google Colab (T4 GPU) using Unsloth + HF TRL.

Usage (Colab):
    !pip install unsloth trl datasets matplotlib
    !python train_grpo.py

Usage (local dry-run, CPU):
    python train_grpo.py --dry-run --steps 2
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Training dataset — built from the deterministic email corpus
# ---------------------------------------------------------------------------

# We import environment internals directly (no HTTP server needed for training)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emails import EMAILS, GROUND_TRUTH, TASK_EMAIL_IDS
from grader import _keywords_found

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support triage agent. Given a customer email, you must output a JSON object with the appropriate action.

RULES:
- "urgent": fraud, security, outages, legal/compliance, data loss, enterprise critical
- "normal": technical issues, billing questions, exchanges, webhooks, moderate issues
- "low": general inquiries, password resets, feature requests, simple returns

DEPARTMENTS:
- "billing": payments, charges, invoices, refunds, pricing, subscriptions
- "technical": bugs, API errors, outages, data issues, security breaches, migrations, GDPR
- "returns": product returns, exchanges, damaged goods
- "general": feature requests, partnerships, support hours, certifications, media inquiries

OUTPUT FORMAT — return ONLY valid JSON, nothing else:
{"priority": "urgent|normal|low", "department": "billing|technical|general|returns", "escalate": true|false, "response": "2-4 sentence professional reply"}"""


def build_training_prompts() -> List[Dict[str, Any]]:
    """Build prompt-answer pairs from ALL emails across all difficulties."""
    prompts = []
    for difficulty in ["easy", "medium", "hard"]:
        for eid in TASK_EMAIL_IDS[difficulty]:
            email = EMAILS[eid]
            gt = GROUND_TRUTH[eid]

            user_msg = f"""EMAIL ID: {eid}
Subject: {email['subject']}
From: {email['sender']} (Tier: {email.get('sender_tier', 'free')})
Body: {email['body']}

Respond with a JSON object containing: priority, department, escalate (boolean), and response (2-4 sentence reply)."""

            prompts.append({
                "email_id": eid,
                "difficulty": difficulty,
                "prompt": user_msg,
                "ground_truth": gt,
                "email": email,
            })
    return prompts


# ---------------------------------------------------------------------------
# 2. Reward function — scores model completions using env grading logic
# ---------------------------------------------------------------------------

def parse_completion(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON from model output."""
    if not text:
        return None
    # Strip markdown fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Nested braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def compute_reward(completion_text: str, ground_truth: Dict[str, Any]) -> float:
    """
    Score a single model completion against ground truth.
    Returns a reward in [-1.0, 1.0].

    Components (matching the environment's grading):
      - Priority match:    +0.25 correct, -0.15 wrong
      - Department match:  +0.25 correct, -0.15 wrong
      - Escalation match:  +0.20 correct, -0.20 wrong
      - Response keywords: +0.0 to +0.20 (fractional)
      - Valid JSON bonus:   +0.10
    """
    parsed = parse_completion(completion_text)

    # If we can't parse JSON at all, heavy penalty
    if parsed is None:
        return -0.50

    reward = 0.10  # JSON parse bonus

    # Priority
    pred_priority = str(parsed.get("priority", "")).lower().strip()
    gt_priority = ground_truth["priority"]
    if pred_priority == gt_priority:
        reward += 0.25
    else:
        reward -= 0.15

    # Department
    pred_dept = str(parsed.get("department", "")).lower().strip()
    gt_dept = ground_truth["department"]
    if pred_dept == gt_dept:
        reward += 0.25
    else:
        reward -= 0.15

    # Escalation
    pred_esc = parsed.get("escalate", False)
    if isinstance(pred_esc, str):
        pred_esc = pred_esc.lower() in ("true", "yes", "1")
    gt_esc = ground_truth.get("escalate", False)
    if bool(pred_esc) == bool(gt_esc):
        reward += 0.20
    else:
        reward -= 0.20

    # Response keyword coverage
    response_text = str(parsed.get("response", ""))
    keywords = ground_truth.get("response_keywords", [])
    if keywords and response_text:
        kw_score = _keywords_found(response_text, keywords)
        reward += 0.20 * kw_score

    return max(-1.0, min(1.0, round(reward, 4)))


def reward_function_batch(completions: List[str], prompts_metadata: List[Dict]) -> List[float]:
    """Score a batch of completions. Used by GRPO trainer."""
    rewards = []
    for comp, meta in zip(completions, prompts_metadata):
        r = compute_reward(comp, meta["ground_truth"])
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# 3. Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(model, tokenizer, prompts: List[Dict], max_samples: int = 30) -> Dict[str, Any]:
    """Run the model on all prompts and compute average reward."""
    from transformers import pipeline
    import torch

    results = []
    samples = prompts[:max_samples]

    for item in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        r = compute_reward(completion, item["ground_truth"])
        parsed = parse_completion(completion)

        results.append({
            "email_id": item["email_id"],
            "difficulty": item["difficulty"],
            "reward": r,
            "parsed_ok": parsed is not None,
            "completion_preview": completion[:200],
        })

    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0
    parse_rate = sum(1 for r in results if r["parsed_ok"]) / len(results) if results else 0

    per_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            per_difficulty[diff] = {
                "avg_reward": round(sum(r["reward"] for r in diff_results) / len(diff_results), 4),
                "parse_rate": round(sum(1 for r in diff_results if r["parsed_ok"]) / len(diff_results), 4),
                "count": len(diff_results),
            }

    return {
        "avg_reward": round(avg_reward, 4),
        "parse_rate": round(parse_rate, 4),
        "per_difficulty": per_difficulty,
        "details": results,
    }


# ---------------------------------------------------------------------------
# 4. GRPO Training with TRL
# ---------------------------------------------------------------------------

def create_training_dataset(prompts: List[Dict]):
    """Convert prompts to HF Dataset format for GRPOTrainer."""
    from datasets import Dataset

    records = []
    for item in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        records.append({
            "prompt": messages,
            "email_id": item["email_id"],
            "difficulty": item["difficulty"],
            "gt_priority": item["ground_truth"]["priority"],
            "gt_department": item["ground_truth"]["department"],
            "gt_escalate": item["ground_truth"].get("escalate", False),
            "gt_keywords": json.dumps(item["ground_truth"].get("response_keywords", [])),
        })

    return Dataset.from_list(records)


def make_reward_fn(tokenizer):
    """Create the reward function closure for GRPOTrainer."""

    def reward_fn(completions, **kwargs):
        """
        GRPOTrainer reward function.
        completions: list of generated text strings
        kwargs contains the prompt metadata columns from the dataset
        """
        rewards = []
        gt_priorities = kwargs.get("gt_priority", [])
        gt_departments = kwargs.get("gt_department", [])
        gt_escalates = kwargs.get("gt_escalate", [])
        gt_keywords_list = kwargs.get("gt_keywords", [])

        for i, comp in enumerate(completions):
            # Extract text from completion
            if hasattr(comp, "text"):
                text = comp.text
            elif isinstance(comp, list):
                # Chat format — extract assistant content
                text = ""
                for msg in comp:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        text = msg.get("content", "")
                        break
                if not text and comp:
                    text = str(comp[-1]) if comp else ""
            else:
                text = str(comp)

            gt = {
                "priority": gt_priorities[i] if i < len(gt_priorities) else "normal",
                "department": gt_departments[i] if i < len(gt_departments) else "general",
                "escalate": gt_escalates[i] if i < len(gt_escalates) else False,
                "response_keywords": json.loads(gt_keywords_list[i]) if i < len(gt_keywords_list) else [],
            }
            r = compute_reward(text, gt)
            rewards.append(r)

        return rewards

    return reward_fn


def plot_training_results(
    baseline_metrics: Dict,
    trained_metrics: Dict,
    training_rewards: List[float],
    output_dir: str,
):
    """Generate reward curve and comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Training reward curve
    if training_rewards:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(range(len(training_rewards)), training_rewards, color="#4A90D9", linewidth=1.5, alpha=0.4, label="Per-step reward")

        # Moving average
        window = min(10, len(training_rewards) // 3 + 1)
        if window > 1 and len(training_rewards) > window:
            moving_avg = []
            for i in range(len(training_rewards)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(training_rewards[start:i+1]) / (i - start + 1))
            ax.plot(range(len(moving_avg)), moving_avg, color="#E74C3C", linewidth=2.5, label=f"Moving avg (window={window})")

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title("OmniTriageEnv — GRPO Training Reward Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150)
        plt.close(fig)
        print(f"✅ Saved reward_curve.png")

    # 2. Before/After comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall comparison
    labels = ["Baseline", "After Training"]
    avg_rewards = [baseline_metrics["avg_reward"], trained_metrics["avg_reward"]]
    parse_rates = [baseline_metrics["parse_rate"], trained_metrics["parse_rate"]]

    colors = ["#95A5A6", "#27AE60"]
    axes[0].bar(labels, avg_rewards, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Average Reward", fontsize=12)
    axes[0].set_title("Overall Reward: Before vs After", fontsize=13, fontweight="bold")
    axes[0].set_ylim(-0.5, 1.0)
    for i, v in enumerate(avg_rewards):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold", fontsize=12)

    # Per-difficulty comparison
    diffs = ["easy", "medium", "hard"]
    x_pos = range(len(diffs))
    width = 0.35

    baseline_by_diff = [baseline_metrics.get("per_difficulty", {}).get(d, {}).get("avg_reward", 0) for d in diffs]
    trained_by_diff = [trained_metrics.get("per_difficulty", {}).get(d, {}).get("avg_reward", 0) for d in diffs]

    bars1 = axes[1].bar([x - width/2 for x in x_pos], baseline_by_diff, width, label="Baseline", color="#95A5A6")
    bars2 = axes[1].bar([x + width/2 for x in x_pos], trained_by_diff, width, label="Trained", color="#27AE60")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels([d.capitalize() for d in diffs], fontsize=11)
    axes[1].set_ylabel("Average Reward", fontsize=12)
    axes[1].set_title("Per-Difficulty Reward Comparison", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(-0.5, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)
    print(f"✅ Saved comparison.png")

    # 3. Print text comparison table
    print("\n" + "=" * 70)
    print("📊 BEFORE vs AFTER TRAINING COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'Trained':>12} {'Delta':>12}")
    print("-" * 70)
    print(f"{'Avg Reward':<25} {baseline_metrics['avg_reward']:>12.4f} {trained_metrics['avg_reward']:>12.4f} {trained_metrics['avg_reward'] - baseline_metrics['avg_reward']:>+12.4f}")
    print(f"{'JSON Parse Rate':<25} {baseline_metrics['parse_rate']:>12.1%} {trained_metrics['parse_rate']:>12.1%} {trained_metrics['parse_rate'] - baseline_metrics['parse_rate']:>+12.1%}")
    for d in diffs:
        b = baseline_metrics.get("per_difficulty", {}).get(d, {}).get("avg_reward", 0)
        t = trained_metrics.get("per_difficulty", {}).get(d, {}).get("avg_reward", 0)
        print(f"  {d.capitalize() + ' Reward':<23} {b:>12.4f} {t:>12.4f} {t - b:>+12.4f}")
    print("=" * 70)

    # Save comparison as JSON
    comparison = {
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "improvement": {
            "avg_reward_delta": round(trained_metrics["avg_reward"] - baseline_metrics["avg_reward"], 4),
            "parse_rate_delta": round(trained_metrics["parse_rate"] - baseline_metrics["parse_rate"], 4),
        }
    }
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"✅ Saved comparison_results.json")


# ---------------------------------------------------------------------------
# 5. Main training pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for OmniTriageEnv")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--output-dir", default="./training_output",
                        help="Output directory for model + plots")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with minimal steps (CPU-safe)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only run evaluation (no training)")
    args = parser.parse_args()

    if args.dry_run:
        args.steps = 2
        args.batch_size = 2
        args.num_generations = 2
        print("🔧 DRY RUN MODE — minimal config for testing\n")

    print("=" * 60)
    print("🚀 OmniTriageEnv GRPO Training Pipeline")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Steps:       {args.steps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Generations: {args.num_generations}")
    print(f"  LR:          {args.lr}")
    print(f"  Output:      {args.output_dir}")
    print()

    # ── Step 1: Build training dataset ────────────────────────────────────
    print("📧 Building training dataset from email corpus...")
    prompts = build_training_prompts()
    print(f"   {len(prompts)} email prompts across easy/medium/hard\n")

    # ── Step 2: Load model ────────────────────────────────────────────────
    print("🤖 Loading model with Unsloth...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        use_unsloth = True
        print("   ✅ Model loaded with Unsloth (4-bit LoRA)\n")
    except ImportError:
        print("   ⚠️  Unsloth not available, falling back to transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32 if args.dry_run else torch.float16,
            device_map="auto" if not args.dry_run else "cpu",
        )
        use_unsloth = False
        print("   ✅ Model loaded with transformers\n")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 3: Baseline evaluation ───────────────────────────────────────
    print("📊 Running baseline evaluation...")
    eval_count = 6 if args.dry_run else 30
    baseline_metrics = evaluate_baseline(model, tokenizer, prompts, max_samples=eval_count)
    print(f"   Baseline avg reward: {baseline_metrics['avg_reward']:.4f}")
    print(f"   Baseline parse rate: {baseline_metrics['parse_rate']:.1%}\n")

    if args.skip_training:
        print("⏭️  Skipping training (--skip-training flag)")
        plot_training_results(baseline_metrics, baseline_metrics, [], args.output_dir)
        return

    # ── Step 4: Create dataset & train ────────────────────────────────────
    print("🎯 Starting GRPO training...")
    dataset = create_training_dataset(prompts)
    reward_fn = make_reward_fn(tokenizer)

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3 if not args.dry_run else 1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=300,
        max_prompt_length=1500,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=50 if not args.dry_run else args.steps,
        report_to="none",
        bf16=not args.dry_run,
        gradient_accumulation_steps=2 if not args.dry_run else 1,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Collect rewards during training via log history
    train_result = trainer.train()

    # Extract reward history from trainer logs
    training_rewards = []
    if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        for log_entry in trainer.state.log_history:
            if "reward" in log_entry:
                training_rewards.append(log_entry["reward"])
            elif "rewards/mean" in log_entry:
                training_rewards.append(log_entry["rewards/mean"])

    print(f"\n   ✅ Training complete! {len(training_rewards)} reward data points logged.\n")

    # ── Step 5: Post-training evaluation ──────────────────────────────────
    print("📊 Running post-training evaluation...")
    trained_metrics = evaluate_baseline(model, tokenizer, prompts, max_samples=eval_count)
    print(f"   Trained avg reward: {trained_metrics['avg_reward']:.4f}")
    print(f"   Trained parse rate: {trained_metrics['parse_rate']:.1%}\n")

    # ── Step 6: Generate plots & comparison ───────────────────────────────
    print("📈 Generating plots and comparison...")
    plot_training_results(baseline_metrics, trained_metrics, training_rewards, args.output_dir)

    # ── Step 7: Save model ────────────────────────────────────────────────
    print("\n💾 Saving trained model...")
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))
    print(f"   ✅ Saved to {args.output_dir}/model")

    print("\n" + "=" * 60)
    print("🏁 TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Plots:  {args.output_dir}/reward_curve.png")
    print(f"          {args.output_dir}/comparison.png")
    print(f"  Model:  {args.output_dir}/model/")
    print(f"  Data:   {args.output_dir}/comparison_results.json")
    print()


if __name__ == "__main__":
    main()
