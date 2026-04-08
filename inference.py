#!/usr/bin/env python3
"""
inference.py — Baseline agent for EmailTriageEnv.

Usage:
    API_BASE_URL=http://localhost:8080 \
    MODEL_NAME=gpt-4o-mini \
    python inference.py [--difficulty easy|medium|hard] [--verbose]

Environment variables:
    API_BASE_URL   Base URL of the OpenEnv server OR OpenAI-compatible LLM endpoint
    MODEL_NAME     Model identifier (e.g., gpt-4o-mini, meta-llama/Llama-3-8b-instruct)
    HF_TOKEN       Hugging Face token (used when MODEL_NAME points to HF-hosted model)

Design:
  - Pure zero-shot prompting (no fine-tuning required)
  - Deterministic temperature=0.0
  - Structured JSON output via system prompt
  - Full episode runs under 20 minutes on 2 vCPU / 8 GB RAM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080")
MODEL_NAME   = os.environ.get("MODEL_NAME", "moonshotai/kimi-k2-instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# Build OpenAI client — works with OpenAI API or any OpenAI-compatible endpoint
_openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key     = os.environ.get("OPENAI_API_KEY") or HF_TOKEN or "placeholder"

llm = OpenAI(base_url=_openai_base, api_key=_api_key)


# ── Environment HTTP client ───────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, difficulty: str) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/reset", json={"difficulty": difficulty}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json()

    def grade(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/grade", timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent for a customer support team.
Your job: read each email and perform EXACTLY these 4 actions in order, then stop:
  1. classify_priority
  2. assign_department  
  3. draft_response
  4. archive

CRITICAL RULES:
- ALWAYS use the exact email_id shown in the current email (e.g. "M001", "H003")
- NEVER repeat an action you have already taken on this email
- Check "Actions already taken on this email" before deciding — skip any already done
- After archive, wait for the next email — do NOT act on the same email again
- Do NOT escalate unless the email clearly involves: data loss, fraud, security breach,
  GDPR/legal request, chargeback, media inquiry, or critical enterprise outage


AVAILABLE ACTIONS:
1. classify_priority  — classify email as "urgent", "normal", or "low"
2. assign_department  — assign to "billing", "technical", "general", or "returns"
3. draft_response     — write an appropriate professional reply
4. escalate           — escalate to senior human agent (only when truly necessary)
5. archive            — close/archive the email (do this last)
6. skip               — defer email (use sparingly, penalized)

DEPARTMENT GUIDE:
- billing:   payment issues, invoices, refunds, pricing disputes, subscriptions
- technical: bugs, API errors, integrations, data issues, performance problems
- returns:   product returns, exchanges, damaged goods
- general:   feature requests, inquiries, certifications, partnerships, press

PRIORITY GUIDE:
- urgent: production down, data loss, fraud, security breach, legal/compliance, chargeback, media inquiry
- normal: bugs affecting work, billing disputes (non-fraud), API limits, migration questions
- low:    feature requests, general inquiries, certifications, return requests without urgency

ESCALATION GUIDE — escalate ONLY for:
- Data loss or corruption
- Security breach or fraud
- Legal/compliance/GDPR requests
- Chargeback disputes
- Media/press inquiries
- Critical outages at enterprise customers

OPTIMAL SEQUENCE PER EMAIL:
  classify_priority → assign_department → draft_response → [escalate if needed] → archive

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no explanation:
{
  "action_type": "<one of the 6 action types>",
  "email_id": "<email id>",
  "priority": "<urgent|normal|low or null>",
  "department": "<billing|technical|general|returns or null>",
  "response_text": "<draft response text or null>",
  "reasoning": "<one sentence explanation>"
}
"""


# ── LLM agent ─────────────────────────────────────────────────────────────────

def call_llm(
    observation: Dict[str, Any],
    last_reward: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call the LLM with current observation and return a parsed action dict.
    Temperature=0.0 for full reproducibility.
    """
    email = observation.get("current_email")
    if email is None:
        # No email left — should not happen, but safe fallback
        return {
            "action_type": "archive",
            "email_id": "",
            "priority": None,
            "department": None,
            "response_text": None,
        }

    email_id = email["email_id"]

    # Build context for prompt
    processed_count = observation.get("processed_count", 0)
    queue_length = observation.get("queue_length", 0)
    session_score = observation.get("session_score", 0.0)
    skip_budget = observation.get("skip_budget", 0)
    action_history = observation.get("action_history", [])

    # Summarise what we've already done for this email
    done_for_email = [
    h["action_type"] for h in action_history if h.get("email_id") == email_id
        ]
    remaining_actions = [a for a in ["classify_priority","assign_department","draft_response","archive"] 
                     if a not in done_for_email]

    reward_context = ""
    if last_reward:
        reward_context = (
            f"\nLast reward: {last_reward.get('value', 0):.3f} | "
            f"Breakdown: {json.dumps(last_reward.get('breakdown', {}))}"
        )
        if last_reward.get("penalty_reason"):
            reward_context += f" | PENALTY: {last_reward['penalty_reason']}"

    user_message = f"""CURRENT EMAIL:
ID: {email_id}
Subject: {email.get('subject', '')}
From: {email.get('sender', '')} ({email.get('sender_tier', '')} tier)
Received: {email.get('received_at', '')}
{f"Category hint: {email.get('category_hint')}" if email.get('category_hint') else ""}

Body:
{email.get('body', '')}

SESSION STATE:
- Emails processed: {processed_count} | Remaining: {queue_length}
- Session score: {session_score:.3f}
- Skip budget remaining: {skip_budget}
- Actions already completed on this email: {done_for_email if done_for_email else "none"}
- NEXT ACTIONS STILL NEEDED (pick the first one): {remaining_actions}
{reward_context}

Decide the NEXT single action to take on email {email_id}."""

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        action = json.loads(raw)
        # Ensure email_id is set correctly
        action["email_id"] = email_id
        time.sleep(1)
        return action

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] LLM parse error: {e}. Defaulting to skip.")
        time.sleep(3)
        return {
            "action_type": "skip",
            "email_id": email_id,
            "priority": None,
            "department": None,
            "response_text": None,
        }


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    client: EnvClient,
    difficulty: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one full episode and return graded results."""
    print(f"\n{'='*60}")
    print(f"  Running episode: difficulty={difficulty}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*60}")

    obs = client.reset(difficulty)
    last_reward: Optional[Dict[str, Any]] = None

    max_steps = 100   # safety cap
    step_count = 0
    start_time = time.time()

    while not obs.get("done", False) and step_count < max_steps:
        if verbose:
            email = obs.get("current_email", {})
            if email:
                print(f"\n[Step {step_count}] Email: {email.get('email_id')} — {email.get('subject', '')[:50]}")

        action = call_llm(obs, last_reward)

        if verbose:
            print(f"  → Action: {action.get('action_type')} | priority={action.get('priority')} | dept={action.get('department')}")

        result = client.step(action)
        last_reward = result.get("reward", {})
        obs = result.get("observation", obs)
        step_count += 1

        if verbose:
            reward_val = last_reward.get("value", 0)
            cumulative = last_reward.get("cumulative", 0)
            print(f"  ← Reward: {reward_val:.3f} (cumulative: {cumulative:.3f})")
            if last_reward.get("penalty_reason"):
                print(f"  ⚠ Penalty: {last_reward['penalty_reason']}")

    elapsed = time.time() - start_time
    grade = client.grade()

    print(f"\n{'─'*60}")
    print(f"  Episode complete | Steps: {step_count} | Time: {elapsed:.1f}s")
    print(f"  Final score: {grade.get('score', 0.0):.4f}")
    print(f"  Passed: {grade.get('passed', False)}")
    print(f"  Base score: {grade.get('base_score', 0.0):.4f}")
    print(f"  Invalid action penalty: {grade.get('invalid_penalty', 0.0):.4f}")
    print(f"  Skip penalty: {grade.get('skip_penalty', 0.0):.4f}")

    if verbose:
        print("\n  Per-email scores:")
        for eid, s in (grade.get("per_email_scores") or {}).items():
            print(f"    {eid}: {s:.4f}")

    return {
        "difficulty": difficulty,
        "score": grade.get("score", 0.0),
        "passed": grade.get("passed", False),
        "steps": step_count,
        "elapsed_seconds": round(elapsed, 2),
        "grade_detail": grade,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EmailTriageEnv baseline inference")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task difficulty (default: all — runs all three)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose step output")
    args = parser.parse_args()

    client = EnvClient(ENV_BASE_URL)

    print(f"Checking environment health at {ENV_BASE_URL}...")
    if not client.health():
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}")
        print("Start the server first: python server.py")
        sys.exit(1)
    print("Environment is healthy.\n")

    difficulties = (
        ["easy", "medium", "hard"]
        if args.difficulty == "all"
        else [args.difficulty]
    )

    all_results: List[Dict[str, Any]] = []
    overall_start = time.time()

    for diff in difficulties:
        result = run_episode(client, diff, verbose=args.verbose)
        all_results.append(result)

    total_time = time.time() - overall_start

    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Difficulty':<12} {'Score':>8} {'Passed':>8} {'Steps':>6} {'Time(s)':>8}")
    print(f"  {'-'*44}")
    scores = []
    for r in all_results:
        score = r["score"]
        scores.append(score)
        passed = "✓" if r["passed"] else "✗"
        print(f"  {r['difficulty']:<12} {score:>8.4f} {passed:>8} {r['steps']:>6} {r['elapsed_seconds']:>8.1f}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average score: {avg:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}\n")

    # Write results to file for CI/CD pipelines
    output_path = os.environ.get("RESULTS_OUTPUT_PATH", "results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "results": all_results,
                "average_score": avg if scores else 0.0,
                "total_elapsed_seconds": round(total_time, 2),
            },
            f,
            indent=2,
        )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
