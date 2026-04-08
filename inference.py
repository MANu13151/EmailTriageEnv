#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# -- Configuration --------------------------------------------------------------
API_BASE_URL  = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME    = os.environ.get("MODEL_NAME", "moonshotai/kimi-k2-instruct")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

BENCHMARK     = "email-triage-env"
MAX_STEPS     = 60

llm = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=HF_TOKEN or "placeholder",
)

# -- Logging helpers ------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_safe = action.replace("\n", " ")[:120]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# -- Environment client ---------------------------------------------------------
class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()

    def reset(self, difficulty: str) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/reset", json={"difficulty": difficulty}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()

    def grade(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/grade", timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

# -- System prompt --------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert email triage agent for a customer support team.
For each email perform these 4 actions in order:
  1. classify_priority
  2. assign_department
  3. draft_response
  4. archive

CRITICAL RULES:
- ALWAYS use the exact email_id shown in the current email
- NEVER repeat an action already taken on this email
- Check NEXT ACTIONS STILL NEEDED and pick the first one
- Do NOT escalate unless: data loss, fraud, security breach, GDPR, chargeback, media inquiry, critical enterprise outage

DEPARTMENT GUIDE:
- billing:   payment issues, invoices, refunds, pricing disputes, subscriptions
- technical: bugs, API errors, integrations, data issues, performance problems
- returns:   product returns, exchanges, damaged goods
- general:   feature requests, inquiries, certifications, partnerships, press

PRIORITY GUIDE:
- urgent: production down, data loss, fraud, security breach, legal/compliance, chargeback, media inquiry
- normal: bugs affecting work, billing disputes, API limits, migration questions
- low:    feature requests, general inquiries, certifications, return requests

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "action_type": "<classify_priority|assign_department|draft_response|escalate|archive|skip>",
  "email_id": "<email id>",
  "priority": "<urgent|normal|low or null>",
  "department": "<billing|technical|general|returns or null>",
  "response_text": "<draft response or null>",
  "reasoning": "<one sentence>"
}"""

# -- LLM call -------------------------------------------------------------------
def call_llm(observation: Dict[str, Any], last_reward: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    email = observation.get("current_email")
    if email is None:
        return {"action_type": "archive", "email_id": "", "priority": None, "department": None, "response_text": None}

    email_id       = email["email_id"]
    action_history = observation.get("action_history", [])
    done_for_email = [h["action_type"] for h in action_history if h.get("email_id") == email_id]
    remaining      = [a for a in ["classify_priority", "assign_department", "draft_response", "archive"]
                      if a not in done_for_email]

    reward_context = ""
    if last_reward and last_reward.get("penalty_reason"):
        reward_context = f"\nLast penalty: {last_reward['penalty_reason']}"

    user_message = f"""CURRENT EMAIL:
ID: {email_id}
Subject: {email.get('subject', '')}
From: {email.get('sender', '')} ({email.get('sender_tier', '')} tier)
{f"Category hint: {email.get('category_hint')}" if email.get('category_hint') else ""}

Body:
{email.get('body', '')}

SESSION STATE:
- Processed: {observation.get('processed_count', 0)} | Remaining: {observation.get('queue_length', 0)}
- Actions already done on this email: {done_for_email if done_for_email else "none"}
- NEXT ACTIONS STILL NEEDED (pick first): {remaining}
{reward_context}

Respond with JSON for the next action on email {email_id}."""

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        action = json.loads(raw)
        action["email_id"] = email_id
        time.sleep(1)
        return action
    except Exception as e:
        time.sleep(3)
        return {"action_type": "skip", "email_id": email_id, "priority": None, "department": None, "response_text": None}

# -- Episode runner -------------------------------------------------------------
def run_episode(client: EnvClient, difficulty: str) -> Dict[str, Any]:
    task_name = f"email-triage-{difficulty}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    last_reward: Optional[Dict[str, Any]] = None

    try:
        obs = client.reset(difficulty)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action = call_llm(obs, last_reward)
            action_str = action.get("action_type", "skip")

            try:
                result     = client.step(action)
                reward_obj = result.get("reward", {})
                reward_val = float(reward_obj.get("value", 0.0))
                done       = result.get("done", False)
                error      = reward_obj.get("penalty_reason", None)
                obs        = result.get("observation", obs)
                last_reward = reward_obj
            except Exception as e:
                reward_val = 0.0
                done       = False
                error      = str(e)

            rewards.append(reward_val)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)

            if done:
                break

        grade   = client.grade()
        score   = float(grade.get("score", 0.0))
        success = bool(grade.get("passed", False))

    except Exception as e:
        error_msg = str(e)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"difficulty": difficulty, "score": score, "passed": success, "steps": steps_taken}

# -- Main -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="all")
    args = parser.parse_args()

    client = EnvClient(API_BASE_URL)

    # Wait for environment to be ready
    for attempt in range(10):
        if client.health():
            break
        print(f"[DEBUG] Waiting for environment... attempt {attempt+1}", flush=True)
        time.sleep(5)
    else:
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
        sys.exit(1)

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    all_scores = []
    for diff in difficulties:
        result = run_episode(client, diff)
        all_scores.append(result["score"])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    output_path = os.environ.get("RESULTS_OUTPUT_PATH", "results.json")
    with open(output_path, "w") as f:
        json.dump({"model": MODEL_NAME, "average_score": avg}, f, indent=2)

if __name__ == "__main__":
    main()
