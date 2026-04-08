#!/usr/bin/env python3
import json, os, sys, time, argparse
from typing import Any, Dict, List, Optional

def _clamp_score(s):
    """Clamp score to strict open interval (0, 1)."""
    return max(0.001, min(float(s), 0.999))

try:
    import requests
    from openai import OpenAI
except ImportError as e:
    print(f"[END] success=false steps=0 score=0.001 rewards=0.00", flush=True)
    sys.exit(0)

# 1. Capture the Hackathon's injected LLM Proxy credentials
LLM_PROXY_URL = os.environ.get("API_BASE_URL") 
LLM_API_KEY   = os.environ.get("API_KEY")
MODEL_NAME    = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-8b-instruct") # Ensure you use the model they expect

# 2. Point to your local Triage Environment server (running on port 7860 in the Space/Docker)
# Do NOT use API_BASE_URL for the EnvClient; that variable is for the LLM.
TRIAGE_ENV_URL = "http://localhost:7860"

# Metadata for logging
BENCHMARK    = "email-triage-env"
MAX_STEPS    = 60

# 3. Initialize the OpenAI client using the ORGANIZER'S proxy and key
try:
    llm = OpenAI(
        base_url=LLM_PROXY_URL, 
        api_key=LLM_API_KEY
    )
except Exception:
    llm = None

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = error.replace("\n"," ")[:200] if error else "null"
    a = str(action).replace("\n"," ")[:100]
    d = "true" if done else "false"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} done={d} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards) if rewards else "0.00"
    s = "true" if success else "false"
    score = _clamp_score(score)
    print(f"[END] success={s} steps={steps} score={score:.3f} rewards={r}", flush=True)

class EnvClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self):
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=15)
            return r.status_code == 200
        except:
            return False

    def reset(self, difficulty="easy"):
        try:
            r = self.session.post(f"{self.base_url}/reset", json={"difficulty": difficulty}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"done": True, "current_email": None, "queue_length": 0,
                    "processed_count": 0, "session_score": 0.0, "skip_budget": 0,
                    "action_history": [], "task_id": difficulty,
                    "task_difficulty": difficulty, "step_number": 0}

    def step(self, action):
        try:
            r = self.session.post(f"{self.base_url}/step", json=action, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"reward": {"value": 0.0, "penalty_reason": str(e)}, "done": False, "observation": {}}

    def grade(self):
        try:
            r = self.session.get(f"{self.base_url}/grade", timeout=30)
            r.raise_for_status()
            return r.json()
        except:
            return {"score": 0.001, "passed": False}

SYSTEM_PROMPT = """You are an expert email triage agent.
For each email perform these 4 actions in order:
  1. classify_priority
  2. assign_department
  3. draft_response
  4. archive

RULES:
- Use exact email_id shown
- Never repeat an action already done
- Pick FIRST action from NEXT ACTIONS STILL NEEDED
- Escalate ONLY for: data loss, fraud, security breach, GDPR, chargeback, media inquiry, critical outage

DEPARTMENTS: billing, technical, returns, general
PRIORITIES: urgent, normal, low

OUTPUT only valid JSON:
{"action_type":"classify_priority","email_id":"E001","priority":"urgent","department":null,"response_text":null,"reasoning":"one sentence"}"""

def call_llm(observation, last_reward=None):
    email = observation.get("current_email")
    if not email:
        return {"action_type": "archive", "email_id": "", "priority": None, "department": None, "response_text": None}
    email_id = email.get("email_id", "")
    history = observation.get("action_history", [])
    done_for = [h["action_type"] for h in history if h.get("email_id") == email_id]
    remaining = [a for a in ["classify_priority","assign_department","draft_response","archive"] if a not in done_for]
    penalty = f"\nLast penalty: {last_reward['penalty_reason']}" if last_reward and last_reward.get("penalty_reason") else ""
    user_msg = f"""EMAIL ID: {email_id}
Subject: {email.get('subject','')}
From: {email.get('sender','')} ({email.get('sender_tier','')} tier)
{f"Hint: {email.get('category_hint')}" if email.get('category_hint') else ""}
Body: {email.get('body','')}
Already done: {done_for if done_for else "none"}
NEXT ACTIONS STILL NEEDED (pick first): {remaining}{penalty}
Respond JSON for email {email_id}."""
    if llm is None:
        return {"action_type": "skip", "email_id": email_id, "priority": None, "department": None, "response_text": None}
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_msg}],
            temperature=0.0, max_tokens=512)
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        action = json.loads(raw)
        action["email_id"] = email_id
        time.sleep(1)
        return action
    except Exception:
        time.sleep(3)
        return {"action_type": "skip", "email_id": email_id, "priority": None, "department": None, "response_text": None}

def run_episode(client, difficulty):
    task = f"email-triage-{difficulty}"
    rewards, steps_taken, score, success = [], 0, 0.0, False
    last_reward = None
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = client.reset(difficulty)
        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break
            action = call_llm(obs, last_reward)
            action_str = action.get("action_type", "skip")
            result = client.step(action)
            reward_obj = result.get("reward", {})
            reward_val = float(reward_obj.get("value", 0.0))
            done = bool(result.get("done", False))
            error = reward_obj.get("penalty_reason", None)
            new_obs = result.get("observation", {})
            if new_obs:
                obs = new_obs
            last_reward = reward_obj
            rewards.append(reward_val)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)
            if done:
                break
        grade = client.grade()
        score = _clamp_score(grade.get("score", 0.001))
        success = bool(grade.get("passed", False))
    except Exception as e:
        log_step(step=max(steps_taken,1), action="error", reward=0.0, done=True, error=str(e)[:200])
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"difficulty": difficulty, "score": score, "passed": success}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy","medium","hard","all"], default="all")
    args = parser.parse_args()
    client = EnvClient(TRIAGE_ENV_URL)
    ready = False
    for i in range(12):
        if client.health():
            ready = True
            break
        time.sleep(5)
    difficulties = ["easy","medium","hard"] if args.difficulty == "all" else [args.difficulty]
    if not ready:
        for diff in difficulties:
            log_start(task=f"email-triage-{diff}", env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="skip", reward=0.0, done=True, error="env_not_reachable")
            log_end(success=False, steps=1, score=0.001, rewards=[0.0])
        sys.exit(0)
    scores = []
    for diff in difficulties:
        r = run_episode(client, diff)
        scores.append(r["score"])
    try:
        with open(os.environ.get("RESULTS_OUTPUT_PATH","results.json"),"w") as f:
            json.dump({"model": MODEL_NAME, "average_score": sum(scores)/len(scores)}, f)
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.001 rewards=0.00", flush=True)
        sys.exit(0)
