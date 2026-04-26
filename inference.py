#!/usr/bin/env python3
"""
Inference Script for OmniTriageEnv
====================================
MANDATORY ENV VARS (injected by hackathon harness):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
import json, os, sys, time, re, argparse
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

# ── Environment variables (hackathon-injected) ───────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-instruct")

# ── Local environment server (runs on port 7860 inside Docker) ───────────────
TRIAGE_ENV_URL = "http://localhost:7860"

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK    = "omni-triage-env"
MAX_STEPS    = 60

# ── OpenAI client for LLM calls ─────────────────────────────────────────────
try:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    llm = None


# ── Structured logging (MUST match hackathon format exactly) ─────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    e = error.replace("\n", " ")[:200] if error else "null"
    a = str(action).replace("\n", " ")[:100]
    d = "true" if done else "false"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} done={d} error={e}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r = ",".join(f"{x:.2f}" for x in rewards) if rewards else "0.00"
    s = "true" if success else "false"
    score = _clamp_score(score)
    print(f"[END] success={s} steps={steps} score={score:.3f} rewards={r}", flush=True)


# ── Environment client (HTTP to local server) ────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=15)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, difficulty: str = "easy") -> dict:
        try:
            r = self.session.post(
                f"{self.base_url}/reset",
                json={"difficulty": difficulty},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return {
                "done": True, "current_email": None, "queue_length": 0,
                "processed_count": 0, "session_score": 0.0, "skip_budget": 0,
                "action_history": [], "task_id": difficulty,
                "task_difficulty": difficulty, "step_number": 0,
            }

    def step(self, action: dict) -> dict:
        try:
            r = self.session.post(
                f"{self.base_url}/step",
                json=action,
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {
                "reward": {"value": 0.0, "penalty_reason": str(e)},
                "done": False,
                "observation": {},
            }

    def grade(self) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/grade", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"score": 0.001, "passed": False}


# ── Escalation heuristics ────────────────────────────────────────────────────
# These keywords/phrases in an email strongly suggest escalation is needed.
ESCALATION_SIGNALS = [
    "fraud", "unauthorized", "chargeback", "security breach", "data loss",
    "gdpr", "legal", "compliance", "media inquiry", "journalist", "press",
    "critical outage", "production down", "data exposed", "pii",
    "regulatory", "article 17", "data deletion",
]


def _should_escalate(email: dict) -> bool:
    """Heuristic: check if the email strongly signals need for escalation."""
    text = (email.get("subject", "") + " " + email.get("body", "")).lower()
    tier = email.get("sender_tier", "free")

    # Count how many escalation signals appear
    signal_count = sum(1 for sig in ESCALATION_SIGNALS if sig in text)

    # Strong signals: 2+ keywords, or enterprise tier + 1 keyword
    if signal_count >= 2:
        return True
    if tier == "enterprise" and signal_count >= 1:
        return True

    return False


# ── LLM prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support email triage agent. Your job is to process each email through these actions in order:

1. classify_priority — Set urgency level
2. assign_department — Route to the correct team
3. draft_response — Write a professional customer reply
4. archive — Close the email

CLASSIFICATION RULES:
- "urgent": billing disputes, fraud, security incidents, production outages, data loss, legal/compliance requests, enterprise customers with critical issues
- "normal": technical issues, moderate billing questions, exchange requests, webhook problems, feature inquiries from paying customers
- "low": general inquiries, simple returns, password resets, feature requests, informational questions

DEPARTMENT RULES:
- "billing": payment disputes, charges, invoices, pricing, subscription issues, chargebacks, pro-rata billing, AND refunds due to service failures (outages, broken features, poor service quality, compensation requests). If a customer wants a refund because your SERVICE failed them, route to billing.
- "technical": bugs, API errors, outages, data loss, migrations, webhooks, browser issues, security breaches, GDPR/compliance (technical implementation)
- "returns": physical product returns, exchanges, damaged physical goods, return shipping labels, wrong size/item. Does NOT include refunds for service failures — those go to billing.
- "general": feature requests, partnership inquiries, support hours, certifications, media/press inquiries

RESPONSE RULES:
- Write a professional 2-4 sentence reply addressing the customer's concern
- Include specific keywords related to the issue (e.g., "refund" for billing, "escalate" for urgent issues)
- Acknowledge the problem, state what action you'll take
- For urgent issues, mention "escalate" or "senior" or "immediately"
- For billing: mention "refund", "process", "charge", "account" as relevant
- For technical: mention "investigate", "team", "issue", "error" as relevant
- For returns: mention "return", "label", "ship", "policy" as relevant

OUTPUT: Return ONLY valid JSON with NO extra text:
{"action_type":"classify_priority","email_id":"E001","priority":"urgent","department":null,"response_text":null,"reasoning":"one sentence"}"""


def _extract_json(raw: str) -> Optional[dict]:
    """Robustly extract JSON from LLM output that may contain markdown or extra text."""
    # Strip markdown code fences
    if "```" in raw:
        # Extract content between code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find JSON with nested braces
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_llm(observation: dict, last_reward: Optional[dict] = None) -> dict:
    """Call the LLM to decide the next action for the current email."""
    email = observation.get("current_email")
    if not email:
        return {"action_type": "archive", "email_id": "", "priority": None,
                "department": None, "response_text": None}

    email_id = email.get("email_id", "")
    history = observation.get("action_history", [])
    done_for = [h["action_type"] for h in history if h.get("email_id") == email_id]
    remaining = [a for a in ["classify_priority", "assign_department", "draft_response", "archive"]
                 if a not in done_for]

    # Build context-rich user message
    penalty = ""
    if last_reward and last_reward.get("penalty_reason"):
        penalty = f"\nLast penalty: {last_reward['penalty_reason']}"

    hint_line = ""
    if email.get("category_hint"):
        hint_line = f"Category Hint: {email['category_hint']}"

    user_msg = f"""EMAIL ID: {email_id}
Subject: {email.get('subject', '')}
From: {email.get('sender', '')} (Tier: {email.get('sender_tier', 'free')})
{hint_line}
Body: {email.get('body', '')}

Already completed actions on this email: {done_for if done_for else 'none'}
NEXT ACTIONS STILL NEEDED (pick the first one): {remaining}{penalty}

Respond with JSON for email {email_id}. Pick the FIRST action from the remaining list."""

    if llm is None:
        return {"action_type": "skip", "email_id": email_id, "priority": None,
                "department": None, "response_text": None}

    # Retry up to 2 times on LLM failure
    for attempt in range(2):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = (resp.choices[0].message.content or "").strip()
            action = _extract_json(raw)
            if action is None:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                # Last resort: return skip
                return {"action_type": "skip", "email_id": email_id, "priority": None,
                        "department": None, "response_text": None}

            # Ensure email_id is correct
            action["email_id"] = email_id
            time.sleep(0.3)
            return action

        except Exception:
            if attempt == 0:
                time.sleep(1)
                continue
            return {"action_type": "skip", "email_id": email_id, "priority": None,
                    "department": None, "response_text": None}

    return {"action_type": "skip", "email_id": email_id, "priority": None,
            "department": None, "response_text": None}


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(client: EnvClient, difficulty: str) -> dict:
    """Run a complete triage episode for one difficulty level."""
    task = f"omni-triage-{difficulty}"
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = client.reset(difficulty)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            email = obs.get("current_email")
            if not email:
                break

            email_id = email.get("email_id", "")
            history = obs.get("action_history", [])
            done_for = [h["action_type"] for h in history if h.get("email_id") == email_id]

            # Check if we should escalate BEFORE archiving
            # Escalate if: draft_response done, archive not done, and email needs escalation
            if ("draft_response" in done_for
                    and "escalate" not in done_for
                    and "archive" not in done_for
                    and _should_escalate(email)):
                action = {
                    "action_type": "escalate",
                    "email_id": email_id,
                    "priority": None,
                    "department": None,
                    "response_text": None,
                }
            else:
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
        log_step(
            step=max(steps_taken, 1), action="error",
            reward=0.0, done=True, error=str(e)[:200],
        )

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"difficulty": difficulty, "score": score, "passed": success}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
    )
    args = parser.parse_args()

    client = EnvClient(TRIAGE_ENV_URL)

    # Wait for the environment server to come up
    ready = False
    for i in range(12):
        if client.health():
            ready = True
            break
        time.sleep(5)

    difficulties = (
        ["easy", "medium", "hard"] if args.difficulty == "all"
        else [args.difficulty]
    )

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

    # Write results summary
    try:
        output_path = os.environ.get("RESULTS_OUTPUT_PATH", "results.json")
        with open(output_path, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "benchmark": BENCHMARK,
                "scores": {d: s for d, s in zip(difficulties, scores)},
                "average_score": sum(scores) / len(scores),
            }, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.001 rewards=0.00", flush=True)
        sys.exit(0)
