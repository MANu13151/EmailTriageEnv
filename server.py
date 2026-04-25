"""
OpenEnv-compliant HTTP server for OmniTriageEnv.
Exposes:
  POST /reset
  POST /step
  GET  /state
  GET  /grade
  GET  /health
  GET  /info
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import Action, Observation, StepResult
from environment import OmniTriageEnv

from typing import Optional, List
from fastapi import Body

app = FastAPI(
    title="OmniTriageEnv",
    description="OpenEnv-compliant omnichannel triage RL environment for customer support automation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (dashboard)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global environment state
_env: Dict[str, OmniTriageEnv] = {}


class ResetRequest(BaseModel):
    difficulty: str = "easy"


class TriageTestRequest(BaseModel):
    """Custom email submitted by a judge for triage testing."""
    subject: str = "Test email"
    body: str = "This is a test email body."
    sender: str = "judge@example.com"
    sender_tier: str = "pro"  # free | pro | enterprise
    channel: str = "email"    # email | grievance | social_media
    expected_department: Optional[str] = None  # judge can optionally set expected values
    expected_priority: Optional[str] = None


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint — shows environment info instead of 404."""
    return {
        "name": "OmniTriageEnv",
        "description": "OpenEnv-compliant omnichannel triage RL environment for customer support automation",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "grade": "GET /grade",
        },
        "tasks": ["easy", "medium", "hard"],
        "documentation": "See /docs for interactive API documentation",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the live demo dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if html_path.is_file():
        return html_path.read_text()
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "OmniTriageEnv"}


@app.get("/info")
def info() -> Dict[str, Any]:
    """Return environment metadata for discovery and validation."""
    return {
        "name": "omni-triage-env",
        "version": "1.0.0",
        "description": "OpenEnv-compliant omnichannel triage RL environment",
        "interface_version": "1.0",
        "tasks": [
            {"id": "easy", "name": "Basic Email Triage", "passing_score": 0.70},
            {"id": "medium", "name": "Ambiguous Email Triage", "passing_score": 0.60},
            {"id": "hard", "name": "Complex & Nuanced Email Triage", "passing_score": 0.50},
        ],
        "action_types": [
            "classify_priority", "assign_department", "draft_response",
            "escalate", "archive", "skip",
        ],
        "reward_range": [-1.0, 1.0],
        "deterministic": True,
    }


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = Body(default=None)) -> Observation:
    if request is None:
        request = ResetRequest(difficulty="easy")
    if request.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
    env = OmniTriageEnv(difficulty=request.difficulty)
    _env["default"] = env
    # Note: OmniTriageEnv.__init__ already calls reset() internally,
    # so we just return the current state. No double-reset needed.
    return env.state()


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    env = _env.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=Observation)
def state() -> Observation:
    env = _env.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


@app.get("/grade")
def grade() -> Dict[str, Any]:
    env = _env.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.grade_episode()


# ── Heuristic classification (same logic the dashboard demo agent uses) ───────

_PRIORITY_RULES = [
    (["urgent", "fraud", "critical", "security", "data loss", "breach", "chargeback",
      "gdpr", "legal", "compliance", "production down", "outage", "unauthorized",
      "harassment", "harass", "immediately", "emergency", "sued", "lawsuit",
      "data exposed", "pii"], "urgent"),
    (["error", "bug", "charge", "invoice", "defect", "webhook", "broken",
      "crash", "overcharg", "not working", "failing", "down", "blocked",
      "cannot access", "500 error", "503 error", "404 error"], "normal"),
    (["password", "login", "log in", "reset", "how do i", "feature request",
      "support hours", "general inquiry", "what are", "can i", "return",
      "question", "help me", "when do", "where can"], "low"),
]

_DEPT_KEYWORDS = {
    "billing": ["refund", "charge", "invoice", "billing", "payment", "subscription",
                 "overcharg", "chargeback", "renewal", "pricing", "discount", "pro-rata",
                 "vat", "tax", "fraud", "scam", "unauthorized charge", "double charge",
                 "overcharge", "price", "plan", "upgrade", "downgrade", "cancellation fee"],
    "technical": ["error", "api", "bug", "crash", "data", "security", "gdpr", "breach",
                   "webhook", "outage", "migrate", "migration", "rate limit", "throttl",
                   "export", "database", "browser", "404", "503", "500", "dark mode",
                   "login", "log in", "password", "credentials", "credential",
                   "authentication", "cannot access", "account locked", "locked out",
                   "sign in", "two factor", "2fa", "mfa", "invalid credentials",
                   "access denied", "not working", "broken", "integration",
                   "ssl", "certificate", "dns", "server", "deploy", "endpoint",
                   "timeout", "latency", "performance"],
    "returns": ["return", "exchange", "damaged", "defect", "replacement", "ship",
                "label", "wrong size", "wrong item", "broken item", "refund request",
                "warranty", "defective", "not as described", "sent wrong"],
    "general": ["feature request", "partnership", "support hours", "when do",
                "how do i", "question about", "feedback", "suggestion", "roadmap",
                "certification", "press", "media", "inquiry", "contact",
                "compliment", "thank"],
}

_ESCALATION_KEYWORDS = [
    "fraud", "unauthorized", "chargeback", "security breach", "data loss",
    "gdpr", "legal", "compliance", "media inquiry", "journalist", "press",
    "critical outage", "production down", "data exposed", "pii",
    "regulatory", "article 17", "data deletion", "harassment", "viral",
    "lawsuit", "attorney", "lawyer", "sued",
]


def _classify_email(subject: str, body: str, sender_tier: str) -> Dict[str, Any]:
    """Classify a custom email using weighted scoring for more accurate results."""
    text = (subject + " " + body).lower()

    # ── Priority: first-match on ordered tiers ──
    priority = "low"
    for keywords, prio in _PRIORITY_RULES:
        if any(kw in text for kw in keywords):
            priority = prio
            break

    # ── Department: weighted scoring (pick dept with most keyword hits) ──
    dept_scores: Dict[str, int] = {}
    dept_matched: Dict[str, list] = {}
    for dept, keywords in _DEPT_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text]
        dept_scores[dept] = len(matches)
        dept_matched[dept] = matches

    # Pick department with highest score, fallback to "general"
    best_dept = max(dept_scores, key=dept_scores.get) if dept_scores else "general"
    department = best_dept if dept_scores.get(best_dept, 0) > 0 else "general"

    # ── Escalation ──
    signal_count = sum(1 for sig in _ESCALATION_KEYWORDS if sig in text)
    should_escalate = signal_count >= 2 or (sender_tier == "enterprise" and signal_count >= 1)

    # ── Confidence based on total keyword match strength ──
    total_matches = sum(dept_scores.values())
    priority_matches = sum(1 for kws, _ in _PRIORITY_RULES for kw in kws if kw in text)
    confidence = min(1.0, (priority_matches + total_matches) / 6)

    # Flatten all matched dept keywords for display
    all_dept_kws = []
    for kws in dept_matched.values():
        all_dept_kws.extend(kws)

    return {
        "priority": priority,
        "department": department,
        "should_escalate": should_escalate,
        "escalation_signals_found": signal_count,
        "confidence": round(confidence, 2),
        "department_scores": {k: v for k, v in dept_scores.items() if v > 0},
        "matched_keywords": {
            "priority_keywords": [kw for kws, _ in _PRIORITY_RULES for kw in kws if kw in text],
            "department_keywords": list(set(all_dept_kws)),
            "escalation_keywords": [kw for kw in _ESCALATION_KEYWORDS if kw in text],
        },
    }


@app.post("/triage-test")
def triage_test(req: TriageTestRequest) -> Dict[str, Any]:
    """Judge endpoint: submit a custom email and see how the agent triages it."""
    result = _classify_email(req.subject, req.body, req.sender_tier)

    # If the judge provided expected values, compute correctness
    correctness = {}
    if req.expected_department:
        correctness["department_correct"] = result["department"] == req.expected_department.lower()
    if req.expected_priority:
        correctness["priority_correct"] = result["priority"] == req.expected_priority.lower()

    return {
        "input": {
            "subject": req.subject,
            "body": req.body[:200] + ("..." if len(req.body) > 200 else ""),
            "sender": req.sender,
            "sender_tier": req.sender_tier,
            "channel": req.channel,
        },
        "triage_result": result,
        "correctness": correctness if correctness else None,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
