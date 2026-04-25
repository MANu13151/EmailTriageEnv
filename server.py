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

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL ANALYSIS ENGINE — reads emails like a human, not a dictionary
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 1: Fraud Pattern Templates ─────────────────────────────────────────
# Each pattern has concept groups. If enough concept groups match, the pattern
# fires — even if the word "fraud" never appears in the email.

FRAUD_PATTERNS = {
    "account_takeover": {
        "name": "Account Takeover",
        "risk_weight": 35,
        "concepts": [
            ["accessed", "logged in", "login", "signed in", "log in", "activity"],
            ["different", "unknown", "suspicious", "unusual", "unfamiliar", "another"],
            ["device", "city", "location", "ip", "country", "place", "browser"],
        ],
        "min_groups": 2,
    },
    "phishing": {
        "name": "Phishing Attempt",
        "risk_weight": 40,
        "concepts": [
            ["click", "link", "url", "website", "email"],
            ["update", "verify", "confirm", "provide", "enter", "share"],
            ["bank", "card", "password", "otp", "details", "credentials", "banking"],
            ["suspicious", "off", "fake", "scam", "legitimate", "legit", "feels"],
        ],
        "min_groups": 2,
    },
    "social_engineering": {
        "name": "Social Engineering",
        "risk_weight": 40,
        "concepts": [
            ["call", "called", "phone", "someone", "person"],
            ["otp", "password", "pin", "code", "verification", "verify"],
            ["claiming", "pretending", "said they", "support team", "your team",
             "your company", "representative"],
        ],
        "min_groups": 2,
    },
    "unauthorized_charge": {
        "name": "Unauthorized Financial Activity",
        "risk_weight": 35,
        "concepts": [
            ["deducted", "charged", "charge", "amount", "money", "transaction",
             "payment", "debit", "withdrawn"],
            ["didn't", "did not", "not", "without", "never", "don't",
             "wasn't", "haven't", "no"],
            ["authorize", "permission", "knowledge", "consent", "approve",
             "remember", "recogni", "place", "subscribe"],
        ],
        "min_groups": 2,
    },
    "double_charge": {
        "name": "Duplicate / Repeated Charge",
        "risk_weight": 25,
        "concepts": [
            ["twice", "double", "multiple", "repeated", "again", "two times",
             "2 times", "duplicate", "every month"],
            ["charge", "deduct", "amount", "payment", "transaction", "billed",
             "debit"],
        ],
        "min_groups": 2,
    },
    "account_compromise": {
        "name": "Account Compromise",
        "risk_weight": 35,
        "concepts": [
            ["compromised", "hacked", "hack", "someone else", "not me",
             "breach", "stolen", "taken over", "hijack"],
            ["account", "login", "profile", "access", "password"],
        ],
        "min_groups": 2,
    },
    "unauthorized_subscription": {
        "name": "Unknown / Unauthorized Subscription",
        "risk_weight": 25,
        "concepts": [
            ["charge", "subscription", "subscribed", "billed", "deducted",
             "amount"],
            ["don't remember", "didn't subscribe", "unknown", "not authorized",
             "don't recogni", "never signed", "didn't sign"],
        ],
        "min_groups": 2,
    },
}

# ── Layer 2: Distress / Urgency Tone Analysis ───────────────────────────────

DISTRESS_WORDS = [
    "worried", "concerned", "scared", "afraid", "panic", "panicking",
    "desperate", "helpless", "distressed", "anxious", "terrified",
    "frightened", "alarming", "alarmed", "shocking", "shocked",
    "devastating", "horrible", "terrible", "nightmare",
]

URGENCY_PHRASES = [
    "immediately", "urgently", "asap", "right away", "right now",
    "please help", "need help", "as soon as possible", "without delay",
    "at the earliest", "time sensitive", "cannot wait",
]

PLEADING_PHRASES = [
    "please", "i beg", "i request", "kindly", "i need",
    "help me", "save my", "protect my", "secure my",
]

DENIAL_PHRASES = [
    "i did not", "i didn't", "i have not", "i haven't",
    "wasn't me", "not me", "never did", "never authorized",
    "i don't remember", "i don't recogni", "without my",
    "not authorized", "didn't authorize", "did not authorize",
    "without my knowledge", "without my permission", "without my consent",
    "without permission", "not my", "i deny",
]


def _detect_fraud_patterns(text: str) -> List[Dict[str, Any]]:
    """Detect fraud patterns by matching concept groups — not individual keywords."""
    detected = []
    for pattern_id, pattern in FRAUD_PATTERNS.items():
        matched_groups = []
        matched_terms = []
        for group in pattern["concepts"]:
            hits = [term for term in group if term in text]
            if hits:
                matched_groups.append(hits)
                matched_terms.extend(hits)

        if len(matched_groups) >= pattern["min_groups"]:
            detected.append({
                "pattern_id": pattern_id,
                "name": pattern["name"],
                "risk_weight": pattern["risk_weight"],
                "groups_matched": len(matched_groups),
                "groups_required": pattern["min_groups"],
                "total_groups": len(pattern["concepts"]),
                "terms_found": list(set(matched_terms)),
            })
    return detected


def _analyze_distress(text: str, raw_text: str) -> Dict[str, Any]:
    """Analyze emotional tone, urgency, and distress level."""
    distress_hits = [w for w in DISTRESS_WORDS if w in text]
    urgency_hits = [p for p in URGENCY_PHRASES if p in text]
    pleading_hits = [p for p in PLEADING_PHRASES if p in text]
    denial_hits = [p for p in DENIAL_PHRASES if p in text]

    # Caps analysis (on raw text, not lowered)
    words = raw_text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = len(caps_words) / max(len(words), 1)

    # Exclamation intensity
    exclamation_count = raw_text.count("!")

    # Compute distress score (0-100)
    distress_score = min(100, (
        len(distress_hits) * 15 +
        len(urgency_hits) * 12 +
        len(pleading_hits) * 5 +
        len(denial_hits) * 10 +
        min(caps_ratio * 80, 30) +
        min(exclamation_count * 5, 20)
    ))

    return {
        "distress_score": round(distress_score),
        "distress_words": distress_hits,
        "urgency_phrases": urgency_hits,
        "pleading_phrases": pleading_hits,
        "denial_phrases": denial_hits,
        "caps_words_found": len(caps_words),
        "exclamation_marks": exclamation_count,
    }


def _compute_risk_score(
    fraud_patterns: List[Dict],
    distress: Dict[str, Any],
) -> int:
    """Compute overall risk score (0-100) from all analysis layers."""
    # Pattern risk (strongest pattern dominates, others add 30%)
    pattern_risks = sorted(
        [p["risk_weight"] * p["groups_matched"] / p["total_groups"]
         for p in fraud_patterns],
        reverse=True,
    )
    pattern_score = 0
    if pattern_risks:
        pattern_score = pattern_risks[0]
        for additional in pattern_risks[1:]:
            pattern_score += additional * 0.3

    # Distress adds up to 25 extra risk points
    distress_boost = min(25, distress["distress_score"] * 0.25)

    # Denial language is a strong signal (people asserting "I didn't do this")
    denial_boost = min(15, len(distress["denial_phrases"]) * 5)

    risk = min(100, pattern_score + distress_boost + denial_boost)
    return round(risk)


# ── Layer 3: Department & Priority (enhanced with contextual analysis) ───────

_DEPT_KEYWORDS = {
    "billing": ["refund", "charge", "invoice", "billing", "payment", "subscription",
                 "overcharg", "chargeback", "renewal", "pricing", "discount", "pro-rata",
                 "vat", "tax", "fraud", "scam", "unauthorized charge", "double charge",
                 "overcharge", "price", "plan", "upgrade", "downgrade", "cancellation fee",
                 "deducted", "debit", "amount", "money"],
    "technical": ["error", "api", "bug", "crash", "data", "security", "gdpr", "breach",
                   "webhook", "outage", "migrate", "migration", "rate limit", "throttl",
                   "export", "database", "browser", "404", "503", "500", "dark mode",
                   "login", "log in", "password", "credentials", "credential",
                   "authentication", "cannot access", "account locked", "locked out",
                   "sign in", "two factor", "2fa", "mfa", "invalid credentials",
                   "access denied", "not working", "broken", "integration",
                   "ssl", "certificate", "dns", "server", "deploy", "endpoint",
                   "timeout", "latency", "performance",
                   "compromised", "hacked", "suspicious login", "suspicious activity",
                   "secure my account", "block my account", "protect my account"],
    "returns": ["return", "exchange", "damaged", "defect", "replacement", "ship",
                "label", "wrong size", "wrong item", "broken item", "refund request",
                "warranty", "defective", "not as described", "sent wrong"],
    "general": ["feature request", "partnership", "support hours", "when do",
                "how do i", "question about", "feedback", "suggestion", "roadmap",
                "certification", "press", "media", "inquiry", "contact",
                "compliment", "thank"],
}

_PRIORITY_KEYWORDS = {
    "urgent": ["urgent", "fraud", "critical", "security", "data loss", "breach",
               "chargeback", "gdpr", "legal", "compliance", "production down",
               "outage", "unauthorized", "harassment", "immediately", "emergency",
               "sued", "lawsuit", "data exposed", "pii"],
    "normal": ["error", "bug", "charge", "invoice", "defect", "webhook", "broken",
               "crash", "overcharg", "not working", "failing", "down", "blocked",
               "cannot access", "500 error", "503 error", "404 error"],
    "low":    ["password", "login", "log in", "reset", "how do i", "feature request",
               "support hours", "general inquiry", "what are", "can i", "return",
               "question", "help me", "when do", "where can"],
}


def _classify_email(subject: str, body: str, sender_tier: str) -> Dict[str, Any]:
    """
    Multi-layered contextual email analysis:
      Layer 1 — Fraud pattern detection (concept-level matching)
      Layer 2 — Distress / urgency tone analysis
      Layer 3 — Risk score computation
      Layer 4 — Priority & department (enhanced with risk context)
    """
    raw_text = subject + " " + body
    text = raw_text.lower()

    # ── Layer 1: Fraud Pattern Detection ──
    fraud_patterns = _detect_fraud_patterns(text)

    # ── Layer 2: Distress / Tone Analysis ──
    distress = _analyze_distress(text, raw_text)

    # ── Layer 3: Risk Score ──
    risk_score = _compute_risk_score(fraud_patterns, distress)

    # ── Layer 4a: Priority ──
    priority = "low"
    priority_matched = []
    for prio in ["urgent", "normal", "low"]:
        hits = [kw for kw in _PRIORITY_KEYWORDS[prio] if kw in text]
        if hits:
            priority = prio
            priority_matched = hits
            break

    # Risk override: high-risk emails are always urgent
    if risk_score >= 40 and priority != "urgent":
        priority = "urgent"

    # ── Layer 4b: Department (weighted scoring) ──
    dept_scores: Dict[str, int] = {}
    dept_matched: Dict[str, list] = {}
    for dept, keywords in _DEPT_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text]
        dept_scores[dept] = len(matches)
        dept_matched[dept] = matches

    best_dept = max(dept_scores, key=dept_scores.get) if dept_scores else "general"
    department = best_dept if dept_scores.get(best_dept, 0) > 0 else "general"

    # ── Escalation Decision (risk-aware) ──
    should_escalate = (
        risk_score >= 30                                   # medium+ risk → escalate
        or len(fraud_patterns) >= 1                        # any fraud pattern → escalate
        or (priority == "urgent" and distress["distress_score"] >= 20)
        or (sender_tier == "enterprise" and risk_score >= 15)
    )

    # Determine escalation reason
    if should_escalate:
        if fraud_patterns:
            escalation_reason = f"Fraud pattern detected: {fraud_patterns[0]['name']}"
        elif risk_score >= 30:
            escalation_reason = f"High risk score: {risk_score}/100"
        else:
            escalation_reason = "Urgent priority with distress signals"
    else:
        escalation_reason = None

    # Confidence from total signal strength
    all_dept_kws = []
    for kws in dept_matched.values():
        all_dept_kws.extend(kws)
    total_matches = sum(dept_scores.values()) + len(priority_matched)
    confidence = min(1.0, (total_matches + len(fraud_patterns) * 3) / 6)

    return {
        "priority": priority,
        "department": department,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "department_scores": {k: v for k, v in dept_scores.items() if v > 0},
        "fraud_patterns_detected": fraud_patterns,
        "distress_analysis": distress,
        "matched_keywords": {
            "priority_keywords": priority_matched,
            "department_keywords": list(set(all_dept_kws)),
            "escalation_keywords": [p["name"] for p in fraud_patterns],
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

