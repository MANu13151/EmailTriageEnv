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
from fastapi.responses import HTMLResponse, RedirectResponse
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
def root():
    """Root endpoint — redirect to the live dashboard."""
    return RedirectResponse(url="/dashboard")


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
            ["click", "link", "url", "website"],
            ["update", "verify", "confirm", "enter", "share"],
            ["bank", "card", "password", "otp", "credentials", "banking", "ssn", "pin"],
            ["suspicious", "fake", "scam", "legitimate", "legit", "feels off", "looks off"],
        ],
        "min_groups": 3,
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
            ["didn't", "did not", "without my", "never", "don't remember",
             "wasn't", "haven't", "i did not", "not authorized"],
            ["authorize", "permission", "knowledge", "consent", "approve",
             "place this", "subscribe"],
        ],
        "min_groups": 3,
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

# ── Legal / Regulatory Escalation Signals ────────────────────────────────────
# These indicate the email must be routed to a human — NOT auto-replied alone.

_LEGAL_ESCALATION_SIGNALS = [
    # Legal threats
    "attorney", "lawyer", "lawsuit", "litigation", "legal action",
    "court", "sue", "sued", "subpoena", "injunction",
    # Regulatory bodies
    "ftc", "fcc", "cfpb", "consumer protection", "attorney general",
    "better business bureau", "bbb", "fdcpa", "fcra",
    "state regulator", "filed complaint", "regulatory",
    # Compliance demands
    "gdpr", "ccpa", "hipaa", "ada", "title iii",
    "article 17", "right to erasure", "data deletion",
    "formal grievance", "formal complaint",
    # Attorney involvement
    "my attorney", "my lawyer", "legal counsel", "cc'd",
    "legal team", "legal department", "legal representative",
]

_HIGH_RISK_ESCALATION_SIGNALS = [
    # Financial risk
    "chargeback", "bank dispute", "credit card dispute",
    "dispute with my bank", "filed with",
    # Reputation risk
    "journalist", "reporter", "press", "media inquiry",
    "going viral", "social media", "public response",
    "declined to comment",
    # Safety / harassment
    "harassment", "threatening", "harassing", "recorded calls",
    "violat",  # catches "violates", "violation"
]

# ── Service-Failure Refund Context (routes to billing, NOT returns) ──────────

_SERVICE_FAILURE_CONTEXT = [
    "service", "outage", "downtime", "broken", "not working",
    "defective service", "failed", "failure", "cancelled",
    "poor service", "unusable", "unreliable", "repeated issues",
    "quality", "compensation", "software", "platform", "app",
    "system", "server", "api", "feature", "bug",
]

_PRODUCT_RETURN_CONTEXT = [
    "return", "ship", "label", "exchange", "damaged goods",
    "wrong size", "wrong item", "packaging", "shipped",
    "delivery", "parcel", "box", "physical", "product",
]

# ── Ambiguity / Confusion Signals (customer needs human guidance) ────────────
# When a customer is unsure what action to take, automation can't decide for them.

_AMBIGUITY_SIGNALS = [
    "not sure", "i'm not sure", "i am not sure",
    "don't know", "do not know", "unsure", "confused",
    "should i", "what should", "which option", "what's best",
    "what is best", "which is better", "what do you suggest",
    "what do you recommend", "please suggest", "please advise",
    "advise me", "guide me", "help me decide", "help me choose",
    "not certain", "can't decide", "either way", "or should i",
    "what are my options", "what option", "prefer your advice",
    "refund or", "return or", "replace or", "exchange or",
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
                 "deducted", "debit", "amount", "money",
                 # Service-failure refund keywords (refunds for broken service, NOT product returns)
                 "service failure refund", "refund for service", "refund due to",
                 "compensation", "credit my account", "reimburse", "reimbursement",
                 "outage refund", "downtime refund", "broken service",
                 "not working refund", "failed service"],
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

    # ── Layer 4a: Priority (context-aware) ──
    # Start with keyword signals as a baseline
    priority = "low"
    priority_matched = []
    for prio in ["urgent", "normal", "low"]:
        hits = [kw for kw in _PRIORITY_KEYWORDS[prio] if kw in text]
        if hits:
            priority = prio
            priority_matched = hits
            break

    # Context-aware priority upgrades based on risk, distress, and patterns
    # Any fraud pattern detected → urgent
    if fraud_patterns and priority != "urgent":
        priority = "urgent"
        priority_matched.append(f"fraud_pattern:{fraud_patterns[0]['name']}")

    # High risk score → urgent
    if risk_score >= 40 and priority != "urgent":
        priority = "urgent"
        priority_matched.append(f"risk_score:{risk_score}")

    # Medium risk (20-39) → at least normal
    if risk_score >= 20 and priority == "low":
        priority = "normal"
        priority_matched.append(f"risk_score:{risk_score}")

    # High distress → bump priority
    if distress["distress_score"] >= 40 and priority == "low":
        priority = "normal"
        priority_matched.append("high_distress")
    if distress["distress_score"] >= 60 and priority != "urgent":
        priority = "urgent"
        priority_matched.append("extreme_distress")

    # Enterprise clients get priority bump
    if sender_tier == "enterprise" and priority == "low":
        priority = "normal"
        priority_matched.append("enterprise_client")

    # ── Layer 4b: Department (weighted scoring) ──
    dept_scores: Dict[str, int] = {}
    dept_matched: Dict[str, list] = {}
    for dept, keywords in _DEPT_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text]
        dept_scores[dept] = len(matches)
        dept_matched[dept] = matches

    # ── Service-failure refund disambiguation ──
    # When "refund" appears, check if it's for a service failure or a product return.
    # Service-failure refunds → billing, product returns → returns.
    if "refund" in text:
        service_ctx = sum(1 for s in _SERVICE_FAILURE_CONTEXT if s in text)
        product_ctx = sum(1 for s in _PRODUCT_RETURN_CONTEXT if s in text)
        if service_ctx > product_ctx:
            # Strong boost to billing — this is a service-failure refund
            dept_scores["billing"] = dept_scores.get("billing", 0) + 3
        elif product_ctx > 0 and service_ctx == 0:
            # Boost returns — this is clearly a product return refund
            dept_scores["returns"] = dept_scores.get("returns", 0) + 2

    best_dept = max(dept_scores, key=dept_scores.get) if dept_scores else "general"
    department = best_dept if dept_scores.get(best_dept, 0) > 0 else "general"

    # ── Escalation Decision (risk-aware + legal/regulatory) ──
    should_escalate = (
        risk_score >= 30                                   # medium+ risk → escalate
        or len(fraud_patterns) >= 1                        # any fraud pattern → escalate
        or (priority == "urgent" and distress["distress_score"] >= 20)
        or (sender_tier == "enterprise" and risk_score >= 15)
    )

    # ── Legal / Regulatory / High-Risk escalation (human review required) ──
    legal_hits = [s for s in _LEGAL_ESCALATION_SIGNALS if s in text]
    risk_hits = [s for s in _HIGH_RISK_ESCALATION_SIGNALS if s in text]
    needs_human_review = False
    human_review_reason = None

    if legal_hits or len(risk_hits) >= 2:
        should_escalate = True
        needs_human_review = True
        human_review_reason = "legal_risk"
    elif len(risk_hits) == 1 and priority == "urgent":
        should_escalate = True
        needs_human_review = True
        human_review_reason = "legal_risk"

    # ── Ambiguity detection (customer is confused / needs human guidance) ──
    ambiguity_hits = [s for s in _AMBIGUITY_SIGNALS if s in text]
    if ambiguity_hits and not needs_human_review:
        needs_human_review = True
        should_escalate = True
        human_review_reason = "ambiguous"

    # ── Insufficient context detection (vague / minimal emails) ──
    # If the email body is too short and has no clear department signals,
    # a human needs to follow up because automation can't do anything useful.
    body_words = len(raw_text.split())
    max_dept_score = max(dept_scores.values()) if dept_scores else 0
    if body_words <= 15 and max_dept_score <= 1 and not needs_human_review:
        needs_human_review = True
        should_escalate = True
        human_review_reason = "insufficient_context"

    # Determine escalation reason
    if should_escalate:
        if human_review_reason == "ambiguous":
            escalation_reason = f"Customer is confused/undecided: {', '.join(ambiguity_hits[:3])}"
        elif human_review_reason == "insufficient_context":
            escalation_reason = f"Insufficient context ({body_words} words, no clear signals)"
        elif legal_hits or risk_hits:
            all_signals = legal_hits + risk_hits
            escalation_reason = f"Legal/risk signals: {', '.join(all_signals[:5])}"
        elif fraud_patterns:
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
        "needs_human_review": needs_human_review,
        "escalation_reason": escalation_reason,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "department_scores": {k: v for k, v in dept_scores.items() if v > 0},
        "fraud_patterns_detected": fraud_patterns,
        "distress_analysis": distress,
        "human_review_reason": human_review_reason,
        "ambiguity_signals": ambiguity_hits,
        "legal_signals": legal_hits,
        "risk_signals": risk_hits,
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




# ---------------------------------------------------------------------------
# Live Email Integration (Gmail IMAP/SMTP)
# ---------------------------------------------------------------------------
# Set these env vars to enable:
#   GMAIL_ADDRESS=your.email@gmail.com
#   GMAIL_APP_PASSWORD=your-16-char-app-password
# ---------------------------------------------------------------------------

import imaplib
import smtplib
import email as email_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time as _time
from collections import deque

# Store for live emails processed
_live_emails: deque = deque(maxlen=50)
_email_thread_started = False

DEPT_EMAILS = {
    "billing": "Billing Department <billing@omnitriage.com>",
    "technical": "Technical Support <techsupport@omnitriage.com>",
    "returns": "Returns & Exchanges <returns@omnitriage.com>",
    "general": "General Support <support@omnitriage.com>",
}

DEPT_NAMES = {
    "billing": "Billing Department",
    "technical": "Technical Support",
    "returns": "Returns & Exchanges",
    "general": "General Support",
}


def _build_reply(triage_result: Dict, original_subject: str, sender_name: str) -> str:
    """Build a natural auto-reply. Only mention urgency if actually urgent."""
    dept = triage_result["department"]
    dept_name = DEPT_NAMES.get(dept, "Support")
    priority = triage_result["priority"]
    is_escalated = triage_result["should_escalate"]
    needs_human = triage_result.get("needs_human_review", False)
    ref_id = f"OT-{int(_time.time()) % 100000}"

    if needs_human:
        # Legal/regulatory/high-risk — a HUMAN will follow up
        return (
            f"Dear {sender_name},\n\n"
            f"Thank you for reaching out. We have received your email and understand "
            f"the seriousness of this matter.\n\n"
            f"Your case has been flagged for immediate human review by our {dept_name} team. "
            f"A dedicated team member — not an automated system — will personally review "
            f"your case and respond as soon as possible.\n\n"
            f"Your reference number is #{ref_id}. Please keep this for your records.\n\n"
            f"We take this matter very seriously and appreciate your patience.\n\n"
            f"Best regards,\n{dept_name}\nOmniTriage Support"
        )
    elif is_escalated:
        return (
            f"Dear {sender_name},\n\n"
            f"Thank you for reaching out. We understand this is an urgent matter "
            f"and have immediately escalated your case to our {dept_name} team.\n\n"
            f"A dedicated agent will review your case and get back to you shortly. "
            f"Your reference number is #{ref_id}.\n\n"
            f"We take this matter seriously and will ensure prompt resolution.\n\n"
            f"Best regards,\n{dept_name}\nOmniTriage Support"
        )
    else:
        return (
            f"Dear {sender_name},\n\n"
            f"Thank you for reaching out to us. We have received your email "
            f"and it has been forwarded to our {dept_name}.\n\n"
            f"A team member will get back to you shortly. "
            f"Your reference number is #{ref_id}.\n\n"
            f"Best regards,\n{dept_name}\nOmniTriage Support"
        )


def _send_reply(gmail_addr: str, gmail_pass: str, to_addr: str,
                subject: str, body: str, dept: str):
    """Send auto-reply via Gmail SMTP."""
    try:
        msg = MIMEMultipart()
        msg["From"] = f"OmniTriage {DEPT_NAMES.get(dept, 'Support')} <{gmail_addr}>"
        msg["To"] = to_addr
        msg["Subject"] = f"Re: {subject}"
        msg["X-OmniTriage-Department"] = dept
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_addr, gmail_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[EMAIL] Failed to send reply: {e}")
        return False


QUARANTINE_LABEL = "Fraud-Quarantine"
HUMAN_REVIEW_LABEL = "Human-Review"
QUARANTINE_DAYS = 7  # Auto-delete quarantined emails after this many days


def _ensure_labels(mail):
    """Create the Fraud-Quarantine and Human-Review labels in Gmail if they don't exist."""
    for label_name in [QUARANTINE_LABEL, HUMAN_REVIEW_LABEL]:
        try:
            status, folders = mail.list('""', label_name)
            if status != "OK" or not folders or folders[0] is None:
                mail.create(label_name)
                print(f"[EMAIL] 📁 Created '{label_name}' label in Gmail")
        except Exception:
            # Label might already exist — that's fine
            try:
                mail.create(label_name)
            except Exception:
                pass


def _cleanup_old_quarantine(mail):
    """Delete quarantined emails older than QUARANTINE_DAYS."""
    try:
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=QUARANTINE_DAYS)).strftime("%d-%b-%Y")
        status = mail.select(QUARANTINE_LABEL)
        if status[0] != "OK":
            return 0

        # Find emails older than cutoff
        status, messages = mail.uid("search", None, f'(BEFORE "{cutoff}")')
        if status != "OK" or not messages[0].strip():
            mail.select("INBOX")  # Switch back
            return 0

        old_uids = messages[0].split()
        if not old_uids:
            mail.select("INBOX")
            return 0

        for uid in old_uids:
            mail.uid("store", uid, "+FLAGS", "(\\Deleted)")
        mail.expunge()
        mail.select("INBOX")  # Switch back to INBOX
        print(f"[EMAIL] 🧹 Auto-cleaned {len(old_uids)} quarantined email(s) older than {QUARANTINE_DAYS} days")
        return len(old_uids)
    except Exception as e:
        try:
            mail.select("INBOX")  # Always switch back
        except Exception:
            pass
        return 0


def _poll_inbox(gmail_addr: str, gmail_pass: str):
    """Background thread: poll Gmail IMAP for new emails, triage, and reply."""
    seen_uids = set()
    first_run = True
    cleanup_counter = 0  # Run cleanup every ~5 minutes (30 poll cycles)
    print(f"[EMAIL] 📬 Live email integration started for {gmail_addr}")

    while True:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(gmail_addr, gmail_pass)

            # Ensure quarantine label exists (only matters on first connect)
            if first_run:
                _ensure_labels(mail)

            mail.select("INBOX")

            # Use UID SEARCH for stable IDs that don't shift after deletes
            from datetime import datetime
            today = datetime.now().strftime("%d-%b-%Y")
            status, messages = mail.uid("search", None, f'(SINCE "{today}" UNSEEN)')
            if status != "OK":
                mail.logout()
                _time.sleep(10)
                continue

            uid_list = messages[0].split()

            # On first run, record existing UIDs to skip
            if first_run:
                for uid in uid_list:
                    seen_uids.add(uid)
                first_run = False
                print(f"[EMAIL] Skipped {len(uid_list)} existing emails from today. Watching for new ones...")
                mail.logout()
                _time.sleep(5)
                continue

            # Collect fraud UIDs to batch-delete after processing all emails
            fraud_uids = []

            for uid in uid_list:
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)

                # Fetch using UID
                status, msg_data = mail.uid("fetch", uid, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw_email = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw_email)

                # Extract fields
                subject = msg["Subject"] or "(No Subject)"
                from_addr = msg["From"] or ""
                # Parse sender name and email
                if "<" in from_addr:
                    sender_name = from_addr.split("<")[0].strip().strip('"')
                    sender_email = from_addr.split("<")[1].split(">")[0]
                else:
                    sender_name = from_addr.split("@")[0]
                    sender_email = from_addr

                # Skip our own auto-replies (but allow self-sent emails for demo)
                if gmail_addr.lower() in from_addr.lower() and subject.startswith("Re:"):
                    continue

                # Get body
                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body_text = part.get_payload(decode=True).decode(
                                part.get_content_charset() or "utf-8", errors="replace"
                            )
                            break
                else:
                    body_text = msg.get_payload(decode=True).decode(
                        msg.get_content_charset() or "utf-8", errors="replace"
                    )

                # Run through triage engine
                triage_result = _classify_email(subject, body_text, "free")
                print(f"[EMAIL] 📧 New email from {sender_email}: '{subject}' → "
                      f"{triage_result['priority']}/{triage_result['department']} "
                      f"(risk: {triage_result['risk_score']})")

                # Decide action based on fraud detection
                fraud_patterns = triage_result.get("fraud_patterns_detected", [])
                risk_score = triage_result.get("risk_score", 0)

                # ── Fraud / scam blocking (multi-layer) ──
                # Layer 1: Named fraud patterns detected by concept-group matching
                is_fraud = len(fraud_patterns) > 0
                # Layer 2: Very high risk score even without named pattern
                is_high_risk = risk_score >= 50

                # Layer 3: Extensive scam/fraud keyword and phrase detection
                combined_text = (subject + " " + body_text).lower()

                _fraud_keywords = [
                    # Classic fraud terms
                    "fraud", "scam", "phishing", "unauthorized transaction",
                    "stolen card", "identity theft", "money laundering",
                    # Prize / lottery scams
                    "lottery winner", "you have won", "you've won", "congratulations you won",
                    "claim your prize", "prize winner", "million dollar",
                    "won a lottery", "lucky winner", "selected as winner",
                    # Free money / giveaway scams
                    "free money", "get free", "win free", "earn free",
                    "million free", "cash prize", "cash reward", "free cash",
                    "free gift card", "gift card free",
                    # Urgency / pressure scams
                    "act now", "act immediately", "limited time offer",
                    "offer expires", "don't miss out", "last chance",
                    "urgent action required", "immediate action",
                    # Phishing / credential theft
                    "verify your account immediately", "account suspended",
                    "account has been locked", "confirm your identity",
                    "update your payment", "update your billing",
                    "your account will be closed", "verify your information",
                    "click here to verify", "click here to confirm",
                    "click below to", "click on this link",
                    # Financial scams
                    "wire transfer", "advance fee", "send money",
                    "nigerian prince", "foreign prince", "inheritance",
                    "unclaimed funds", "bank transfer", "western union",
                    "bitcoin investment", "crypto investment", "guaranteed return",
                    "double your money", "investment opportunity",
                    "make money fast", "work from home earn",
                    # Impersonation
                    "irs notification", "tax refund", "government grant",
                    "fbi warning", "court notice", "legal action against you",
                ]
                has_fraud_keywords = any(kw in combined_text for kw in _fraud_keywords)

                # Layer 4: Suspicious URL patterns (sketchy domains / too-good-to-be-true)
                import re as _re
                _suspicious_url_patterns = [
                    r"https?://.*(?:free|win|prize|cash|money|lucky|reward|gift).*\.",  # URLs with scam words
                    r"https?://.*(?:giveme|getfree|claimyour|earnfree).*",  # Obvious scam domains
                    r"https?://(?:\d{1,3}\.){3}\d{1,3}",  # Raw IP address URLs
                    r"https?://.*\.(?:xyz|top|buzz|click|loan|work|gq|cf|tk|ml)\b",  # Sketchy TLDs
                    r"bit\.ly|tinyurl|shorturl|t\.co.*(?:free|win|money)",  # Shortened URLs with scam context
                ]
                has_suspicious_url = any(_re.search(p, combined_text) for p in _suspicious_url_patterns)

                # Layer 5: Scam phrase combos (money amount + free/win/click)
                has_money_scam = bool(
                    _re.search(r"(?:\$|₹|€|£)\s*[\d,]+.*(?:free|win|claim|click|earn)", combined_text)
                    or _re.search(r"(?:free|win|claim|click|earn).*(?:\$|₹|€|£)\s*[\d,]+", combined_text)
                    or _re.search(r"\d+\s*(?:million|thousand|lakh|crore).*free", combined_text)
                    or _re.search(r"free.*\d+\s*(?:million|thousand|lakh|crore)", combined_text)
                )

                should_block = is_fraud or is_high_risk or has_fraud_keywords or has_suspicious_url or has_money_scam

                # Track whether this email needs human review
                needs_human = triage_result.get("needs_human_review", False)

                if should_block:
                    # Determine block reason for logging
                    if is_fraud:
                        block_reason = f"fraud_pattern:{fraud_patterns[0]['name']}"
                    elif is_high_risk:
                        block_reason = f"high_risk_score:{risk_score}"
                    elif has_suspicious_url:
                        block_reason = "suspicious_url_detected"
                    elif has_money_scam:
                        block_reason = "money_scam_pattern"
                    else:
                        block_reason = "fraud_keywords_detected"

                    print(f"[EMAIL] 🛡️ BLOCKED fraud email from {sender_email}: "
                          f"{block_reason} (risk: {risk_score})")

                    # Queue for batch deletion (don't expunge mid-loop)
                    fraud_uids.append(uid)
                    sent = False
                    action = "blocked"
                elif needs_human:
                    # LEGAL / HIGH-RISK: Copy to Human-Review label for human attention
                    try:
                        mail.uid("copy", uid, HUMAN_REVIEW_LABEL)
                        print(f"[EMAIL] 👤 HUMAN REVIEW needed for email from {sender_email}: "
                              f"{triage_result.get('escalation_reason', 'risk signals')}")
                    except Exception as hr_err:
                        print(f"[EMAIL] ⚠️ Human-Review label error: {hr_err}")

                    # Still send acknowledgment (but with human-review wording)
                    reply_body = _build_reply(triage_result, sender_name, sender_email)
                    sent = _send_reply(
                        gmail_addr, gmail_pass, sender_email,
                        subject, reply_body, triage_result["department"]
                    )
                    action = "human_review"
                else:
                    # SAFE: Send auto-reply
                    reply_body = _build_reply(triage_result, sender_name, sender_email)
                    sent = _send_reply(
                        gmail_addr, gmail_pass, sender_email,
                        subject, reply_body, triage_result["department"]
                    )
                    action = "escalated" if triage_result["should_escalate"] else "replied"

                # Store for dashboard
                _live_emails.appendleft({
                    "timestamp": _time.strftime("%H:%M:%S"),
                    "from": sender_email,
                    "from_name": sender_name,
                    "subject": subject,
                    "body_preview": body_text[:300],
                    "triage": {
                        "priority": triage_result["priority"],
                        "department": triage_result["department"],
                        "should_escalate": triage_result["should_escalate"],
                        "needs_human_review": triage_result.get("needs_human_review", False),
                        "risk_score": triage_result["risk_score"],
                        "escalation_reason": triage_result.get("escalation_reason"),
                        "fraud_detected": should_block,
                        "fraud_name": fraud_patterns[0]["name"] if fraud_patterns else None,
                        "block_reason": block_reason if should_block else None,
                    },
                    "action": action,
                    "reply_sent": sent,
                    "reply_dept": triage_result["department"],
                })

            # ── Move fraud emails to Quarantine label (not Trash) ──
            if fraud_uids:
                try:
                    for fuid in fraud_uids:
                        # Copy to Fraud-Quarantine label, then remove from inbox
                        mail.uid("copy", fuid, QUARANTINE_LABEL)
                        mail.uid("store", fuid, "+FLAGS", "(\\Deleted)")
                    mail.expunge()
                    print(f"[EMAIL] 📁 Quarantined {len(fraud_uids)} fraud email(s) → '{QUARANTINE_LABEL}' "
                          f"(auto-deletes after {QUARANTINE_DAYS} days)")
                except Exception as del_err:
                    print(f"[EMAIL] ⚠️ Quarantine error: {del_err}")

            # ── Periodic cleanup of old quarantined emails ──
            cleanup_counter += 1
            if cleanup_counter >= 30:  # Every ~5 minutes
                cleanup_counter = 0
                _cleanup_old_quarantine(mail)

            mail.logout()
        except Exception as e:
            print(f"[EMAIL] Error polling inbox: {e}")

        _time.sleep(10)  # Poll every 10 seconds


@app.get("/live-emails")
def get_live_emails():
    """Return list of live emails received and triaged."""
    return {"emails": list(_live_emails), "count": len(_live_emails)}


@app.get("/email-config")
def email_config():
    """Check if live email integration is active."""
    addr = os.environ.get("GMAIL_ADDRESS", "")
    active = bool(addr and os.environ.get("GMAIL_APP_PASSWORD", ""))
    return {
        "active": active,
        "email": addr[:3] + "***" + addr[addr.index("@"):] if active and "@" in addr else None,
    }


# Start email polling thread on server startup
@app.on_event("startup")
def _start_email_thread():
    global _email_thread_started
    gmail_addr = os.environ.get("GMAIL_ADDRESS", "")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
    if gmail_addr and gmail_pass and not _email_thread_started:
        _email_thread_started = True
        t = threading.Thread(target=_poll_inbox, args=(gmail_addr, gmail_pass), daemon=True)
        t.start()
        print(f"[EMAIL] ✅ Live email thread started for {gmail_addr}")
    else:
        print("[EMAIL] ⚠️ Live email disabled (set GMAIL_ADDRESS & GMAIL_APP_PASSWORD env vars)")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
