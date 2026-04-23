"""
Strictly typed Pydantic models for OmniTriageEnv.
All OpenEnv-required types: Observation, Action, Reward.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class Priority(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    RETURNS = "returns"


class ActionType(str, Enum):
    CLASSIFY_PRIORITY = "classify_priority"
    ASSIGN_DEPARTMENT = "assign_department"
    DRAFT_RESPONSE = "draft_response"
    ESCALATE = "escalate"
    ARCHIVE = "archive"
    SKIP = "skip"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Email data ────────────────────────────────────────────────────────────────

class Email(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    sender_tier: str          # "free" | "pro" | "enterprise"
    received_at: str          # ISO-8601 string — deterministic, no datetime obj
    category_hint: Optional[str] = None   # hidden from agent in hard tasks
    channel: str = "email"    # "email" | "grievance" | "social_media"


class EmailState(BaseModel):
    email_id: str
    priority_assigned: Optional[Priority] = None
    department_assigned: Optional[Department] = None
    response_drafted: Optional[str] = None
    escalated: bool = False
    archived: bool = False
    skip_count: int = 0


# ── OpenEnv Core Types ────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Agent-visible state at each step."""
    current_email: Optional[Email]
    queue_length: int
    processed_count: int
    session_score: float
    skip_budget: int                       # remaining skips before penalty
    action_history: List[Dict[str, Any]]   # last N actions for loop detection
    task_id: str
    task_difficulty: TaskDifficulty
    step_number: int
    done: bool


class Action(BaseModel):
    """Typed action the agent sends to step()."""
    action_type: ActionType
    email_id: str
    # payload fields — only one used per action type
    priority: Optional[Priority] = None
    department: Optional[Department] = None
    response_text: Optional[str] = None


class Reward(BaseModel):
    """Dense reward signal returned by step()."""
    value: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float]   # component scores for interpretability
    penalty_reason: Optional[str] = None
    cumulative: float


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]
