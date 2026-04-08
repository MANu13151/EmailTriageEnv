"""
Deterministic graders for all three task difficulties.
All graders are pure functions: same input → same score.
No randomness. All scores in [0.0, 1.0].
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple

from emails import GROUND_TRUTH


# ── Utility ───────────────────────────────────────────────────────────────────

def _keywords_found(text: str, keywords: List[str]) -> float:
    """Fraction of required keywords found in response text (case-insensitive)."""
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matched / len(keywords)


def _score_single_email(
    email_id: str,
    email_state: Dict[str, Any],
    require_hints: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Score a single processed email against ground truth.

    Returns:
        (total_score_0_to_1, component_breakdown)

    Components (equal weight):
        - priority_score   (0 or 1)
        - department_score (0 or 1)
        - response_score   (0 to 1 — keyword coverage)
        - escalation_score (0 or 1)
    """
    gt = GROUND_TRUTH.get(email_id)
    if gt is None:
        return 0.0, {}

    components: Dict[str, float] = {}

    # 1. Priority classification
    priority_assigned = email_state.get("priority_assigned")
    components["priority_score"] = (
        1.0 if priority_assigned == gt["priority"] else 0.0
    )

    # 2. Department assignment
    dept_assigned = email_state.get("department_assigned")
    components["department_score"] = (
        1.0 if dept_assigned == gt["department"] else 0.0
    )

    # 3. Response quality (keyword coverage)
    response_text = email_state.get("response_drafted") or ""
    keywords = gt.get("response_keywords", [])
    components["response_score"] = _keywords_found(response_text, keywords)

    # 4. Escalation decision
    escalated = email_state.get("escalated", False)
    expected_escalation = gt.get("escalate", False)
    if expected_escalation:
        components["escalation_score"] = 1.0 if escalated else 0.0
    else:
        # Penalize unnecessary escalation
        components["escalation_score"] = 0.0 if escalated else 1.0

    total = sum(components.values()) / max(len(components), 1)
    return total, components


# ── Task-level graders ────────────────────────────────────────────────────────

class EasyGrader:
    """
    EASY task grader.
    - Category hints ARE visible to agent
    - Full credit for each component
    - No partial penalty for skips
    Passing threshold: >= 0.70
    """

    TASK_ID = "easy"

    @staticmethod
    def grade(
        email_states: Dict[str, Dict[str, Any]],
        skip_count: int,
        invalid_action_count: int,
    ) -> Dict[str, Any]:
        if not email_states:
            return {"score": 0.0, "breakdown": {}, "passed": False}

        per_email_scores: Dict[str, float] = {}
        per_email_components: Dict[str, Dict[str, float]] = {}

        for email_id, state in email_states.items():
            score, components = _score_single_email(
                email_id, state, require_hints=True
            )
            per_email_scores[email_id] = score
            per_email_components[email_id] = components

        base_score = (
            sum(per_email_scores.values()) / len(per_email_scores)
            if per_email_scores else 0.0
        )

        # Penalty: 2% per invalid action, max 20%
        invalid_penalty = min(invalid_action_count * 0.02, 0.20)
        # Penalty: 3% per skip over 2 free skips
        excess_skips = max(0, skip_count - 2)
        skip_penalty = min(excess_skips * 0.03, 0.15)

        final_score = max(0.0, base_score - invalid_penalty - skip_penalty)
        final_score = round(min(final_score, 1.0), 4)

        return {
            "score": final_score,
            "base_score": round(base_score, 4),
            "invalid_penalty": round(invalid_penalty, 4),
            "skip_penalty": round(skip_penalty, 4),
            "per_email_scores": per_email_scores,
            "per_email_components": per_email_components,
            "passed": final_score >= 0.70,
        }


class MediumGrader:
    """
    MEDIUM task grader.
    - No category hints
    - Stricter escalation weighting (1.5x)
    - Skip budget = 1 (vs 2 for easy)
    Passing threshold: >= 0.60
    """

    TASK_ID = "medium"

    @staticmethod
    def grade(
        email_states: Dict[str, Dict[str, Any]],
        skip_count: int,
        invalid_action_count: int,
    ) -> Dict[str, Any]:
        if not email_states:
            return {"score": 0.0, "breakdown": {}, "passed": False}

        per_email_scores: Dict[str, float] = {}
        per_email_components: Dict[str, Dict[str, float]] = {}

        for email_id, state in email_states.items():
            _, components = _score_single_email(
                email_id, state, require_hints=False
            )
            # Escalation weighted 1.5x in medium
            components["escalation_score"] = components.get("escalation_score", 0.0) * 1.5
            total_weight = 1 + 1 + 1 + 1.5  # priority + dept + response + escalation
            weighted_sum = (
                components.get("priority_score", 0.0)
                + components.get("department_score", 0.0)
                + components.get("response_score", 0.0)
                + components.get("escalation_score", 0.0)
            )
            score = weighted_sum / total_weight
            per_email_scores[email_id] = score
            per_email_components[email_id] = components

        base_score = (
            sum(per_email_scores.values()) / len(per_email_scores)
            if per_email_scores else 0.0
        )

        # Stricter penalties in medium
        invalid_penalty = min(invalid_action_count * 0.03, 0.25)
        excess_skips = max(0, skip_count - 1)
        skip_penalty = min(excess_skips * 0.05, 0.20)

        final_score = max(0.0, base_score - invalid_penalty - skip_penalty)
        final_score = round(min(final_score, 1.0), 4)

        return {
            "score": final_score,
            "base_score": round(base_score, 4),
            "invalid_penalty": round(invalid_penalty, 4),
            "skip_penalty": round(skip_penalty, 4),
            "per_email_scores": per_email_scores,
            "per_email_components": per_email_components,
            "passed": final_score >= 0.60,
        }


class HardGrader:
    """
    HARD task grader.
    - No category hints whatsoever
    - Escalation weighted 2x (many hard emails require escalation)
    - Response quality weighted 1.5x (nuanced language required)
    - Zero skip tolerance
    - Harder invalid-action penalty
    Passing threshold: >= 0.50
    """

    TASK_ID = "hard"

    @staticmethod
    def grade(
        email_states: Dict[str, Dict[str, Any]],
        skip_count: int,
        invalid_action_count: int,
    ) -> Dict[str, Any]:
        if not email_states:
            return {"score": 0.0, "breakdown": {}, "passed": False}

        per_email_scores: Dict[str, float] = {}
        per_email_components: Dict[str, Dict[str, float]] = {}

        for email_id, state in email_states.items():
            _, components = _score_single_email(
                email_id, state, require_hints=False
            )
            # Escalation 2x, response 1.5x weights
            esc = components.get("escalation_score", 0.0) * 2.0
            resp = components.get("response_score", 0.0) * 1.5
            pri = components.get("priority_score", 0.0)
            dept = components.get("department_score", 0.0)

            total_weight = 1 + 1 + 1.5 + 2.0
            weighted_sum = pri + dept + resp + esc
            score = weighted_sum / total_weight

            components["escalation_score_weighted"] = esc
            components["response_score_weighted"] = resp

            per_email_scores[email_id] = score
            per_email_components[email_id] = components

        base_score = (
            sum(per_email_scores.values()) / len(per_email_scores)
            if per_email_scores else 0.0
        )

        # Harsh penalties in hard mode
        invalid_penalty = min(invalid_action_count * 0.05, 0.30)
        skip_penalty = min(skip_count * 0.07, 0.25)  # no free skips

        final_score = max(0.0, base_score - invalid_penalty - skip_penalty)
        final_score = round(min(final_score, 1.0), 4)

        return {
            "score": final_score,
            "base_score": round(base_score, 4),
            "invalid_penalty": round(invalid_penalty, 4),
            "skip_penalty": round(skip_penalty, 4),
            "per_email_scores": per_email_scores,
            "per_email_components": per_email_components,
            "passed": final_score >= 0.50,
        }


# ── Grader registry ───────────────────────────────────────────────────────────

GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}


def grade_episode(
    difficulty: str,
    email_states: Dict[str, Dict[str, Any]],
    skip_count: int,
    invalid_action_count: int,
) -> Dict[str, Any]:
    """Entry point — selects grader by difficulty and returns grading result."""
    grader_cls = GRADERS.get(difficulty)
    if grader_cls is None:
        raise ValueError(f"Unknown difficulty: {difficulty!r}")
    return grader_cls.grade(email_states, skip_count, invalid_action_count)
