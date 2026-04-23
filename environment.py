"""
OmniTriageEnv — OpenEnv-compliant omnichannel triage environment.

Implements the required OpenEnv interface:
    reset() -> Observation
    step(action: Action) -> StepResult
    state() -> Observation
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from models import (
    Action,
    ActionType,
    Department,
    Email,
    EmailState,
    Observation,
    Priority,
    Reward,
    StepResult,
    TaskDifficulty,
)
from emails import EMAILS, TASK_EMAIL_IDS
from grader import grade_episode

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HISTORY_LEN = 20
SKIP_BUDGET = {
    TaskDifficulty.EASY:   2,
    TaskDifficulty.MEDIUM: 1,
    TaskDifficulty.HARD:   0,
}

# Reward weights
R_PRIORITY_CORRECT    =  0.15
R_DEPT_CORRECT        =  0.15
R_RESPONSE_KEYWORD    =  0.10   # per fraction of keywords matched
R_ESCALATION_CORRECT  =  0.15
R_ARCHIVE_COMPLETE    =  0.05   # bonus for cleanly closing email
R_INVALID_ACTION      = -0.10
R_LOOP_PENALTY        = -0.05
R_SKIP_OVER_BUDGET    = -0.08
R_SKIP_IN_BUDGET      = -0.01   # small cost even for budget skips


class OmniTriageEnv:
    """
    OpenEnv-compliant omnichannel triage environment.

    Episode lifecycle:
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        while not obs.done:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
        final_grade = env.grade_episode()
    """

    def __init__(self, difficulty: str = "easy") -> None:
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be easy/medium/hard, got {difficulty!r}")
        self.difficulty = TaskDifficulty(difficulty)
        self._email_ids: List[str] = TASK_EMAIL_IDS[difficulty]

        # Internal state (initialised in reset)
        self._queue: List[str] = []
        self._current_idx: int = 0
        self._email_states: Dict[str, Dict[str, Any]] = {}
        self._action_history: List[Dict[str, Any]] = []
        self._step_number: int = 0
        self._cumulative_reward: float = 0.0
        self._invalid_action_count: int = 0
        self._skip_count: int = 0
        self._done: bool = False

        self.reset()

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        self._queue = list(self._email_ids)   # deterministic order
        self._current_idx = 0
        self._email_states = {
            eid: {
                "email_id": eid,
                "priority_assigned": None,
                "department_assigned": None,
                "response_drafted": None,
                "escalated": False,
                "archived": False,
                "skip_count": 0,
            }
            for eid in self._email_ids
        }
        self._action_history = []
        self._step_number = 0
        self._cumulative_reward = 0.0
        self._invalid_action_count = 0
        self._skip_count = 0
        self._done = False
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply action and advance environment state.
        Returns StepResult with next observation, reward, done flag, and info.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        reward, info = self._apply_action(action)
        self._cumulative_reward = round(self._cumulative_reward + reward.value, 4)
        reward = Reward(
            value=reward.value,
            breakdown=reward.breakdown,
            penalty_reason=reward.penalty_reason,
            cumulative=self._cumulative_reward,
        )
        self._step_number += 1

        # Advance to next email if current one is fully processed or archived
        self._advance_if_needed()

        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> Observation:
        """Return current observation without advancing state."""
        return self._build_observation()

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        """Dispatch action to handler and compute dense reward."""
        email_id = action.email_id
        info: Dict[str, Any] = {"action": action.action_type, "email_id": email_id}

        # Validate email_id
        if email_id not in self._email_states:
            self._invalid_action_count += 1
            self._record_history(action.action_type, email_id, valid=False)
            return self._invalid_reward(f"Unknown email_id: {email_id}"), info

        # Validate it's the current email (no jumping ahead)
        current_email_id = self._current_email_id()
        if email_id != current_email_id:
            self._invalid_action_count += 1
            self._record_history(action.action_type, email_id, valid=False)
            return self._invalid_reward(
                f"Cannot act on {email_id} — current email is {current_email_id}"
            ), info

        es = self._email_states[email_id]

        # Loop detection: same action type on same email more than twice
        if self._is_loop(action):
            self._invalid_action_count += 1
            return (
                Reward(
                    value=R_LOOP_PENALTY,
                    breakdown={"loop_penalty": R_LOOP_PENALTY},
                    penalty_reason="Repeated action on same email (loop detected)",
                    cumulative=self._cumulative_reward,
                ),
                {**info, "loop_detected": True},
            )

        self._record_history(action.action_type, email_id, valid=True)

        if action.action_type == ActionType.CLASSIFY_PRIORITY:
            return self._handle_classify_priority(email_id, es, action)

        elif action.action_type == ActionType.ASSIGN_DEPARTMENT:
            return self._handle_assign_department(email_id, es, action)

        elif action.action_type == ActionType.DRAFT_RESPONSE:
            return self._handle_draft_response(email_id, es, action)

        elif action.action_type == ActionType.ESCALATE:
            return self._handle_escalate(email_id, es)

        elif action.action_type == ActionType.ARCHIVE:
            return self._handle_archive(email_id, es)

        elif action.action_type == ActionType.SKIP:
            return self._handle_skip(email_id, es)

        else:
            self._invalid_action_count += 1
            return self._invalid_reward(f"Unknown action_type: {action.action_type}"), info

    def _handle_classify_priority(
        self,
        email_id: str,
        es: Dict[str, Any],
        action: Action,
    ) -> tuple[Reward, Dict[str, Any]]:
        if action.priority is None:
            self._invalid_action_count += 1
            return self._invalid_reward("classify_priority requires priority field"), {}

        es["priority_assigned"] = action.priority.value

        from emails import GROUND_TRUTH
        gt = GROUND_TRUTH.get(email_id, {})
        correct = action.priority.value == gt.get("priority")
        score = R_PRIORITY_CORRECT if correct else -R_PRIORITY_CORRECT * 0.5

        return Reward(
            value=round(score, 4),
            breakdown={"priority_score": score},
            penalty_reason=None if correct else "Incorrect priority classification",
            cumulative=self._cumulative_reward,
        ), {"correct": correct}

    def _handle_assign_department(
        self,
        email_id: str,
        es: Dict[str, Any],
        action: Action,
    ) -> tuple[Reward, Dict[str, Any]]:
        if action.department is None:
            self._invalid_action_count += 1
            return self._invalid_reward("assign_department requires department field"), {}

        es["department_assigned"] = action.department.value

        from emails import GROUND_TRUTH
        gt = GROUND_TRUTH.get(email_id, {})
        correct = action.department.value == gt.get("department")
        score = R_DEPT_CORRECT if correct else -R_DEPT_CORRECT * 0.5

        return Reward(
            value=round(score, 4),
            breakdown={"department_score": score},
            penalty_reason=None if correct else "Incorrect department assignment",
            cumulative=self._cumulative_reward,
        ), {"correct": correct}

    def _handle_draft_response(
        self,
        email_id: str,
        es: Dict[str, Any],
        action: Action,
    ) -> tuple[Reward, Dict[str, Any]]:
        if not action.response_text or len(action.response_text.strip()) < 10:
            self._invalid_action_count += 1
            return self._invalid_reward("draft_response requires non-empty response_text (>=10 chars)"), {}

        es["response_drafted"] = action.response_text

        from emails import GROUND_TRUTH
        from grader import _keywords_found
        gt = GROUND_TRUTH.get(email_id, {})
        keywords = gt.get("response_keywords", [])
        kw_score = _keywords_found(action.response_text, keywords)
        score = round(R_RESPONSE_KEYWORD * kw_score, 4)

        return Reward(
            value=score,
            breakdown={"response_keyword_coverage": kw_score, "response_score": score},
            penalty_reason=None,
            cumulative=self._cumulative_reward,
        ), {"keyword_coverage": kw_score}

    def _handle_escalate(
        self,
        email_id: str,
        es: Dict[str, Any],
    ) -> tuple[Reward, Dict[str, Any]]:
        if es["escalated"]:
            self._invalid_action_count += 1
            return self._invalid_reward("Email already escalated"), {}

        es["escalated"] = True

        from emails import GROUND_TRUTH
        gt = GROUND_TRUTH.get(email_id, {})
        should_escalate = gt.get("escalate", False)
        score = R_ESCALATION_CORRECT if should_escalate else -R_ESCALATION_CORRECT

        return Reward(
            value=round(score, 4),
            breakdown={"escalation_score": score},
            penalty_reason=None if should_escalate else "Unnecessary escalation",
            cumulative=self._cumulative_reward,
        ), {"correct_escalation": should_escalate}

    def _handle_archive(
        self,
        email_id: str,
        es: Dict[str, Any],
    ) -> tuple[Reward, Dict[str, Any]]:
        if es["archived"]:
            self._invalid_action_count += 1
            return self._invalid_reward("Email already archived"), {}

        # Partial completeness bonus
        completed_steps = sum([
            es["priority_assigned"] is not None,
            es["department_assigned"] is not None,
            es["response_drafted"] is not None,
        ])
        completeness_bonus = R_ARCHIVE_COMPLETE * (completed_steps / 3)
        es["archived"] = True

        return Reward(
            value=round(completeness_bonus, 4),
            breakdown={"archive_completeness_bonus": completeness_bonus},
            penalty_reason=None,
            cumulative=self._cumulative_reward,
        ), {"completeness_steps": completed_steps}

    def _handle_skip(
        self,
        email_id: str,
        es: Dict[str, Any],
    ) -> tuple[Reward, Dict[str, Any]]:
        budget = SKIP_BUDGET[self.difficulty]
        self._skip_count += 1
        es["skip_count"] = es.get("skip_count", 0) + 1
        es["archived"] = True  # skip = defer to end, treated as closed for scoring

        over_budget = self._skip_count > budget
        score = R_SKIP_OVER_BUDGET if over_budget else R_SKIP_IN_BUDGET

        return Reward(
            value=round(score, 4),
            breakdown={"skip_penalty": score},
            penalty_reason="Skip over budget" if over_budget else "Skip (within budget)",
            cumulative=self._cumulative_reward,
        ), {"skip_count": self._skip_count, "over_budget": over_budget}

    # ── State helpers ─────────────────────────────────────────────────────────

    def _current_email_id(self) -> Optional[str]:
        for eid in self._queue:
            es = self._email_states[eid]
            if not es["archived"]:
                return eid
        return None

    def _advance_if_needed(self) -> None:
        """Check if all emails are archived/done → mark episode done."""
        all_done = all(
            self._email_states[eid]["archived"]
            for eid in self._queue
        )
        if all_done:
            self._done = True

    def _build_observation(self) -> Observation:
        """Construct the Observation from current environment state."""
        current_eid = self._current_email_id()
        current_email: Optional[Email] = None

        if current_eid:
            raw = EMAILS[current_eid].copy()
            # In hard mode, strip category_hint
            if self.difficulty == TaskDifficulty.HARD:
                raw["category_hint"] = None
            current_email = Email(**raw)

        queue_length = sum(
            1 for eid in self._queue
            if not self._email_states[eid]["archived"]
        )
        processed_count = len(self._queue) - queue_length
        budget = SKIP_BUDGET[self.difficulty]
        remaining_skips = max(0, budget - self._skip_count)

        return Observation(
            current_email=current_email,
            queue_length=queue_length,
            processed_count=processed_count,
            session_score=round(self._cumulative_reward, 4),
            skip_budget=remaining_skips,
            action_history=list(self._action_history[-MAX_HISTORY_LEN:]),
            task_id=self.difficulty.value,
            task_difficulty=self.difficulty,
            step_number=self._step_number,
            done=self._done,
        )

    def _record_history(
        self,
        action_type: ActionType,
        email_id: str,
        valid: bool,
    ) -> None:
        self._action_history.append({
            "step": self._step_number,
            "action_type": action_type,
            "email_id": email_id,
            "valid": valid,
        })

    def _is_loop(self, action: Action) -> bool:
        """Detect if this action type has been used > 2 times on the same email."""
        same = [
            h for h in self._action_history
            if h["email_id"] == action.email_id
            and h["action_type"] == action.action_type
        ]
        return len(same) >= 2

    @staticmethod
    def _invalid_reward(reason: str) -> Reward:
        return Reward(
            value=R_INVALID_ACTION,
            breakdown={"invalid_action_penalty": R_INVALID_ACTION},
            penalty_reason=reason,
            cumulative=0.0,
        )

    # ── Episode scoring ───────────────────────────────────────────────────────

    def grade_episode(self) -> Dict[str, Any]:
        """Return final deterministic grade for the completed episode."""
        result = grade_episode(
            difficulty=self.difficulty.value,
            email_states=self._email_states,
            skip_count=self._skip_count,
            invalid_action_count=self._invalid_action_count,
        )
        result["cumulative_reward"] = self._cumulative_reward
        result["total_steps"] = self._step_number
        return result
