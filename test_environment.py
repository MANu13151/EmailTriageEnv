"""
tests/test_environment.py — Validation tests for OmniTriageEnv.
Run: python -m pytest test_environment.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import Action, ActionType, Priority, Department
from environment import OmniTriageEnv


class TestOpenEnvInterface:
    """Verify OpenEnv interface is correctly implemented."""

    def test_reset_returns_observation(self):
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        assert obs is not None
        assert obs.current_email is not None
        assert obs.queue_length == 14
        assert obs.processed_count == 0
        assert not obs.done

    def test_state_returns_observation(self):
        env = OmniTriageEnv(difficulty="easy")
        obs = env.state()
        assert obs is not None
        assert obs.step_number == 0

    def test_step_returns_step_result(self):
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        email_id = obs.current_email.email_id
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id=email_id,
            priority=Priority.URGENT,
        )
        result = env.step(action)
        assert result.observation is not None
        assert result.reward is not None
        assert result.reward.value is not None
        assert isinstance(result.done, bool)

    def test_reward_in_range(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        email_id = "E001"
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id=email_id,
            priority=Priority.URGENT,
        )
        result = env.step(action)
        assert -1.0 <= result.reward.value <= 1.0


class TestDeterminism:
    """Verify same inputs produce same outputs."""

    def test_deterministic_reset(self):
        env1 = OmniTriageEnv(difficulty="easy")
        env2 = OmniTriageEnv(difficulty="easy")
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.current_email.email_id == obs2.current_email.email_id
        assert obs1.queue_length == obs2.queue_length

    def test_deterministic_grading(self):
        """Same sequence of actions → same final score."""
        def run_fixed_episode(diff):
            env = OmniTriageEnv(difficulty=diff)
            obs = env.reset()
            # Take fixed actions on first email only
            eid = obs.current_email.email_id
            actions = [
                Action(action_type=ActionType.CLASSIFY_PRIORITY, email_id=eid, priority=Priority.URGENT),
                Action(action_type=ActionType.ASSIGN_DEPARTMENT, email_id=eid, department=Department.BILLING),
                Action(action_type=ActionType.DRAFT_RESPONSE, email_id=eid, response_text="We will process your refund and apologize for the inconvenience."),
                Action(action_type=ActionType.ARCHIVE, email_id=eid),
            ]
            for a in actions:
                if not obs.done:
                    result = env.step(a)
                    obs = result.observation
            return env.grade_episode()

        grade1 = run_fixed_episode("easy")
        grade2 = run_fixed_episode("easy")
        assert grade1["score"] == grade2["score"], "Grading must be deterministic"


class TestRewardFunction:
    """Verify reward signals are correct."""

    def test_correct_priority_gives_positive_reward(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        # E001 has ground truth priority=urgent
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id="E001",
            priority=Priority.URGENT,
        )
        result = env.step(action)
        assert result.reward.value > 0, "Correct priority should give positive reward"

    def test_wrong_priority_gives_negative_reward(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        # E001 ground truth = urgent; we send LOW
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id="E001",
            priority=Priority.LOW,
        )
        result = env.step(action)
        assert result.reward.value < 0, "Wrong priority should give negative reward"

    def test_invalid_action_penalized(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        # Send action with wrong email_id
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id="NONEXISTENT",
            priority=Priority.URGENT,
        )
        result = env.step(action)
        assert result.reward.value < 0, "Invalid action should be penalized"

    def test_loop_detection(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        eid = "E001"
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            email_id=eid,
            priority=Priority.URGENT,
        )
        env.step(action)
        env.step(action)
        result = env.step(action)  # Third time — loop
        assert result.reward.value < 0

    def test_empty_response_invalid(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        action = Action(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id="E001",
            response_text="Hi",  # too short
        )
        result = env.step(action)
        assert result.reward.value < 0


class TestGraders:
    """Verify grader scoring logic."""

    def test_perfect_episode_score_near_1(self):
        from emails import GROUND_TRUTH
        from grader import grade_episode

        # Build perfect email states
        email_states = {}
        for eid, gt in GROUND_TRUTH.items():
            if eid.startswith("E"):
                # Build a perfect response containing all keywords
                kws = gt.get("response_keywords", [])
                resp = " ".join(kws) + " customer support resolved apologize processed"
                email_states[eid] = {
                    "priority_assigned": gt["priority"],
                    "department_assigned": gt["department"],
                    "response_drafted": resp,
                    "escalated": gt.get("escalate", False),
                    "archived": True,
                    "skip_count": 0,
                }

        result = grade_episode("easy", email_states, skip_count=0, invalid_action_count=0)
        assert result["score"] >= 0.85, f"Perfect episode should score high, got {result['score']}"

    def test_empty_episode_scores_minimum(self):
        from grader import grade_episode
        result = grade_episode("easy", {}, skip_count=0, invalid_action_count=0)
        assert result["score"] == 0.0001, f"Empty episode should return minimum clamped score, got {result['score']}"

    def test_grader_score_in_range(self):
        from grader import grade_episode
        from emails import TASK_EMAIL_IDS

        states = {eid: {"priority_assigned": None, "department_assigned": None,
                        "response_drafted": None, "escalated": False,
                        "archived": True, "skip_count": 1}
                  for eid in TASK_EMAIL_IDS["medium"]}
        result = grade_episode("medium", states, skip_count=5, invalid_action_count=3)
        assert 0.0 < result["score"] < 1.0


class TestEscalation:
    """Verify escalation scoring works correctly."""

    def test_correct_escalation_gives_positive_reward(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        # E004 ground truth: escalate=True
        eid = "E004"
        # Process first 3 emails to get to E004
        for skip_eid in ["E001", "E002", "E003"]:
            env.step(Action(action_type=ActionType.ARCHIVE, email_id=skip_eid))

        result = env.step(Action(action_type=ActionType.ESCALATE, email_id=eid))
        assert result.reward.value > 0, "Correct escalation should give positive reward"

    def test_unnecessary_escalation_gives_negative_reward(self):
        env = OmniTriageEnv(difficulty="easy")
        env.reset()
        # E001 ground truth: escalate=False
        result = env.step(Action(action_type=ActionType.ESCALATE, email_id="E001"))
        assert result.reward.value < 0, "Unnecessary escalation should give negative reward"


class TestEpisodeLifecycle:
    """Full episode integration tests."""

    def test_full_easy_episode_completes(self):
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        steps = 0
        while not obs.done and steps < 200:
            eid = obs.current_email.email_id if obs.current_email else None
            if eid is None:
                break
            action = Action(
                action_type=ActionType.ARCHIVE,
                email_id=eid,
            )
            result = env.step(action)
            obs = result.observation
            steps += 1
        assert obs.done, "Episode should complete"
        grade = env.grade_episode()
        assert 0.0 < grade["score"] < 1.0

    def test_step_after_done_raises(self):
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        # Skip all emails
        for _ in range(15):
            if obs.done:
                break
            eid = obs.current_email.email_id if obs.current_email else None
            if not eid:
                break
            result = env.step(Action(action_type=ActionType.ARCHIVE, email_id=eid))
            obs = result.observation

        if obs.done:
            with pytest.raises(RuntimeError):
                env.step(Action(action_type=ActionType.ARCHIVE, email_id="E001"))

    def test_all_difficulties_run(self):
        """All three difficulty levels should initialize and run."""
        for diff in ["easy", "medium", "hard"]:
            env = OmniTriageEnv(difficulty=diff)
            obs = env.reset()
            assert obs.queue_length == 14
            assert obs.current_email is not None
            assert obs.task_difficulty.value == diff

    def test_complete_triage_workflow(self):
        """Test the full classify→assign→draft→escalate→archive workflow."""
        env = OmniTriageEnv(difficulty="easy")
        obs = env.reset()
        eid = obs.current_email.email_id  # E001

        # Full workflow
        r1 = env.step(Action(action_type=ActionType.CLASSIFY_PRIORITY, email_id=eid, priority=Priority.URGENT))
        assert r1.reward.value > 0  # Correct priority

        r2 = env.step(Action(action_type=ActionType.ASSIGN_DEPARTMENT, email_id=eid, department=Department.BILLING))
        assert r2.reward.value > 0  # Correct department

        r3 = env.step(Action(
            action_type=ActionType.DRAFT_RESPONSE, email_id=eid,
            response_text="We sincerely apologize for the double charge. Your refund has been processed and should appear within 3-5 business days."
        ))
        assert r3.reward.value >= 0  # Keywords matched

        r4 = env.step(Action(action_type=ActionType.ARCHIVE, email_id=eid))
        assert r4.reward.value >= 0  # Archive bonus

        # E001 is now archived, queue should have 13 left
        assert r4.observation.queue_length == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
