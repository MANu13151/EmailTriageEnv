"""
EmailTriageEnv — OpenEnv-compliant email triage RL environment.

Exports the core types needed for environment interaction.
"""
from models import Action, Observation, Reward, StepResult, Email, EmailState
from models import ActionType, Priority, Department, TaskDifficulty
from environment import EmailTriageEnv

__all__ = [
    "Action",
    "Observation",
    "Reward",
    "StepResult",
    "Email",
    "EmailState",
    "ActionType",
    "Priority",
    "Department",
    "TaskDifficulty",
    "EmailTriageEnv",
]
