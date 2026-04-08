"""
OpenEnv-compliant HTTP server for EmailTriageEnv.
Exposes:
  POST /reset
  POST /step
  GET  /state
  GET  /grade
  GET  /health
"""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, Observation, StepResult
from environment import EmailTriageEnv

app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv-compliant email triage RL environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment state (single-session for simplicity; extend to sessions dict for multi-agent)
_env: Dict[str, EmailTriageEnv] = {}


class ResetRequest(BaseModel):
    difficulty: str = "easy"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "EmailTriageEnv"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    if request.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
    env = EmailTriageEnv(difficulty=request.difficulty)
    _env["default"] = env
    return env.reset()


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
