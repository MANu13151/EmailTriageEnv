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

from typing import Optional
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
