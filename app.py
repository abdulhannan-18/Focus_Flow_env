"""
FocusFlow RL Environment — app.py
FastAPI server exposing the OpenEnv HTTP API.

Endpoints:
  POST /reset          → FocusObservation
  POST /step           → FocusObservation + reward + done
  GET  /state          → FocusState (full internal debug state)
  GET  /health         → {"status": "ok"}
  GET  /tasks          → list of all tasks
  GET  /metrics        → episode-level training metrics (for reward curve UI)
  POST /reset_metrics  → clear metrics history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import FocusAction, FocusObservation, FocusState
from environment import FocusFlowEnvironment, TASKS
from typing import Optional, List, Dict
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title       = "FocusFlow RL Environment",
    description = (
        "OpenEnv-compatible RL environment for student focus & distraction management. "
        "LLM-hard: requires natural language reasoning, multi-day planning, "
        "and urgency-aware event handling."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

# Map session IDs to their specific environment instances
sessions: Dict[str, FocusFlowEnvironment] = {}

# Track metrics and episode counts per session
session_metrics: Dict[str, List[dict]] = {}
session_episodes: Dict[str, int] = {}


# ── Response model ────────────────────────────────────────────────────────────
class StepResponse(FocusObservation):
    reward:  float
    done:    bool
    info:    dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "environment":     "FocusFlow",
        "version":         "2.0.0",
        "sessions_active": len(sessions),
    }


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id":          t["id"],
                "description": t["description"],
                "max_steps":   t["max_steps"],
                "bonus_desc":  t.get("bonus_desc", ""),
                "days":        t.get("days", 1),
            }
            for t in TASKS
        ]
    }


@app.post("/reset", response_model=FocusObservation)
def reset(task_id: str = "task_1", seed: int = 42, session_id: str = "default"):
    """
    Reset the environment for a new episode.
    Call this before the first /step and at the start of each new episode.
    """
    valid_ids = [t["id"] for t in TASKS]
    if task_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {valid_ids}"
        )
    
    if session_id not in session_episodes:
        session_episodes[session_id] = 0
        session_metrics[session_id] = []
        
    sessions[session_id] = FocusFlowEnvironment(task_id=task_id, seed=seed)
    session_episodes[session_id] += 1
    
    return sessions[session_id].reset()


@app.post("/step", response_model=StepResponse)
def step(action: FocusAction, session_id: str = "default"):
    """
    Submit one action. Returns next observation + reward + done flag.

    The `reasoning` field in FocusAction is REQUIRED and graded.
    Empty or low-quality reasoning incurs a reward penalty.
    """
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not initialised. Call POST /reset first."
        )
        
    obs, reward, done, info = env.step(action)

    # Log for metrics endpoint
    session_metrics[session_id].append({
        "episode":    session_episodes[session_id],
        "step":       info["step"],
        "reward":     reward,
        "cumulative": info["cumulative"],
        "reasoning_q": obs.reasoning_quality_score,
        "success":    info.get("success", False),
    })

    return StepResponse(
        **obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=FocusState)
def state(session_id: str = "default"):
    """Return full internal environment state (for debugging and logging)."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not initialised. Call POST /reset first."
        )
    return env.state()


@app.get("/metrics")
def metrics(session_id: str = "default"):
    """
    Returns per-step training metrics for reward curve plotting.
    Use this in your Colab notebook to visualise training progress.
    """
    metrics_log = session_metrics.get(session_id, [])
    
    if not metrics_log:
        return {"message": "No data yet. Run some episodes first.", "data": []}

    # Aggregate by episode
    from collections import defaultdict
    ep_rewards = defaultdict(float)
    ep_steps   = defaultdict(int)
    ep_success = defaultdict(bool)
    for m in metrics_log:
        ep = m["episode"]
        ep_rewards[ep] += m["reward"]
        ep_steps[ep]   += 1
        ep_success[ep]  = ep_success[ep] or m["success"]

    episodes_summary = [
        {
            "episode":      ep,
            "total_reward": round(ep_rewards[ep], 4),
            "steps":        ep_steps[ep],
            "success":      ep_success[ep],
        }
        for ep in sorted(ep_rewards.keys())
    ]

    return {
        "total_steps":    len(metrics_log),
        "total_episodes": len(episodes_summary),
        "episodes":       episodes_summary,
        "raw_steps":      metrics_log[-100:],   # last 100 steps
    }


@app.post("/reset_metrics")
def reset_metrics(session_id: str = "default"):
    """Clear the metrics log. Call this between training runs."""
    session_metrics[session_id] = []
    session_episodes[session_id] = 0
    return {"message": f"Metrics cleared for session '{session_id}'."}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
