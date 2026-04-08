"""
FocusFlow RL Environment — app.py
FastAPI server exposing the OpenEnv HTTP API:
  POST /reset
  POST /step
  GET  /state
  GET  /health
  GET  /tasks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import FocusAction, FocusObservation, FocusState
from environment import FocusFlowEnvironment, TASKS
from typing import Optional
import uvicorn

app = FastAPI(
    title="FocusFlow RL Environment",
    description="OpenEnv-compatible RL environment for student focus & anti-distraction agent training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment per server instance (stateful server pattern as per OpenEnv)
env: Optional[FocusFlowEnvironment] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "FocusFlow", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {"tasks": TASKS}


@app.post("/reset", response_model=FocusObservation)
def reset(task_id: str = "task_1", seed: int = 42):
    """
    Reset the environment and return initial observation.
    Optionally specify which task to load.
    """
    global env
    if task_id not in [t["id"] for t in TASKS]:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}. Available: {[t['id'] for t in TASKS]}")
    env = FocusFlowEnvironment(task_id=task_id, seed=seed)
    obs = env.reset()
    return obs


class StepResponse(FocusObservation):
    reward: float
    done: bool
    info: dict


@app.post("/step", response_model=StepResponse)
def step(action: FocusAction):
    """
    Submit one action and receive the next observation + reward.
    """
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    obs, reward, done, info = env.step(action)
    return StepResponse(**obs.model_dump(), reward=reward, done=done, info=info)


@app.get("/state", response_model=FocusState)
def state():
    """Return the full internal environment state."""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return env.state()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
