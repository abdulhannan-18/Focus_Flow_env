"""
FocusFlow RL Environment — models.py
OpenEnv hackathon submission: Meta x Scaler 2026
Pydantic models for Action, Observation, State
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from enum import Enum


class AppCategory(str, Enum):
    social_media = "social_media"
    video        = "video"
    messaging    = "messaging"
    gaming       = "gaming"
    news         = "news"


class DistractingApp(BaseModel):
    name: str
    category: AppCategory
    temptation_level: float = Field(..., ge=0.0, le=1.0, description="How tempting (0=low, 1=high)")


# ─── Action ───────────────────────────────────────────────────────────────────

class FocusAction(BaseModel):
    """
    The agent submits one of these actions each step.

    action_type options:
      - focus        : continue working, no distractions
      - block_app    : block a specific distracting app
      - take_break   : voluntarily take a break (strategic)
      - check_app    : give in to a distraction (penalised)
      - adjust_timer : change the current pomodoro duration
    """
    action_type: Literal["focus", "block_app", "take_break", "check_app", "adjust_timer"]
    app_name: Optional[str]       = Field(None, description="App to block or check (if applicable)")
    timer_minutes: Optional[int]  = Field(None, ge=5, le=60, description="New timer duration (adjust_timer only)")
    reasoning: Optional[str]      = Field(None, description="Agent's reasoning for this action (used by LLM grader)")


# ─── Observation ──────────────────────────────────────────────────────────────

class FocusObservation(BaseModel):
    """What the agent sees after each step."""
    time_remaining_seconds: int              = Field(..., description="Seconds left in current session")
    current_phase: Literal["focus", "break"] = Field(..., description="Whether we are in a focus or break phase")
    active_distractions: List[str]           = Field(..., description="Apps currently tempting the agent")
    blocked_apps: List[str]                  = Field(..., description="Apps the agent has blocked so far")
    sessions_completed: int                  = Field(..., description="Number of completed pomodoro sessions")
    focus_score: float                       = Field(..., ge=0.0, le=1.0, description="Running focus quality score")
    last_action_feedback: str                = Field(..., description="Human-readable feedback on last action")
    distraction_event: Optional[str]         = Field(None, description="A new temptation that just appeared, if any")


# ─── State ────────────────────────────────────────────────────────────────────

class FocusState(BaseModel):
    """Full internal environment state (returned by state() API call)."""
    episode_step: int
    max_steps: int
    total_focus_seconds: int
    total_distraction_seconds: int
    sessions_completed: int
    breaks_taken: int
    apps_blocked: List[str]
    apps_checked: List[str]         = Field(default_factory=list, description="Distractions the agent gave in to")
    current_phase: Literal["focus", "break"]
    time_remaining_seconds: int
    cumulative_reward: float
    done: bool
