"""
FocusFlow RL Environment — models.py
OpenEnv hackathon submission: Meta x Scaler 2026
Pydantic models for Action, Observation, State
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AppCategory(str, Enum):
    social_media = "social_media"
    video        = "video"
    messaging    = "messaging"
    gaming       = "gaming"
    news         = "news"
    shopping     = "shopping"

class SessionPhase(str, Enum):
    focus     = "focus"
    break_    = "break"
    planning  = "planning"
    overload  = "overload"

class DistractionType(str, Enum):
    app_notification  = "app_notification"
    social_message    = "social_message"
    urgent_task       = "urgent_task"
    environment_noise = "environment_noise"
    internal_urge     = "internal_urge"


# ─── Core data objects ────────────────────────────────────────────────────────

class DistractingApp(BaseModel):
    name: str
    category: AppCategory
    temptation_level: float = Field(..., ge=0.0, le=1.0)

class DistractionEvent(BaseModel):
    """Rich natural-language distraction — requires LLM reasoning to handle."""
    id: str
    type: DistractionType
    description: str
    urgency: float = Field(..., ge=0.0, le=1.0)
    can_defer: bool = True
    deadline_steps: Optional[int] = None
    correct_action: str = ""


class DayContext(BaseModel):
    """Multi-day persistent context — forces long-horizon planning."""
    day_number: int = Field(1, ge=1)
    total_days: int = Field(7, ge=1)
    pending_deadlines: List[Dict[str, Any]] = Field(default_factory=list)
    energy_level: float = Field(1.0, ge=0.0, le=1.0)
    completed_tasks: List[str] = Field(default_factory=list)
    deferred_events: List[DistractionEvent] = Field(default_factory=list)
    streak_days: int = Field(0, ge=0)


# ─── Action ───────────────────────────────────────────────────────────────────

class FocusAction(BaseModel):
    """
    Agent action. `reasoning` field is REQUIRED — forces chain-of-thought.
    This is what makes your env LLM-specific: a rule-based policy can't fill this.
    """
    action_type: Literal[
        "focus", "block_app", "take_break", "defer_event", 
        "respond_to_event", "plan_day", "adjust_energy", "quit_session", "check_app"
    ] = Field(..., description="The exact action the agent intends to take.")
    
    app_name: Optional[str]         = Field(None, description="Required if action is block_app.")
    event_id: Optional[str]         = Field(None, description="Required for defer/respond actions.")
    response_text: Optional[str]    = Field(None, description="Agent's NL reply to a social message.")
    timer_minutes: Optional[int]    = Field(None, ge=1)
    day_plan: Optional[List[str]]   = Field(None, description="List of tasks for plan_day action.")
    
    reasoning: str = Field(
        ...,
        min_length=10,
        description="MANDATORY: Agent MUST explain its reasoning. Empty or short reasoning = heavy penalty."
    )


# ─── Observation ──────────────────────────────────────────────────────────────

class FocusObservation(BaseModel):
    time_remaining_seconds: int = Field(..., ge=0)
    current_phase: SessionPhase
    sessions_completed: int = Field(..., ge=0)
    focus_score: float = Field(..., ge=0.0, le=1.0)
    
    active_distractions: List[str] = Field(default_factory=list)
    blocked_apps: List[str] = Field(default_factory=list)
    pending_event: Optional[DistractionEvent] = None
    
    day_context: DayContext
    cognitive_load: float = Field(0.0, ge=0.0, le=1.0)
    deadline_pressure: float = Field(0.0, ge=0.0, le=1.0)
    
    last_action_feedback: str
    last_action_reward: float
    reasoning_quality_score: float = Field(0.0, ge=0.0, le=1.0)


# ─── Full internal state (for /state endpoint) ────────────────────────────────

class FocusState(BaseModel):
    episode_step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    total_focus_seconds: int = Field(..., ge=0)
    total_distraction_seconds: int = Field(..., ge=0)
    sessions_completed: int = Field(..., ge=0)
    breaks_taken: int = Field(..., ge=0)
    
    apps_blocked: List[str] = Field(default_factory=list)
    apps_checked: List[str] = Field(default_factory=list)
    events_deferred: List[str] = Field(default_factory=list)
    events_responded: List[str] = Field(default_factory=list)
    
    current_phase: SessionPhase
    time_remaining_seconds: int = Field(..., ge=0)
    cumulative_reward: float
    day_context: DayContext
    cognitive_load: float = Field(..., ge=0.0, le=1.0)
    done: bool
