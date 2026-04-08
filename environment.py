"""
FocusFlow RL Environment — environment.py
Core logic: tasks, reward shaping, grader, episode management
"""

import random
from typing import Tuple, List, Optional
from models import (
    FocusAction, FocusObservation, FocusState,
    DistractingApp, AppCategory
)


# ─── Configurable tasks ───────────────────────────────────────────────────────

TASKS = [
    {
        "id": "task_1",
        "description": "Complete one 25-minute focus session without checking any distracting app.",
        "success_condition": "sessions_completed >= 1 and len(apps_checked) == 0",
        "max_steps": 60,
        "bonus": "Block at least 3 apps before the session ends for a 0.2 bonus.",
    },
    {
        "id": "task_2",
        "description": "Complete two focus sessions with strategically timed breaks (take_break at the right time).",
        "success_condition": "sessions_completed >= 2 and breaks_taken >= 2",
        "max_steps": 120,
        "bonus": "Never check a distracting app for a full 0.15 bonus.",
    },
    {
        "id": "task_3",
        "description": "Manage a high-distraction environment: block all 5 apps within 10 steps and maintain focus.",
        "success_condition": "len(apps_blocked) >= 5 and sessions_completed >= 1",
        "max_steps": 80,
        "bonus": "Block all apps within first 8 steps for 0.25 bonus.",
    },
]

# ─── Distraction pool ─────────────────────────────────────────────────────────

DISTRACTION_POOL: List[DistractingApp] = [
    DistractingApp(name="Instagram",  category=AppCategory.social_media, temptation_level=0.85),
    DistractingApp(name="YouTube",    category=AppCategory.video,        temptation_level=0.90),
    DistractingApp(name="WhatsApp",   category=AppCategory.messaging,    temptation_level=0.70),
    DistractingApp(name="Twitter",    category=AppCategory.social_media, temptation_level=0.75),
    DistractingApp(name="BGMI",       category=AppCategory.gaming,       temptation_level=0.95),
    DistractingApp(name="Reddit",     category=AppCategory.news,         temptation_level=0.80),
    DistractingApp(name="Netflix",    category=AppCategory.video,        temptation_level=0.88),
    DistractingApp(name="Snapchat",   category=AppCategory.social_media, temptation_level=0.72),
]

FOCUS_DURATION_SECONDS  = 25 * 60   # 25 minutes
SHORT_BREAK_SECONDS     = 5  * 60   # 5 minutes
LONG_BREAK_SECONDS      = 15 * 60   # 15 minutes (every 4 sessions)


class FocusFlowEnvironment:
    """
    OpenEnv-compatible RL environment for the FocusFlow anti-distraction agent.
    Implements step() / reset() / state() as per OpenEnv spec.
    """

    def __init__(self, task_id: str = "task_1", seed: int = 42):
        random.seed(seed)
        self.task = next(t for t in TASKS if t["id"] == task_id)
        self._reset_internal()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_internal(self):
        self.step_count           = 0
        self.max_steps            = self.task["max_steps"]
        self.total_focus_secs     = 0
        self.total_distraction_s  = 0
        self.sessions_completed   = 0
        self.breaks_taken         = 0
        self.apps_blocked: List[str] = []
        self.apps_checked: List[str] = []
        self.current_phase        = "focus"
        self.time_remaining       = FOCUS_DURATION_SECONDS
        self.cumulative_reward    = 0.0
        self.done                 = False
        self.active_distractions  = self._sample_distractions(3)

    def _sample_distractions(self, n: int) -> List[str]:
        """Pick n random distracting apps not already blocked."""
        available = [d.name for d in DISTRACTION_POOL if d.name not in self.apps_blocked]
        return random.sample(available, min(n, len(available)))

    def _maybe_spawn_distraction(self) -> Optional[str]:
        """30% chance each step to surface a new distraction."""
        if random.random() < 0.30:
            available = [
                d.name for d in DISTRACTION_POOL
                if d.name not in self.apps_blocked
                and d.name not in self.active_distractions
            ]
            if available:
                new_app = random.choice(available)
                self.active_distractions.append(new_app)
                return new_app
        return None

    def _compute_reward(self, action: FocusAction) -> Tuple[float, str]:
        """
        Reward function — clean and interpretable for RL training.

        Positive rewards:
          +0.5   per completed focus session (no distractions)
          +0.3   for a well-timed voluntary break
          +0.2   for blocking a high-temptation app before being distracted
          +0.05  per step spent in pure focus mode

        Negative rewards:
          -0.5   for checking a distracting app
          -0.1   for taking a break at the wrong time (mid-session, not at boundary)
          -0.05  per step in focus mode with unblocked high-temptation app active
        """
        reward = 0.0
        feedback = ""

        if action.action_type == "focus":
            reward   += 0.05
            feedback  = "Good. Staying focused adds a small step reward."

        elif action.action_type == "block_app":
            if action.app_name and action.app_name not in self.apps_blocked:
                app_obj = next((d for d in DISTRACTION_POOL if d.name == action.app_name), None)
                if app_obj:
                    self.apps_blocked.append(action.app_name)
                    if action.app_name in self.active_distractions:
                        self.active_distractions.remove(action.app_name)
                    reward   += 0.20 * app_obj.temptation_level  # scale by how tempting it was
                    feedback  = f"Blocked {action.app_name}. Reward scaled by temptation level ({app_obj.temptation_level:.2f})."
                else:
                    feedback = "App not found in distraction pool — no reward."
            else:
                feedback = "App already blocked or not specified."

        elif action.action_type == "take_break":
            if self.current_phase == "focus" and self.time_remaining <= 30:
                # Strategic: break at session boundary
                reward   += 0.30
                feedback  = "Well-timed break at session boundary! +0.30 reward."
                self.current_phase  = "break"
                self.time_remaining = SHORT_BREAK_SECONDS if (self.sessions_completed + 1) % 4 != 0 else LONG_BREAK_SECONDS
                self.breaks_taken  += 1
            elif self.current_phase == "break":
                feedback = "Already on a break. No reward."
            else:
                reward   -= 0.10
                feedback  = "Break taken mid-session. -0.10 penalty."
                self.breaks_taken += 1

        elif action.action_type == "check_app":
            app = action.app_name or (self.active_distractions[0] if self.active_distractions else None)
            if app:
                reward   -= 0.50
                feedback  = f"Gave in to {app}! Hard penalty: -0.50."
                self.apps_checked.append(app)
                self.total_distraction_s += 60  # assume 1 min lost per check
            else:
                feedback = "No active distraction to check."

        elif action.action_type == "adjust_timer":
            # Neutral but allows personalisation
            reward   += 0.01
            feedback  = f"Timer adjusted to {action.timer_minutes} min. Minimal reward."

        return reward, feedback

    def _advance_time(self, seconds: int = 60):
        """Advance simulation by `seconds`. Transitions phase when timer hits 0."""
        self.time_remaining -= seconds
        if self.time_remaining <= 0:
            if self.current_phase == "focus":
                self.sessions_completed += 1
                self.total_focus_secs   += FOCUS_DURATION_SECONDS
                # start break
                self.current_phase  = "break"
                self.time_remaining = SHORT_BREAK_SECONDS if self.sessions_completed % 4 != 0 else LONG_BREAK_SECONDS
            else:
                # break ended, start new focus session
                self.current_phase  = "focus"
                self.time_remaining = FOCUS_DURATION_SECONDS
                self.active_distractions = self._sample_distractions(2)

    def _check_success(self) -> bool:
        """Evaluate the task success condition."""
        sessions_completed = self.sessions_completed
        apps_blocked       = self.apps_blocked
        apps_checked       = self.apps_checked
        breaks_taken       = self.breaks_taken
        try:
            return eval(self.task["success_condition"])  # noqa: S307
        except Exception:
            return False

    # ── Public OpenEnv API ────────────────────────────────────────────────────

    def reset(self) -> FocusObservation:
        """Reset the environment and return the initial observation."""
        self._reset_internal()
        return FocusObservation(
            time_remaining_seconds = self.time_remaining,
            current_phase          = self.current_phase,
            active_distractions    = list(self.active_distractions),
            blocked_apps           = list(self.apps_blocked),
            sessions_completed     = self.sessions_completed,
            focus_score            = 0.0,
            last_action_feedback   = f"Environment reset. Task: {self.task['description']}",
            distraction_event      = None,
        )

    def step(self, action: FocusAction) -> Tuple[FocusObservation, float, bool, dict]:
        """
        Process one agent action.
        Returns: (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.step_count += 1

        # Advance simulated time (each step = 1 minute in the student's world)
        self._advance_time(seconds=60)

        # Compute reward and get feedback
        reward, feedback = self._compute_reward(action)

        # Maybe spawn a new distraction
        new_distraction = self._maybe_spawn_distraction()

        # Compute running focus score
        focus_ratio = (
            self.total_focus_secs /
            max(1, self.total_focus_secs + self.total_distraction_s)
        )

        # Check episode termination
        success = self._check_success()
        self.done = self.step_count >= self.max_steps or success

        self.cumulative_reward += reward

        obs = FocusObservation(
            time_remaining_seconds = self.time_remaining,
            current_phase          = self.current_phase,
            active_distractions    = list(self.active_distractions),
            blocked_apps           = list(self.apps_blocked),
            sessions_completed     = self.sessions_completed,
            focus_score            = round(focus_ratio, 3),
            last_action_feedback   = feedback,
            distraction_event      = new_distraction,
        )

        info = {
            "step":       self.step_count,
            "success":    success,
            "cumulative": round(self.cumulative_reward, 4),
        }

        return obs, round(reward, 4), self.done, info

    def state(self) -> FocusState:
        """Return the full internal state (for debugging / logging)."""
        return FocusState(
            episode_step              = self.step_count,
            max_steps                 = self.max_steps,
            total_focus_seconds       = self.total_focus_secs,
            total_distraction_seconds = self.total_distraction_s,
            sessions_completed        = self.sessions_completed,
            breaks_taken              = self.breaks_taken,
            apps_blocked              = list(self.apps_blocked),
            apps_checked              = list(self.apps_checked),
            current_phase             = self.current_phase,
            time_remaining_seconds    = self.time_remaining,
            cumulative_reward         = round(self.cumulative_reward, 4),
            done                      = self.done,
        )
