"""
FocusFlow RL Environment — environment.py

What makes this LLM-hard (not solvable by rule-based policy):
  1. Natural-language distraction events require reading comprehension + judgment
  2. Reasoning quality is graded — empty/bad reasoning is penalised
  3. Multi-day context: decisions today affect energy/deadlines tomorrow
  4. Cognitive load dynamics: overworking degrades future performance
  5. Deferred events expire — agent must track time-sensitive commitments
  6. Plan quality grading: plan_day action scored against actual task completion
"""

import random
import re
from typing import Tuple, List, Optional, Dict, Any

from models import (
    FocusAction, FocusObservation, FocusState,
    DistractingApp, DistractionEvent, DayContext,
    AppCategory, DistractionType, SessionPhase
)

# ─── Timing constants ─────────────────────────────────────────────────────────
FOCUS_DURATION_SECONDS  = 25 * 60
SHORT_BREAK_SECONDS     = 5  * 60
LONG_BREAK_SECONDS      = 15 * 60
SECONDS_PER_STEP        = 60   # 1 step = 1 simulated minute

# ─── Distraction pool ─────────────────────────────────────────────────────────
DISTRACTION_POOL: List[DistractingApp] = [
    DistractingApp(name="Instagram",  category=AppCategory.social_media, temptation_level=0.85),
    DistractingApp(name="YouTube",    category=AppCategory.video,        temptation_level=0.90),
    DistractingApp(name="WhatsApp",   category=AppCategory.messaging,    temptation_level=0.70),
    DistractingApp(name="Twitter",    category=AppCategory.social_media, temptation_level=0.75),
    DistractingApp(name="BGMI",       category=AppCategory.gaming,       temptation_level=0.95),
    DistractingApp(name="Reddit",     category=AppCategory.news,         temptation_level=0.80),
    DistractingApp(name="Netflix",    category=AppCategory.video,        temptation_level=0.88),
    DistractingApp(name="Snapchat",   category=AppCategory.social_media, temptation_level=0.82),
    DistractingApp(name="Flipkart",   category=AppCategory.shopping,     temptation_level=0.60),
]

# ─── Rich NL distraction events ───────────────────────────────────────────────
# Each has a `correct_action` so the reward function can grade the agent's choice.
EVENT_POOL: List[Dict[str, Any]] = [
    {
        "type": DistractionType.social_message,
        "description": "Rahul texted: 'bhai BGMI chalate hain, sirf 1 ghanta, kal exam nahi hai'",
        "urgency": 0.3, "can_defer": True,  "deadline_steps": 10,
        "correct_action": "defer_event",
        "hint": "Low urgency social request during focus time → defer, don't respond now"
    },
    {
        "type": DistractionType.urgent_task,
        "description": "Professor posted: 'Assignment submission deadline moved to TODAY 11:59 PM'",
        "urgency": 0.95, "can_defer": False, "deadline_steps": 5,
        "correct_action": "respond_to_event",
        "hint": "High urgency, hard deadline → must act immediately"
    },
    {
        "type": DistractionType.social_message,
        "description": "Mom called twice. No message left.",
        "urgency": 0.80, "can_defer": True, "deadline_steps": 3,
        "correct_action": "respond_to_event",
        "hint": "Family + multiple missed calls → respond quickly"
    },
    {
        "type": DistractionType.internal_urge,
        "description": "You feel the urge to check your Instagram likes on the post from yesterday.",
        "urgency": 0.20, "can_defer": True, "deadline_steps": None,
        "correct_action": "focus",
        "hint": "Internal urge, no real urgency → ignore and stay focused"
    },
    {
        "type": DistractionType.environment_noise,
        "description": "Neighbours are playing loud music. Very hard to concentrate.",
        "urgency": 0.60, "can_defer": False, "deadline_steps": None,
        "correct_action": "adjust_energy",
        "hint": "Environmental distraction → adapt (use headphones, move location)"
    },
    {
        "type": DistractionType.urgent_task,
        "description": "Study group chat: 'Meeting in 30 min for exam prep — are you coming?'",
        "urgency": 0.70, "can_defer": True, "deadline_steps": 8,
        "correct_action": "defer_event",
        "hint": "Somewhat urgent but can defer with a quick reply"
    },
    {
        "type": DistractionType.social_message,
        "description": "Friend texted: 'Congrats bhai! You got selected for the interview at XYZ!'",
        "urgency": 0.50, "can_defer": True, "deadline_steps": None,
        "correct_action": "defer_event",
        "hint": "Good news but not urgent — acknowledge later, stay focused now"
    },
    {
        "type": DistractionType.internal_urge,
        "description": "You've been studying for 40 minutes straight. Your mind is drifting.",
        "urgency": 0.65, "can_defer": False, "deadline_steps": None,
        "correct_action": "take_break",
        "hint": "Cognitive fatigue signal → take a break before performance crashes"
    },
]

# ─── Tasks ────────────────────────────────────────────────────────────────────
#In my previous logic ive used val() it doesn't perform well and giving random outputs so I've now added lambda function to work properly and give correct output after evaluation

TASKS = [
    {
        "id": "task_1",
        "description": (
            "Single focused session: Complete one 25-min Pomodoro with zero app checks. "
            "Handle any distraction events correctly using good reasoning."
        ),
        "max_steps": 60,
        "success_fn": lambda s: s["sessions_completed"] >= 1 and len(s["apps_checked"]) == 0,
        #The bonus fn here is giving the good scores on top of it if agent did really well .
        "bonus_fn":   lambda s: 0.25 if s["reasoning_scores"] and
                                  sum(s["reasoning_scores"]) / len(s["reasoning_scores"]) > 0.7
                                  else 0.0,
        "bonus_desc": "+0.25 if average reasoning quality > 70%",
        "days": 1,
    },
    {
        "id": "task_2",
        "description": (
            "Multi-session day: Complete 2 focus sessions with well-timed breaks. "
            "Correctly defer low-urgency events and respond to high-urgency ones. "
            "Manage cognitive load — don't let it exceed 0.85."
        ),
        "max_steps": 120,
        "success_fn": lambda s: (
            s["sessions_completed"] >= 2 and
            s["breaks_taken"] >= 2 and
            s["max_cognitive_load"] <= 0.85
        ),
        "bonus_fn":   lambda s: 0.30 if len(s["apps_checked"]) == 0 else 0.0,
        "bonus_desc": "+0.30 for zero app checks across both sessions",
        "days": 1,
    },
    {
        "id": "task_3",
        "description": (
            "Week planner (3 days): Plan each day's study schedule, complete sessions, "
            "handle shifting deadlines, and maintain a focus streak. "
            "Energy degrades each day — plan accordingly."
        ),
        "max_steps": 240,
        "success_fn": lambda s: (
            s["sessions_completed"] >= 5 and
            s["streak_days"] >= 2 and
            s["deadlines_missed"] == 0
        ),
        "bonus_fn":   lambda s: 0.40 if s["streak_days"] >= 3 else 0.0,
        "bonus_desc": "+0.40 for a 3-day perfect focus streak",
        "days": 3,
    },
]


# ─── Reasoning quality grader ─────────────────────────────────────────────────

def grade_reasoning(reasoning: str, action_type: str, event: Optional[DistractionEvent]) -> float:
    """
    Simple heuristic grader for reasoning quality (0–1).
    Real training would use an LLM-as-judge here.
    """
    if not reasoning or len(reasoning.strip()) < 10:
        return 0.0

    score = 0.3   # baseline for non-empty reasoning

    text = reasoning.lower()

    # Reward mentioning relevant concepts
    #It checks how many of these words appear in the reasoning text. More relevant words = higher score.
    focus_keywords   = ["focus", "deadline", "study", "priority", "session", "pomodoro"]
    context_keywords = ["urgent", "can wait", "defer", "later", "energy", "tired", "break"]
    planning_words   = ["because", "since", "therefore", "so that", "in order to", "plan"]

    score += 0.1 * min(2, sum(1 for k in focus_keywords   if k in text)) / 2
    score += 0.2 * min(2, sum(1 for k in context_keywords if k in text)) / 2
    score += 0.2 * min(2, sum(1 for k in planning_words   if k in text)) / 2

    # Bonus: reasoning matches correct action for event
    if event and event.correct_action == action_type:
        score += 0.2
        #If score above 0.5 reward else penalty

    return round(min(1.0, score), 3)


# ─── Environment ──────────────────────────────────────────────────────────────

class FocusFlowEnvironment:
    """
    OpenEnv-compatible RL environment.

    Key upgrades over v1:
    - Rich NL distraction events with urgency & correct_action grading
    - Mandatory reasoning field scored by grade_reasoning()
    - Multi-day context with energy decay and deadline tracking
    - Cognitive load dynamics (overwork → worse performance)
    - Deferred events expire after deadline_steps
    - plan_day action graded against actual completion
    """

    def __init__(self, task_id: str = "task_1", seed: int = 42):
        random.seed(seed)
        self.task = next(t for t in TASKS if t["id"] == task_id)
        self._reset_internal()

    # ── Internal helpers ──────────────────────────────────────────────────────
    #It makes eveything to back on zero and make a fresh run state 
    def _reset_internal(self):
        self.step_count          = 0
        self.max_steps           = self.task["max_steps"]
        self.total_focus_secs    = 0
        self.total_distraction_s = 0
        self.sessions_completed  = 0
        self.breaks_taken        = 0
        self.apps_blocked: List[str]            = []
        self.apps_checked: List[str]            = []
        self.events_deferred: List[str]         = []
        self.events_responded: List[str]        = []
        self.reasoning_scores: List[float]      = []
        self.deadlines_missed                   = 0
        self.current_phase                      = "focus"
        self.time_remaining                     = FOCUS_DURATION_SECONDS
        self.cumulative_reward                  = 0.0
        self.done                               = False
        self.cognitive_load                     = 0.0
        self.max_cognitive_load                 = 0.0
        self.active_distractions: List[str]     = self._sample_apps(3)
        self.pending_event: Optional[DistractionEvent] = None
        self.day_context                        = DayContext(
            day_number=1,
            total_days=self.task["days"],
            energy_level=1.0,
            pending_deadlines=self._generate_deadlines(),
        )
        # Day plan set by agent via plan_day action
        self._agent_day_plan: List[str] = []
        self._last_reasoning_score      = 0.0
     
    def _generate_deadlines(self) -> List[Dict[str, Any]]:
        deadlines = [
            {"task": "Math Assignment",    "due_day": 1, "due_step": 45, "completed": False},
            {"task": "Physics Lab Report", "due_day": 2, "due_step": 90, "completed": False},
            {"task": "CS Project Demo",    "due_day": 3, "due_step": 200,"completed": False},
        ]
        return deadlines[:self.task["days"]]
     #Randomly picking apps which are not blocked and called at the start when new session begin 
    def _sample_apps(self, n: int) -> List[str]:
        available = [d.name for d in DISTRACTION_POOL if d.name not in self.apps_blocked]
        return random.sample(available, min(n, len(available)))
    
    def _maybe_spawn_event(self) -> Optional[DistractionEvent]:
        """25% chance per step to surface a rich NL distraction event."""
        if self.pending_event is not None:
            return None   # one event at a time
        if random.random() < 0.25:
            raw = random.choice(EVENT_POOL)
            event = DistractionEvent(
                id=f"evt_{self.step_count}",
                type=raw["type"],
                description=raw["description"],
                urgency=raw["urgency"],
                can_defer=raw["can_defer"],
                deadline_steps=raw.get("deadline_steps"),
                correct_action=raw.get("correct_action", "focus"),
            )
            return event
        return None
    
    def _tick_event(self):
        """Age pending event. Penalise if it expires un-handled."""
        if self.pending_event and self.pending_event.deadline_steps is not None:
            self.pending_event.deadline_steps -= 1
            if self.pending_event.deadline_steps <= 0:
                # Event expired
                if not self.pending_event.can_defer:
                    self.deadlines_missed += 1
                self.pending_event = None

    def _update_cognitive_load(self, action_type: str):
        """
        Cognitive load rises with focus, falls with breaks.
        High load degrades focus_score and increases distraction spawn rate.
        """
        if action_type == "focus":
            self.cognitive_load = min(1.0, self.cognitive_load + 0.04)
        elif action_type == "take_break":
            self.cognitive_load = max(0.0, self.cognitive_load - 0.25)
        elif action_type == "adjust_energy":
            self.cognitive_load = max(0.0, self.cognitive_load - 0.10)
        self.max_cognitive_load = max(self.max_cognitive_load, self.cognitive_load)
     #subtract 60 second everytime when it hits 0 
    def _advance_time(self):
        self.time_remaining -= SECONDS_PER_STEP
        if self.time_remaining <= 0:
            if self.current_phase == "focus":
                self.sessions_completed += 1
                self.total_focus_secs += FOCUS_DURATION_SECONDS
                # Mark relevant deadlines as completed
                for dl in self.day_context.pending_deadlines:
                    if not dl["completed"] and dl["due_step"] <= self.step_count:
                        dl["completed"] = True
                self.current_phase = "break"
                self.time_remaining = (
                    SHORT_BREAK_SECONDS if self.sessions_completed % 4 != 0
                    else LONG_BREAK_SECONDS
                )
                # Energy decay each completed session
                self.day_context.energy_level = max(
                    0.1,
                    self.day_context.energy_level - 0.08
                )
            else:
                self.current_phase = "focus"
                self.time_remaining = FOCUS_DURATION_SECONDS
                self.active_distractions = self._sample_apps(2)
     
    def _compute_reward(self, action: FocusAction) -> Tuple[float, str]:
        reward   = 0.0
        feedback_parts = []

        # ── 1. Reasoning quality (universal) ─────────────────────────────────
        r_score = grade_reasoning(
            action.reasoning, action.action_type, self.pending_event
        )
        self._last_reasoning_score = r_score
        self.reasoning_scores.append(r_score)

        reasoning_bonus = (r_score - 0.5) * 0.20   # range: -0.10 to +0.10
        reward += reasoning_bonus
        if r_score < 0.3:
            feedback_parts.append(f"⚠ Weak reasoning (score {r_score:.2f}): -0.10 penalty.")
        elif r_score > 0.7:
            feedback_parts.append(f"✓ Good reasoning (score {r_score:.2f}): +0.10 bonus.")

        # ── 2. Action-specific rewards ────────────────────────────────────────
        atype = action.action_type
         #focus — +0.05 × (1 − cognitive_load × 0.8)
        if atype == "focus":
            base = 0.05
            # Cognitive load penalty: reward shrinks when overloaded
            base *= max(0.2, 1.0 - self.cognitive_load * 0.8)
            reward += base
            feedback_parts.append(f"Focused. Step reward: +{base:.3f} (load={self.cognitive_load:.2f}).")
         
        elif atype == "block_app":
            if action.app_name and action.app_name not in self.apps_blocked:
                app_obj = next((d for d in DISTRACTION_POOL if d.name == action.app_name), None)
                if app_obj:
                    self.apps_blocked.append(action.app_name)
                    if action.app_name in self.active_distractions:
                        self.active_distractions.remove(action.app_name)
                    r = 0.20 * app_obj.temptation_level
                    reward += r
                    feedback_parts.append(
                        f"Blocked {action.app_name} (temptation={app_obj.temptation_level}): +{r:.2f}."
                    )
                else:
                    feedback_parts.append("App not in pool — no reward.")
            else:
                feedback_parts.append("Already blocked or not specified.")

        elif atype == "take_break":
            if self.current_phase == "focus" and self.time_remaining <= 120:
                # Well-timed: within 2 min of session end
                reward += 0.30
                feedback_parts.append("Well-timed break at session boundary: +0.30.")
                self.current_phase = "break"
                self.time_remaining = SHORT_BREAK_SECONDS
                self.breaks_taken  += 1
            elif self.cognitive_load > 0.75:
                # Needed break due to high cognitive load
                reward += 0.20
                feedback_parts.append(f"Recovery break (load={self.cognitive_load:.2f}): +0.20.")
                self.current_phase = "break"
                self.time_remaining = SHORT_BREAK_SECONDS
                self.breaks_taken  += 1
            elif self.current_phase == "break":
                feedback_parts.append("Already on break. No reward.")
            else:
                reward -= 0.10
                feedback_parts.append("Premature break: -0.10.")
                self.breaks_taken += 1
        #whether I can defer this event or not it gives reward based on the differ of the events 
        elif atype == "defer_event":
            if self.pending_event:
                if self.pending_event.can_defer:
                    r = 0.15 if self.pending_event.correct_action == "defer_event" else -0.05
                    reward += r
                    self.events_deferred.append(self.pending_event.id)
                    self.day_context.deferred_events.append(self.pending_event)
                    label = "Correct defer" if r > 0 else "Should have responded"
                    feedback_parts.append(f"{label}: {r:+.2f}.")
                    self.pending_event = None
                else:
                    reward -= 0.20
                    self.deadlines_missed += 1
                    feedback_parts.append("Cannot defer this event! -0.20 penalty.")
            else:
                feedback_parts.append("No pending event to defer.")
         #This event is urgent to do and take action urgently 
        elif atype == "respond_to_event":
            if self.pending_event:
                correct = self.pending_event.correct_action == "respond_to_event"
                r = 0.20 if correct else -0.10
                reward += r
                # Extra: score the response text quality
                if action.response_text and len(action.response_text) > 15:
                    reward += 0.05
                    feedback_parts.append("Good response text: +0.05.")
                self.events_responded.append(self.pending_event.id)
                self.pending_event = None
                feedback_parts.append(
                    f"{'Correct' if correct else 'Wrong'} response to event: {r:+.2f}."
                )
            else:
                feedback_parts.append("No pending event.")

        elif atype == "plan_day":
            if action.day_plan and len(action.day_plan) >= 2:
                # Basic plan quality: does it mention sessions and breaks?
                plan_text = " ".join(action.day_plan).lower()
                has_sessions = "focus" in plan_text or "study" in plan_text or "session" in plan_text
                has_breaks   = "break" in plan_text or "rest"  in plan_text
                has_deadlines = any(
                    dl["task"].lower().split()[0] in plan_text
                    for dl in self.day_context.pending_deadlines
                )
                score = sum([has_sessions, has_breaks, has_deadlines]) / 3.0
                reward += 0.30 * score
                self._agent_day_plan = action.day_plan
                feedback_parts.append(
                    f"Day plan quality: {score:.0%} → +{0.30*score:.2f}."
                )
            else:
                reward -= 0.10
                feedback_parts.append("Empty or trivial plan: -0.10.")
         #If energy is less or cognitive load is greater than the given criteria reward else less reward for minor tasks  
        elif atype == "adjust_energy":
            if self.day_context.energy_level < 0.5 or self.cognitive_load > 0.6:
                reward += 0.10
                feedback_parts.append("Energy management action: +0.10.")
            else:
                reward += 0.01
                feedback_parts.append("Energy fine, minor action: +0.01.")
         #It checks for app whether it is in the distraction apps or not if its not give none otherwise give -0.50 penalty 
        elif atype == "check_app":
            app = action.app_name or (
                self.active_distractions[0] if self.active_distractions else None
            )
            if app:
                reward -= 0.50
                #Which app for checked for later analysis 
                self.apps_checked.append(app)
                self.total_distraction_s += 60#Adds 60 seconds when total time wasted on distractions 
                self.cognitive_load = min(1.0, self.cognitive_load + 0.10)
                feedback_parts.append(f"Gave in to {app}: -0.50 hard penalty.")
            else:
                feedback_parts.append("No active distraction to check.")

        elif atype == "quit_session":
            reward -= 0.30
            self.done = True
            feedback_parts.append("Session quit early: -0.30.")

        else:
            reward -= 0.05
            feedback_parts.append(f"Unknown action '{atype}': -0.05.")

        return reward, " | ".join(feedback_parts)
   '''For each uncompleted deadline, it calculates how close you are to missing it. At 50+ steps away → pressure = 0.0. At 0 steps away → pressure=1.0.
  Returns the highest pressure across all deadlines. 
  This number appears in the observation so the LLM knows when to stop chatting and start studying.'''
    def _compute_deadline_pressure(self) -> float:
        if not self.day_context.pending_deadlines:
            return 0.0
        pressures = []
        for dl in self.day_context.pending_deadlines:
            if dl["completed"]:
                continue
            steps_left = dl["due_step"] - self.step_count
            if steps_left <= 0:
                pressures.append(1.0)
            else:
                pressures.append(max(0.0, 1.0 - steps_left / 50.0))
        return max(pressures) if pressures else 0.0

    # ── Public OpenEnv API ────────────────────────────────────────────────────
    def reset(self) -> FocusObservation:
        self._reset_internal()
        return FocusObservation(
            time_remaining_seconds = self.time_remaining,
            current_phase          = self.current_phase,
            active_distractions    = list(self.active_distractions),
            blocked_apps           = list(self.apps_blocked),
            sessions_completed     = 0,
            focus_score            = 0.0,
            pending_event          = None,
            day_context            = self.day_context,
            cognitive_load         = self.cognitive_load,
            deadline_pressure      = self._compute_deadline_pressure(),
            last_action_feedback   = f"Environment reset. Task: {self.task['description']}",
            last_action_reward     = 0.0,
            reasoning_quality_score= 0.0,
        )
        '''The main loop. Every call does this in order:'''
    def step(self, action: FocusAction) -> Tuple[FocusObservation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        self.step_count += 1

        # Tick timers
        self._advance_time()
        self._tick_event()
        self._update_cognitive_load(action.action_type)

        # Compute reward
        reward, feedback = self._compute_reward(action)

        # Maybe spawn new event (higher chance at high cognitive load)
        spawn_chance = 0.25 + 0.15 * self.cognitive_load
        if self.pending_event is None and random.random() < spawn_chance:
            self.pending_event = self._maybe_spawn_event()

        # Focus score
        focus_ratio = (
            self.total_focus_secs /
            max(1, self.total_focus_secs + self.total_distraction_s)
        )

        # Deadline pressure
        deadline_pressure = self._compute_deadline_pressure()

        # Success check
        state_snapshot = {
            "sessions_completed":  self.sessions_completed,
            "apps_checked":        self.apps_checked,
            "breaks_taken":        self.breaks_taken,
            "max_cognitive_load":  self.max_cognitive_load,
            "deadlines_missed":    self.deadlines_missed,
            "streak_days":         self.day_context.streak_days,
            "reasoning_scores":    self.reasoning_scores,
        }
        success   = self.task["success_fn"](state_snapshot)
        timed_out = self.step_count >= self.max_steps

        if success or timed_out:
            self.done = True
            if success:
                bonus = self.task["bonus_fn"](state_snapshot)
                reward += bonus
                if bonus > 0:
                    feedback += f" | 🎉 Bonus: +{bonus:.2f} ({self.task['bonus_desc']})"

        self.cumulative_reward += reward

        obs = FocusObservation(
            time_remaining_seconds  = self.time_remaining,
            current_phase           = self.current_phase,
            active_distractions     = list(self.active_distractions),
            blocked_apps            = list(self.apps_blocked),
            sessions_completed      = self.sessions_completed,
            focus_score             = round(focus_ratio, 3),
            pending_event           = self.pending_event,
            day_context             = self.day_context,
            cognitive_load          = round(self.cognitive_load, 3),
            deadline_pressure       = round(deadline_pressure, 3),
            last_action_feedback    = feedback,
            last_action_reward      = round(reward, 4),
            reasoning_quality_score = self._last_reasoning_score,
        )

        info = {
            "step":            self.step_count,
            "success":         success,
            "timed_out":       timed_out,
            "cumulative":      round(self.cumulative_reward, 4),
            "deadlines_missed":self.deadlines_missed,
            "reasoning_avg":   round(
                sum(self.reasoning_scores) / max(1, len(self.reasoning_scores)), 3
            ),
        }

        return obs, round(reward, 4), self.done, info

    def state(self) -> FocusState:
        return FocusState(
            episode_step             = self.step_count,
            max_steps                = self.max_steps,
            total_focus_seconds      = self.total_focus_secs,
            total_distraction_seconds= self.total_distraction_s,
            sessions_completed       = self.sessions_completed,
            breaks_taken             = self.breaks_taken,
            apps_blocked             = list(self.apps_blocked),
            apps_checked             = list(self.apps_checked),
            events_deferred          = list(self.events_deferred),
            events_responded         = list(self.events_responded),
            current_phase            = self.current_phase,
            time_remaining_seconds   = self.time_remaining,
            cumulative_reward        = round(self.cumulative_reward, 4),
            day_context              = self.day_context,
            cognitive_load           = round(self.cognitive_load, 3),
            done                     = self.done,
        )
