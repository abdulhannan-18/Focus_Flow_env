# FocusFlow RL Environment v2.0
### Meta × Scaler OpenEnv Hackathon 2026

> An LLM-hard RL environment where an AI agent manages a student's real cognitive world —
> navigating natural language distractions, shifting deadlines, and multi-day energy dynamics.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-colab-link)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/your-space)

---

## Why This Environment Is LLM-Hard

Unlike toy RL environments solvable by a simple rule-based policy, FocusFlow requires genuine LLM reasoning:

| Challenge | Why It Needs an LLM |
|---|---|
| Natural language distraction events | Agent must read and interpret messages to judge urgency |
| Mandatory `reasoning` field (graded) | Empty reasoning = reward penalty. LLMs must justify decisions |
| Cognitive load dynamics | Overworking degrades future rewards — requires adaptive strategy |
| Multi-day deadline tracking | Planning today affects energy and deadlines tomorrow |
| Deferred events expire | Agent must track time-sensitive commitments across steps |
| Urgency vs. deferability trade-off | "Mom called twice" ≠ "Friend wants to play BGMI" |

A simple `if-else` policy cannot pass Task 2 or Task 3 without understanding language.

---

## Environment Design

### Action Space (8 actions)

| Action | When to Use | Reward |
|---|---|---|
| `focus` | Stay on task | +0.05 × (1 − cognitive_load) |
| `block_app` | Block a distracting app | +0.20 × temptation_level |
| `take_break` | Rest at session boundary or when load > 0.75 | +0.20 to +0.30 |
| `defer_event` | Postpone a low-urgency event | +0.15 if correct, −0.05 if wrong |
| `respond_to_event` | Handle urgent events immediately | +0.20 if correct |
| `plan_day` | Set a study schedule at day start | +0.00 to +0.30 based on quality |
| `adjust_energy` | Recover from fatigue/environmental noise | +0.10 |
| `check_app` | **(BAD)** Give in to distraction | −0.50 |

### Reasoning Quality Reward (Universal)

Every action carries a **reasoning bonus/penalty** (±0.10) based on:
- Mentions of relevant concepts (urgency, priority, focus, deadlines)
- Use of causal language ("because", "therefore", "in order to")
- Whether the action matches the correct response for the active event

This is what separates an LLM from a rule-based bot.

### Observation Space

```json
{
  "time_remaining_seconds": 1140,
  "current_phase": "focus",
  "sessions_completed": 1,
  "focus_score": 0.923,
  "active_distractions": ["Instagram", "BGMI"],
  "blocked_apps": ["YouTube", "Netflix"],
  "cognitive_load": 0.62,
  "deadline_pressure": 0.45,
  "pending_event": {
    "type": "social_message",
    "description": "Rahul texted: 'bhai BGMI chalate hain, sirf 1 ghanta, kal exam nahi hai'",
    "urgency": 0.30,
    "can_defer": true,
    "deadline_steps": 8,
    "correct_action": "defer_event"
  },
  "day_context": {
    "day_number": 1,
    "energy_level": 0.84,
    "pending_deadlines": [
      {"task": "Math Assignment", "due_step": 45, "completed": false}
    ]
  },
  "last_action_feedback": "Well-timed break: +0.30 | Good reasoning (0.82): +0.10",
  "reasoning_quality_score": 0.82
}
```

### Tasks

| Task | Description | Max Steps | Key Challenge |
|---|---|---|---|
| `task_1` | One session, zero distractions | 60 | Reasoning quality + event handling |
| `task_2` | Two sessions, manage cognitive load | 120 | Break timing + multi-event judgment |
| `task_3` | 3-day week plan, maintain streak | 240 | Long-horizon planning + energy decay |

---

## Reward Function Summary

```
Universal (every step):
  reasoning_quality ∈ [-0.10, +0.10]   scored by heuristic grader

Action-specific:
  focus            → +0.05 × (1 - cognitive_load)
  block_app        → +0.20 × temptation_level
  take_break       → +0.20 to +0.30 (well-timed) or -0.10 (premature)
  defer_event      → +0.15 (correct) / -0.05 (wrong) / -0.20 (non-deferrable)
  respond_to_event → +0.20 (correct) / -0.10 (wrong)
  plan_day         → +0.00 to +0.30 (based on plan quality scoring)
  adjust_energy    → +0.10 (when needed) / +0.01 (unnecessary)
  check_app        → -0.50 (hard penalty)

Episode bonuses:
  task_1: +0.25 if avg reasoning quality > 70%
  task_2: +0.30 for zero app checks across both sessions
  task_3: +0.40 for 3-day perfect focus streak
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Start environment server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Reset and step (in another terminal)
curl -X POST "http://localhost:7860/reset?task_id=task_1"

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "defer_event",
    "event_id": "evt_3",
    "reasoning": "This is a low urgency social message from a friend asking to play games. Since I have a Math Assignment due at step 45 and I am currently in focus phase, I should defer this and stay focused. The friend can wait.",
    "response_text": "bhai abhi padh raha hoon, baad mein baat karte hain"
  }'
```

### Run LLM Agent

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export GROQ_API_KEY=your_key_here
export MODEL_NAME=llama-3.1-8b-instant
export ENV_BASE_URL=http://localhost:7860
export TASK_ID=task_2
export MAX_EPISODES=5

python inference.py
```

### Train with GRPO (Google Colab T4)

Open `training_colab.py` in Colab. It will:
1. Load `Llama-3.2-1B-Instruct` with Unsloth 4-bit quantisation
2. Collect environment episodes as training data
3. Fine-tune with GRPO using environment rewards
4. Plot reward curves
5. Push trained model to HuggingFace Hub

---

## Project Structure

```
focusflow_rl_env/
├── models.py           # Pydantic: FocusAction, FocusObservation, FocusState, DistractionEvent
├── environment.py      # Core RL logic: step(), reset(), reward, NL event grading
├── app.py              # FastAPI server (OpenEnv HTTP API + /metrics endpoint)
├── inference.py        # LLM baseline agent with chain-of-thought prompting
├── training_colab.py   # GRPO training script (Unsloth + HF TRL)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## What Was Upgraded in v2.0

| Feature | v1 (original) | v2 (this submission) |
|---|---|---|
| Distraction events | App names only | Rich NL messages with urgency & deferability |
| Reasoning | Not required | Mandatory, graded, rewarded |
| Action space | 5 simple actions | 8 actions including plan_day, defer, respond |
| Cognitive load | Not modelled | Dynamic: rises with focus, falls with breaks |
| Multi-day context | Single session | 3-day week with energy decay & deadlines |
| Training script | Missing | Full GRPO Colab notebook with reward curves |
| Success criteria | Fixed string eval() | Type-safe lambda functions |
| Metrics endpoint | None | /metrics for reward curve plotting |

---

## Submitted by

Abdul Hannan — Meta × Scaler OpenEnv Hackathon 2026
