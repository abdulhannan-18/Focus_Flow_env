# FocusFlow RL Environment
### Meta x Scaler OpenEnv Hackathon 2026

> An RL environment where an AI agent learns to manage a student's focus session —
> blocking distracting apps, timing breaks, and maximising deep-focus time.

---

## What It Is

FocusFlow is an **OpenEnv-compatible reinforcement learning environment** built on top of
Meta's OpenEnv framework. An LLM agent is placed in a student's digital world and must:

- **Block** distracting apps (Instagram, YouTube, BGMI, etc.) before they steal focus
- **Time breaks** correctly using the Pomodoro technique (25 min focus / 5 min break)
- **Resist** distraction events that spawn randomly during the session
- **Maximise** the focus score across multiple study sessions

The environment simulates a realistic student productivity scenario — making it a strong
candidate for training agents that improve human focus and wellbeing.

---

## Environment Design

### Action Space (5 discrete actions)

| Action | Description | Reward |
|---|---|---|
| `focus` | Stay focused, do nothing | +0.05 per step |
| `block_app` | Block a distracting app | +0.20 × temptation_level |
| `take_break` | Take a voluntary break | +0.30 if timed correctly |
| `adjust_timer` | Change pomodoro duration | +0.01 |
| `check_app` | Give in to distraction | **-0.50** |

### Observation Space

```json
{
  "time_remaining_seconds": 1200,
  "current_phase": "focus",
  "active_distractions": ["Instagram", "YouTube"],
  "blocked_apps": ["BGMI"],
  "sessions_completed": 0,
  "focus_score": 0.85,
  "last_action_feedback": "Blocked BGMI. Reward scaled by temptation level (0.95).",
  "distraction_event": "Reddit"
}
```

### Reward Function

Simple, clean rewards for stable RL training (binary/shaped hybrid):

```
+ 0.05  per step in pure focus mode
+ 0.20 × temptation  for blocking an app proactively
+ 0.30  for a well-timed break (at session boundary)
- 0.50  for checking a distracting app (hard penalty)
- 0.10  for taking a break mid-session
```

### Tasks

Three tasks of increasing difficulty:

| Task | Goal | Max Steps |
|---|---|---|
| `task_1` | Complete 1 session with zero distractions | 60 |
| `task_2` | Complete 2 sessions with correct break timing | 120 |
| `task_3` | Block all 5 apps within 10 steps, then complete a session | 80 |

---

## OpenEnv API

The server exposes the standard OpenEnv HTTP API:

```
POST /reset?task_id=task_1    → FocusObservation
POST /step  (body: FocusAction) → FocusObservation + reward + done
GET  /state                   → FocusState (full internal state)
GET  /health                  → {"status": "ok"}
GET  /tasks                   → list of all tasks
```

### Quick Start (local)

```bash
# Install
pip install -r requirements.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal: reset and take a step
curl -X POST http://localhost:7860/reset?task_id=task_1
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "block_app", "app_name": "Instagram", "reasoning": "Block high temptation early"}'
```

### Run the LLM Agent

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your_token_here
export ENV_BASE_URL=http://localhost:7860
export TASK_ID=task_1

python inference.py
```

### Deploy to HF Spaces

```bash
# Install OpenEnv CLI
pip install openenv

# Push to Hugging Face Spaces
openenv deploy --space YOUR_HF_USERNAME/focusflow-env
```

---

## Project Structure

```
focusflow_rl_env/
├── models.py        # Pydantic: FocusAction, FocusObservation, FocusState
├── environment.py   # Core RL logic: step(), reset(), state(), reward
├── app.py           # FastAPI server exposing OpenEnv HTTP API
├── inference.py     # LLM baseline agent (Groq/OpenAI compatible)
├── Dockerfile       # Container for HF Spaces deployment
├── requirements.txt
├── openenv.yaml     # OpenEnv metadata
└── README.md
```

---

## Why This Problem?

Student distraction is one of the most real, measurable problems in the world.
Phones, social media and short-form video are scientifically proven to reduce
deep work capacity. An RL agent that learns optimal focus management strategies
could be embedded in productivity apps, study tools, or OS-level focus modes —
making it immediately useful beyond the hackathon.

---

## Submitted by
Abdul Hannan — Meta x Scaler OpenEnv Hackathon 2026
