# 🧠 FocusFlow: LLM-Hard RL Environment for Cognitive Management
### Meta × Scaler OpenEnv Hackathon 2026 — Grand Finale Submission

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Live%20API-yellow)](https://huggingface.co/spaces/hannan2859r/focusflow_env)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Links:
Google_colab:(https://colab.research.google.com/drive/16wJ4mw6sdcTuOYABpdoV2AuO6_KYnc4Q?usp=sharing)
Hugging_Face:(https://huggingface.co/spaces/hannan2859r/focusflow_env)

> **Executive Summary:** FocusFlow is an OpenEnv-compliant reinforcement learning environment that simulates the cognitive friction of modern digital life. It abandons traditional spatial tasks (like moving a robot arm) in favor of **LLM-hard cognitive tasks**: managing mental energy, tracking shifting deadlines, and utilizing natural language comprehension to filter informal social distractions from urgent professional tasks.

---

## 🎯 Hackathon Theme Alignment

**Core Themes Addressed:** Long-Horizon Planning & Instruction Following | World Modeling across Professional/Personal Tasks

* **The Problem Statement:** Modern digital workspaces cause catastrophic context-switching. Traditional RL bots fail here because evaluating a distraction requires contextual language understanding. The problem is designing an environment that forces an AI agent to manage time, mental energy, and dynamic deadlines while processing rich natural-language interruptions.
* **The Environment:** A fully Dockerized, RESTful API environment. The world state dynamically models time progression, cognitive load (rising with work, decaying with breaks), and an event engine that injects multi-tiered distractions.
* **Agent Capabilities Required:** Agents must possess reading comprehension (urgency evaluation), multi-day memory (tracking deferred events before they expire), and Chain-of-Thought (CoT) reasoning to justify scheduling decisions.

---

## 🏗️ System Architecture & Observation Space

The environment operates via a FastAPI backend, serving strictly typed JSON payloads. The observation space is designed to be highly complex, forcing the LLM to synthesize multiple data streams.

### Example Observation Payload
```json
{
  "time_remaining_seconds": 1140,
  "current_phase": "focus",
  "sessions_completed": 1,
  "focus_score": 0.923,
  "cognitive_load": 0.62,
  "deadline_pressure": 0.45,
  "active_distractions": ["Instagram", "BGMI"],
  "blocked_apps": ["YouTube"],
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

---

## ⚖️ Dual-Layer Reward Model & Evaluation Logic

FocusFlow implements a hybrid objective/subjective reward function. 

### 1. Objective Mechanical Rewards
| Action | Environmental Trigger | Reward / Penalty |
|---|---|---|
| `focus` | Executed during work phase | `+0.05 × (1 − cognitive_load)` |
| `block_app` | Targets an active high-temptation app | `+0.20 × temptation_level` |
| `take_break` | Executed when `cognitive_load > 0.75` | `+0.20` to `+0.30` |
| `defer_event` | Postpones a low-urgency social text | `+0.15` (Correct) / `-0.05` (Wrong) |
| `respond_to_event` | Handles urgent/hard deadlines | `+0.20` (Correct) / `-0.10` (Wrong) |
| `plan_day` | Sets schedule aligning with deadlines | `+0.00` to `+0.30` (Quality scaled) |
| `check_app` | **(BAD)** Agent gives in to temptation | **`-0.50` Hard Penalty** |

### 2. Subjective Reasoning Grader
To prevent random action-spamming, the `grade_reasoning()` heuristic parses the agent's mandatory reasoning field. 
* It applies a `±0.10` multiplier based on the use of causal language, task-awareness, and logical alignment with the current `pending_event`. 
* Empty or repetitive reasoning results in immediate reward degradation.

---

## 📋 Task Progressions

| Task ID | Challenge Pillar | Success Criteria | Horizon |
|---|---|---|---|
| `task_1` | **Execution** | Complete a 25-min session with 0 app checks. Handle basic distractions logically. | 60 Steps |
| `task_2` | **Load Management** | Complete a multi-session day. Keep `cognitive_load < 0.85` via strategic breaks. | 120 Steps |
| `task_3` | **Long-Horizon** | Execute a 3-day plan, manage energy decay, and maintain a perfect focus streak. | 240 Steps |

---

## 🚀 Post-Training & Self-Improvement Strategy (GRPO)

A baseline LLM will struggle with FocusFlow's delayed rewards (e.g., deferring an event now to save energy for a deadline 50 steps later). 

To achieve an optimal policy, the project includes a **Group Relative Policy Optimization (GRPO)** pipeline:
1.  **Framework:** Uses `TRL` (Transformer Reinforcement Learning) and `Unsloth` for efficient 4-bit quantization on consumer hardware (T4 GPUs).
2.  **Data Generation:** The baseline agent explores the live FastAPI environment, collecting trajectories of observations, actions, and rewards.
3.  **Optimization:** GRPO updates the LLM weights directly based on the environment's trajectory rewards, teaching the model that maintaining cognitive load and providing high-quality reasoning yields the highest cumulative return.

---

## 💻 Technical Setup & Quick Start

### Local Installation
```bash
# Clone the repository
git clone [https://github.com/abdulhannan-18/Focus_Flow_env.git](https://github.com/abdulhannan-18/Focus_Flow_env.git)
cd Focus_Flow_env

# Install dependencies
pip install -r requirements.txt

# Start the OpenEnv FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### API Interaction Example
```bash
# Reset the environment for Task 2
curl -X POST "http://localhost:7860/reset?task_id=task_2"

# Step the environment
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "defer_event",
    "reasoning": "This is a low-urgency gaming request. I have a Math assignment due at step 45, so I must defer this and maintain my focus phase.",
    "response_text": "Can not play right now, studying."
  }'
```

---


## 📁 Repository Architecture

```text
Focus_Flow_env/
├── app.py              # FastAPI OpenEnv REST Server
├── environment.py      # Core RL dynamics, Cognitive Load Math, Event Engine
├── models.py           # Pydantic schemas enforcing strict LLM typing
├── inference.py        # Baseline LLM Agent (CoT Prompting)
├── training_colab.py   # GRPO Fine-tuning Pipeline
├── openenv.yaml        # OpenEnv specifications
├── Dockerfile          # HuggingFace Space deployment configuration
└── requirements.txt    # Python dependencies
```

---
**Submitted by:** Abdul Hannan
**Event:** Meta × Scaler OpenEnv Hackathon 2026
