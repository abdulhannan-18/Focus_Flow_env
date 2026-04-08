"""
FocusFlow RL Environment — inference.py
HACKATHON SUBMISSION — Meta x Scaler OpenEnv 2026

CRITICAL: Logs MUST follow [START] / [STEP] / [END] format exactly.
          Uses OpenAI client as required by the hackathon spec.
          Runtime < 20 min | Runs on vcpu=2, memory=8gb
"""

import os
import json
import httpx
from openai import OpenAI

# ── Env vars (required by hackathon spec) ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",    "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN",      "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL",  "http://localhost:7860")
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "30"))

# ── OpenAI client (REQUIRED by hackathon — do not use httpx for LLM calls) ──
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an AI agent managing a student's focus session.

Goal: maximise focus, minimise distractions across the episode.

Actions you can take — respond ONLY with valid JSON:
  focus        -> stay focused (small step reward)
  block_app    -> block a distracting app (include "app_name")
  take_break   -> take a voluntary break (reward if timed at session boundary)
  check_app    -> give in to distraction (HEAVY -0.50 PENALTY, never do this)
  adjust_timer -> change pomodoro length (include "timer_minutes": int)

Response format (JSON only, no markdown fences):
{
  "action_type": "block_app",
  "app_name": "Instagram",
  "reasoning": "Block high-temptation app early."
}

Strategy:
1. Block high-temptation apps in the first few steps.
2. Stay in focus mode to accumulate +0.05 per step.
3. Take a break only when time_remaining < 60 seconds (session boundary).
4. NEVER use check_app.
"""

def call_llm(messages: list) -> dict:
    """Call LLM via OpenAI client and parse JSON action."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=200,
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def run_episode(task_id: str, episode_num: int):
    """Run one full episode with required stdout markers."""
    base = ENV_BASE_URL.rstrip("/")

    # Reset environment
    reset_resp = httpx.post(f"{base}/reset", params={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    # [START] marker
    print(f"[START] task={task_id}", flush=True)

    messages     = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    step         = 0
    done         = False

    while not done and step < MAX_STEPS:
        step += 1

        user_content = (
            f"Step {step}. "
            f"phase={obs['current_phase']} | "
            f"time_remaining={obs['time_remaining_seconds']}s | "
            f"sessions_done={obs['sessions_completed']} | "
            f"focus_score={obs['focus_score']} | "
            f"active_distractions={obs['active_distractions']} | "
            f"blocked_apps={obs['blocked_apps']} | "
            f"last_feedback={obs['last_action_feedback']} | "
            f"new_distraction={obs.get('distraction_event')}"
        )
        messages.append({"role": "user", "content": user_content})

        try:
            action = call_llm(messages)
        except Exception as e:
            action = {"action_type": "focus", "reasoning": f"LLM error: {e}"}

        messages.append({"role": "assistant", "content": json.dumps(action)})

        step_resp = httpx.post(f"{base}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()

        reward       = result["reward"]
        done         = result["done"]
        obs          = result
        total_reward += reward

        # [STEP] marker
        print(f"[STEP] step={step} reward={reward}", flush=True)

    # Normalize reward to a score between 0 and 1
    raw_score = total_reward / max(step, 1)

    # Clamp score strictly inside (0, 1)
    safe_score = max(1e-6, min(raw_score, 1 - 1e-6))

    # [END] marker
    print(f"[END] task={task_id} score={safe_score} steps={step}", flush=True)

    return {
        "episode":      episode_num,
        "task_id":      task_id,
        "total_reward": round(total_reward, 4),
        "steps":        step,
        "success":      result.get("info", {}).get("success", False),
    }

def main():
    tasks   = ["task_1", "task_2", "task_3"]
    results = []

    for i, task_id in enumerate(tasks, start=1):
        try:
            result = run_episode(task_id=task_id, episode_num=i)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] episode={i} error={e}", flush=True)

    avg_reward   = sum(r["total_reward"] for r in results) / max(len(results), 1)
    success_rate = sum(1 for r in results if r["success"]) / max(len(results), 1)
    print(f"SUMMARY avg_reward={avg_reward} success_rate={success_rate}", flush=True)

if __name__ == "__main__":
    main()
