"""
FocusFlow RL Environment — inference.py
 
LLM agent that:
  1. Reads the full observation (NL events, deadlines, cognitive load)
  2. Produces a chain-of-thought reasoning string
  3. Selects the best action
  4. Logs reward curves for showing training progress
 
Usage:
  export API_BASE_URL=https://api.groq.com/openai/v1
  export MODEL_NAME=llama-3.1-8b-instant
  export ENV_BASE_URL=http://localhost:7860
  export TASK_ID=task_1
  python inference.py
"""
 
import os
import json
import requests
import time
from typing import Optional
from openai import OpenAI
 
# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
ENV_URL      = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_ID      = os.getenv("TASK_ID",      "task_1")
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "5"))
 
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", "dummy")),
    base_url=API_BASE_URL,
)
 
SYSTEM_PROMPT = """You are FocusAgent — an AI assistant helping a student stay focused during study sessions.
 
You will receive an observation about the student's current state and must choose the best action.
 
AVAILABLE ACTIONS:
- focus              : Stay on task, no special action needed
- block_app          : Block a distracting app (specify app_name)
- take_break         : Take a study break
- defer_event        : Postpone handling a distraction event for later
- respond_to_event   : Immediately handle an urgent event (provide response_text)
- plan_day           : Set a day plan (provide day_plan as a list of steps)
- adjust_energy      : Do something to restore energy/focus (stretch, hydrate, etc.)
- check_app          : (BAD) Give in to distraction — avoid this!
- quit_session       : (BAD) End the session early — avoid this!
 
DECISION FRAMEWORK:
1. Check pending_event first — is it urgent? Can it be deferred?
2. Check cognitive_load — if > 0.75, consider a break
3. Check deadline_pressure — if > 0.7, prioritise study tasks
4. Block apps proactively, especially high-temptation ones
5. Plan the day at the start of each day (step 1)
 
CRITICAL: Your `reasoning` field MUST explain:
  - What the most important signal in the observation is
  - Why you chose this action over alternatives
  - How this action serves the long-term goal
 
Poor reasoning = lower reward. Think carefully."""
 
 
def obs_to_prompt(obs: dict, step: int) -> str:
    event_str = "None"
    if obs.get("pending_event"):
        e = obs["pending_event"]
        event_str = (
            f"[{e['type']}] {e['description']} "
            f"(urgency={e['urgency']:.2f}, can_defer={e['can_defer']}, "
            f"expires_in={e.get('deadline_steps', 'N/A')} steps)"
        )
 
    dc = obs.get("day_context", {})
    deadlines = dc.get("pending_deadlines", [])
    dl_str = ", ".join(
        f"{d['task']} (due step {d['due_step']})"
        for d in deadlines if not d.get("completed")
    ) or "None"
 
    return f"""=== STEP {step} OBSERVATION ===
 
SESSION STATE:
  Phase            : {obs['current_phase']}
  Time remaining   : {obs['time_remaining_seconds']//60}m {obs['time_remaining_seconds']%60}s
  Sessions done    : {obs['sessions_completed']}
  Focus score      : {obs['focus_score']:.3f}
 
ENVIRONMENT:
  Active distractions : {', '.join(obs['active_distractions']) or 'None'}
  Blocked apps        : {', '.join(obs['blocked_apps']) or 'None'}
 
COGNITIVE & ENERGY:
  Cognitive load    : {obs['cognitive_load']:.2f}  {'⚠ HIGH' if obs['cognitive_load'] > 0.75 else '✓ OK'}
  Energy level      : {dc.get('energy_level', 1.0):.2f}
  Deadline pressure : {obs['deadline_pressure']:.2f}  {'⚠ URGENT' if obs['deadline_pressure'] > 0.7 else '✓ OK'}
 
PENDING EVENT:
  {event_str}
 
PENDING DEADLINES:
  {dl_str}
 
LAST FEEDBACK:
  {obs['last_action_feedback']}
  Last reward      : {obs.get('last_action_reward', 0.0):.4f}
  Reasoning score  : {obs.get('reasoning_quality_score', 0.0):.2f}
 
Choose your next action and provide clear reasoning."""
 
 
def call_llm(prompt: str) -> dict:
    """Call the LLM and parse its action response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt + "\n\nRespond ONLY with valid JSON matching FocusAction schema."}
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)
 
 
def run_episode(task_id: str, episode: int) -> dict:
    """Run a single episode. Returns episode stats."""
    # Reset environment
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()
 
    step           = 0
    total_reward   = 0.0
    reward_history = []
    done           = False
 
    print(f"\n{'='*60}")
    print(f"EPISODE {episode+1} | Task: {task_id}")
    print(f"{'='*60}")
 
    while not done:
        step += 1
        prompt = obs_to_prompt(obs, step)
 
        # LLM decides action
        try:
            action = call_llm(prompt)
        except Exception as e:
            print(f"  [LLM ERROR] {e} — defaulting to focus")
            action = {"action_type": "focus", "reasoning": "LLM call failed, defaulting to focus."}
 
        # Ensure reasoning is present
        if not action.get("reasoning") or len(action["reasoning"].strip()) < 10:
            action["reasoning"] = "Staying focused to complete the session efficiently."
 
        # Step environment
        try:
            resp = requests.post(f"{ENV_URL}/step", json=action)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"  [ENV ERROR] {e}")
            break
 
        obs    = result
        reward = result.get("last_action_reward", 0.0)
        done   = result.get("done", False)
 
        total_reward += reward
        reward_history.append(reward)
 
        print(
            f"  Step {step:3d} | action={action['action_type']:<18} "
            f"reward={reward:+.4f} | cumulative={total_reward:.4f} | "
            f"reasoning_q={result.get('reasoning_quality_score', 0):.2f}"
        )
 
        time.sleep(0.1)   # rate limit
 
    success = result.get("success", False) if not isinstance(result, Exception) else False
    print(f"\n  Episode {episode+1} done. Total reward: {total_reward:.4f} | Success: {success}")
 
    return {
        "episode":      episode + 1,
        "total_reward": round(total_reward, 4),
        "steps":        step,
        "success":      success,
        "reward_history": reward_history,
    }
 
 
def main():
    print(f"FocusFlow Agent | Model: {MODEL_NAME} | Task: {TASK_ID}")
    print(f"Environment: {ENV_URL}")
    print()
 
    # Health check
    h = requests.get(f"{ENV_URL}/health")
    print(f"Health: {h.json()}\n")
 
    all_stats = []
    for ep in range(MAX_EPISODES):
        stats = run_episode(TASK_ID, ep)
        all_stats.append(stats)
 
    # Summary
    rewards = [s["total_reward"] for s in all_stats]
    print(f"\n{'='*60}")
    print(f"SUMMARY over {MAX_EPISODES} episodes:")
    print(f"  Rewards : {[f'{r:.3f}' for r in rewards]}")
    print(f"  Mean    : {sum(rewards)/len(rewards):.4f}")
    print(f"  Best    : {max(rewards):.4f}")
    print(f"  Success : {sum(s['success'] for s in all_stats)}/{MAX_EPISODES}")
 
    # Save for reward curve plotting
    with open("reward_log.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("\nReward log saved to reward_log.json")
 
 
if __name__ == "__main__":
    main()
 
