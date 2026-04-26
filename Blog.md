FocusFlow: Training LLM Agents to Protect Student Attention
Meta × Scaler OpenEnv Hackathon 2026 — Abdul Hannan

Why I Built This
Hey everyone, I'm Abdul Hannan.
While looking at other hackathon submissions, I noticed most people were building bots, simple games, or environments that don't connect to real problems. I wanted to build something that actually matters.
One problem I've seen destroy people's potential — including my own — is distraction while studying. YouTube, BGMI, Instagram — these apps are scientifically engineered to steal your attention. Students lose hours every day and don't even realize it.
So I built FocusFlow — an RL environment that trains an LLM agent to protect student attention.

The Core Idea
FocusFlow is built on the Pomodoro technique — 25 minutes of deep focus, 5 minutes of break. But the agent doesn't just follow a timer. It has to:

Read natural language distraction events and judge urgency
Manage cognitive load — the mental fatigue of studying
Block the right apps at the right time
Justify every single decision with graded reasoning


What Makes It LLM-Hard
This is the part I'm most proud of.
Most RL environments can be solved by a simple rule-based bot. FocusFlow cannot.
Every action requires a mandatory reasoning field. The agent must explain:

What signal did it see?
Why did it choose this action?
How does this protect long-term focus?

Empty reasoning = -0.15 penalty. Good reasoning = +0.10 bonus.
A bot that just types "focus" repeatedly scores zero. The agent has to genuinely think.

The 6 Actions
1. block_app — Block a distracting app proactively
BGMI has temptation 0.95, YouTube 0.90. The higher the temptation, the bigger the reward for blocking early.
2. take_break — Rest when cognitive load > 0.80
If the agent keeps studying without breaks, cognitive load hits 1.0 and performance collapses. Smart agents rest before that happens.
3. defer_event — Reschedule low-urgency distractions
Example: "Rohan texted — bhai BGMI chalate hain" → urgency 0.3 → defer it, stay focused.
4. respond_to_event — Handle urgent events immediately
Example: "Professor posted — deadline moved to TODAY 11:59 PM" → urgency 0.95, can_defer=False → respond immediately or miss the deadline.
5. focus — Stay on task when no threats are active
Earns small positive reward every step. The foundation of every good session.
6. adjust_energy — Restore energy when it drops below 0.35
Multi-day sessions drain energy. The agent must manage this across 3 days in Task 3.

Anti-Reward-Hacking Design
In RL, agents are notorious for finding shortcuts — they maximize reward without actually solving the problem.
I designed FocusFlow with 4 independent reward signals that are hard to game simultaneously:
Reward = reasoning_quality × 0.20
       + action_correctness
       + cognitive_load_dynamics  
       + deadline_pressure_tracking
Plus an anti-spam check — if the agent repeats the same words in reasoning, unique_ratio drops below 0.5 and score = 0. You can't just repeat "focus focus focus" and win.

Training Results
I trained Qwen2.5-0.5B using GRPO via HuggingFace TRL.
The agent went through 3 curriculum levels:

Level 1 (Beginner): Learn to block apps
Level 2 (Intermediate): Handle events + manage breaks
Level 3 (Advanced): Full strategy across multi-day sessions

Episode reward improved from -3.50 → +2.54
The agent learned to:

Block high-temptation apps before they cause distractions
Defer low-urgency social messages
Take recovery breaks exactly when cognitive load hits 0.75+


Real World Impact
This isn't just a hackathon project.
The reward logic in FocusFlow maps directly to proven productivity strategies used by top students and professionals worldwide. With some engineering, this agent's policy could be embedded into:

OS-level focus modes
Study apps like Forest or Focusmate
Student productivity tools

500 million students globally struggle with distraction. FocusFlow is a step toward an AI that fights for their attention instead of stealing it.

Links

Live Environment: https://hannan2859r-focusflow-env.hf.space
GitHub: https://github.com/abdulhannan-18/Focus_Flow_env


Submitted by Abdul Hannan — Meta × Scaler OpenEnv Hackathon 2026
