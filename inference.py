"""
Baseline inference script for the Bug Triage Environment.

Uses the OpenAI API to run a model against all 3 tasks and reports
reproducible baseline scores.

Usage
-----
    # Set credentials:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
    export OPENAI_MODEL="gpt-4o-mini"                    # optional

    # Run against a locally running server:
    python baseline.py --base-url http://localhost:8000

    # Run against a Hugging Face Space:
    python baseline.py --base-url https://<your-hf-space>.hf.space

    # Dry-run (random actions, no API key needed):
    python baseline.py --base-url http://localhost:8000 --dry-run

Expected approximate baseline scores (gpt-4o-mini, seed=42)
-------------------------------------------------------------
  easy   : ~0.85
  medium : ~0.72
  hard   : ~0.55
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Try to import openai; graceful fallback for dry-run mode
# ---------------------------------------------------------------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Try to import the environment client; fall back to pure HTTP requests
# ---------------------------------------------------------------------------
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PRIORITIES = ["critical", "high", "medium", "low"]
VALID_LABELS = [
    "bug", "regression", "security", "performance", "ux",
    "data-loss", "crash", "memory-leak", "api", "documentation",
    "flaky-test", "needs-repro",
]
VALID_TEAMS = ["backend", "frontend", "infra", "security", "mobile", "data"]

SYSTEM_PROMPT = """\
You are an expert software engineer conducting bug triage for a SaaS product.
You will be given a bug report and must make a structured triage decision.

Respond ONLY with a valid JSON object matching this schema (no markdown, no extra text):
{
  "priority": "<critical|high|medium|low>",
  "labels": ["<label1>", "<label2>"],
  "assigned_team": "<backend|frontend|infra|security|mobile|data>",
  "needs_escalation": <true|false>,
  "comment": "<one sentence justification>"
}

Priority guidelines:
- critical: system down, data loss, or security breach — escalate immediately
- high:     major feature broken or significant user impact
- medium:   degraded experience but workaround exists
- low:      cosmetic or minor annoyance

Valid labels: bug, regression, security, performance, ux, data-loss, crash,
              memory-leak, api, documentation, flaky-test, needs-repro
Valid teams:  backend, frontend, infra, security, mobile, data
"""

USER_PROMPT_TEMPLATE = """\
Bug ID: {bug_id}
Title: {title}
Reporter: {reporter}
Created: {created_at}
Environment: {environment}

Description:
{description}

Attachments:
{attachments}
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

from client import BugTriageEnv
from models import TriageAction


def call_llm(client: "openai.OpenAI", model: str, bug_obs: dict) -> dict:
    """Call the LLM and parse the JSON triage decision."""
    attachments_text = "\n".join(bug_obs.get("attachments", [])) or "(none)"
    env_text = json.dumps(bug_obs.get("environment", {}))

    user_msg = USER_PROMPT_TEMPLATE.format(
        bug_id=bug_obs.get("bug_id", ""),
        title=bug_obs.get("title", ""),
        reporter=bug_obs.get("reporter", ""),
        created_at=bug_obs.get("created_at", ""),
        environment=env_text,
        description=bug_obs.get("description", ""),
        attachments=attachments_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        seed=42,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip optional markdown fences
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        action = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  LLM returned invalid JSON: {exc}\n  Raw: {raw_text[:200]}")
        action = {
            "priority": "medium",
            "labels": ["bug"],
            "assigned_team": "backend",
            "needs_escalation": False,
            "comment": "Fallback — could not parse LLM response.",
        }

    # Sanitise
    if action.get("priority") not in VALID_PRIORITIES:
        action["priority"] = "medium"
    action["labels"] = [l for l in action.get("labels", []) if l in VALID_LABELS] or ["bug"]
    if action.get("assigned_team") not in VALID_TEAMS:
        action["assigned_team"] = "backend"
    if not isinstance(action.get("needs_escalation"), bool):
        action["needs_escalation"] = False

    return action


def random_action() -> dict:
    """Generate a random valid action (used in dry-run mode)."""
    rng = random.Random()
    return {
        "priority": rng.choice(VALID_PRIORITIES),
        "labels": rng.sample(VALID_LABELS, k=rng.randint(1, 3)),
        "assigned_team": rng.choice(VALID_TEAMS),
        "needs_escalation": rng.random() > 0.7,
        "comment": "random baseline action",
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_task(
    env: Any,
    task_id: str,
    llm_client: "openai.OpenAI | None",
    model: str,
    dry_run: bool,
    seed: int = 42,
) -> float:
    """Run one task to completion and return mean reward."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*60}")
    
    print(f"[START] task={task_id}", flush=True)

    step_result = env.reset(task_id=task_id, seed=seed)
    
    obs = step_result.observation
    done = step_result.done

    total_reward = 0.0
    step_count = 0

    while not done:
        bug_id = getattr(obs, "bug_id", "?")
        title = getattr(obs, "title", "(no title)")
        print(f"\n  [Step {step_count+1}] Bug: {bug_id} — {title[:60]}")

        if dry_run or llm_client is None:
            raw_action = random_action()
            action = TriageAction(**raw_action)
            print(f"  Action (random): priority={action.priority}, team={action.assigned_team}")
        else:
            obs_dict = obs.model_dump()
            raw_action = call_llm(llm_client, model, obs_dict)
            action = TriageAction(**raw_action)
            print(f"  Action (LLM):    priority={action.priority}, team={action.assigned_team}, escalate={action.needs_escalation}")

        step_result = env.step(action)
        reward = step_result.reward or 0.0
        obs = step_result.observation
        feedback = getattr(obs, "feedback", "")
        done = step_result.done

        total_reward += reward
        step_count += 1

        print(f"  Reward: {reward:.3f}")
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

        for line in feedback.splitlines():
            print(f"    {line}")

        if step_count > 20:  # safety guard
            print("  ⚠️  Exceeded max steps safety guard")
            break

    mean_reward = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n  ✅ Task complete. Steps={step_count}, Mean reward={mean_reward:.3f}")
    print(f"[END] task={task_id} score={mean_reward} steps={step_count}", flush=True)
    
    return mean_reward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bug Triage Environment — Baseline Inference")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="URL of the running environment server",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use random actions instead of calling the OpenAI API",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Which tasks to run",
    )
    args = parser.parse_args(argv)

    # Build Env client
    print(f"✅ Connecting to Environment at {args.base_url}")

    try:
        with BugTriageEnv(base_url=args.base_url).sync() as env:
            # Build LLM client
            llm_client = None
            if not args.dry_run:
                if not OPENAI_AVAILABLE:
                    print("⚠️  openai package not installed. Falling back to dry-run mode.")
                    args.dry_run = True
                else:
                    # Prefer hackathon-injected proxy creds; fall back to OPENAI_* vars
                    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
                    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                    if not api_key:
                        print("⚠️  No API key found (API_KEY / OPENAI_API_KEY). Falling back to dry-run mode.")
                        args.dry_run = True
                    else:
                        llm_client = openai.OpenAI(
                            api_key=api_key,
                            **({"base_url": base_url} if base_url else {}),
                        )
                        print(f"✅ Using model: {args.model} via {base_url or 'default OpenAI'}")

        # Run tasks
        task_scores: dict[str, float] = {}
        for task_id in args.tasks:
            score = run_task(
                env=env,
                task_id=task_id,
                llm_client=llm_client,
                model=args.model,
                dry_run=args.dry_run,
                seed=args.seed,
            )
            task_scores[task_id] = score
            time.sleep(0.5)  # brief pause between tasks

    except Exception as e:
        print(f"❌ Connection or unhandled environment error: {e}")
        print("Please ensure compiling OpenEnv environments are reachable explicitly before running inference scripts.")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:  {'random' if args.dry_run else args.model}")
    print(f"  Seed:   {args.seed}")
    print()
    for task_id, score in task_scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} {score:.3f}  {bar}")
    print()
    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    print(f"  Overall mean: {overall:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
