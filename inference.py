"""
Inference script for the Bug Triage Environment.

Calls BugTriageEnvironment DIRECTLY (no HTTP/WebSocket) so it works
both inside the hackathon validator container and in local testing.

Environment variables (injected by hackathon LiteLLM proxy):
  API_KEY          - LLM proxy key  (fallback: OPENAI_API_KEY)
  API_BASE_URL     - LLM proxy URL  (fallback: OPENAI_BASE_URL)
  OPENAI_MODEL     - model name     (default: gpt-4o-mini)

Usage
-----
    # Dry-run (random actions, no API key needed):
    python inference.py --dry-run

    # Real LLM via hackathon proxy:
    API_KEY=<key> API_BASE_URL=<url> python inference.py

    # Override tasks:
    python inference.py --tasks easy --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Adjust sys.path so `models` and `server` are importable when the script
# is run from /tmp/workspace (hackathon validator) or from the project root.
# ---------------------------------------------------------------------------
import pathlib

_HERE = pathlib.Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Try to import openai; graceful fallback for dry-run mode
# ---------------------------------------------------------------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import the environment directly — no network required
# ---------------------------------------------------------------------------
try:
    from server.bug_triage_environment import BugTriageEnvironment
    from models import TriageAction
    ENV_AVAILABLE = True
except Exception as _env_import_err:  # noqa: BLE001
    ENV_AVAILABLE = False
    _ENV_IMPORT_ERR = _env_import_err


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
# LLM helper
# ---------------------------------------------------------------------------

def call_llm(client: "openai.OpenAI", model: str, obs) -> dict:
    """Call the LLM and parse the JSON triage decision."""
    attachments_text = "\n".join(getattr(obs, "attachments", [])) or "(none)"
    env_text = json.dumps(getattr(obs, "environment", {}))

    user_msg = USER_PROMPT_TEMPLATE.format(
        bug_id=getattr(obs, "bug_id", ""),
        title=getattr(obs, "title", ""),
        reporter=getattr(obs, "reporter", ""),
        created_at=getattr(obs, "created_at", ""),
        environment=env_text,
        description=getattr(obs, "description", ""),
        attachments=attachments_text,
    )

    try:
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
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}", flush=True)
        return _fallback_action()

    # Strip optional markdown fences
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        action = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  LLM returned invalid JSON: {exc}\n  Raw: {raw_text[:200]}", flush=True)
        return _fallback_action()

    return _sanitise(action)


def _fallback_action() -> dict:
    return {
        "priority": "medium",
        "labels": ["bug"],
        "assigned_team": "backend",
        "needs_escalation": False,
        "comment": "Fallback — could not get a valid LLM response.",
    }


def _sanitise(action: dict) -> dict:
    if action.get("priority") not in VALID_PRIORITIES:
        action["priority"] = "medium"
    action["labels"] = [la for la in action.get("labels", []) if la in VALID_LABELS] or ["bug"]
    if action.get("assigned_team") not in VALID_TEAMS:
        action["assigned_team"] = "backend"
    if not isinstance(action.get("needs_escalation"), bool):
        action["needs_escalation"] = False
    return action


def random_action() -> dict:
    rng = random.Random()
    return _sanitise({
        "priority": rng.choice(VALID_PRIORITIES),
        "labels": rng.sample(VALID_LABELS, k=rng.randint(1, 3)),
        "assigned_team": rng.choice(VALID_TEAMS),
        "needs_escalation": rng.random() > 0.7,
        "comment": "random baseline action",
    })


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    env: "BugTriageEnvironment",
    task_id: str,
    llm_client: "openai.OpenAI | None",
    model: str,
    dry_run: bool,
    seed: int = 42,
) -> float:
    """Run one task episode and return mean reward."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Task: {task_id.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"[START] task={task_id}", flush=True)

    obs = env.reset(task_id=task_id, seed=seed)

    total_reward = 0.0
    step_count = 0

    while not getattr(obs, "done", False):
        bug_id = getattr(obs, "bug_id", "?")
        title = getattr(obs, "title", "(no title)")
        print(f"\n  [Step {step_count+1}] Bug: {bug_id} — {title[:60]}", flush=True)

        if dry_run or llm_client is None:
            raw = random_action()
            print(f"  Action (random): priority={raw['priority']}, team={raw['assigned_team']}", flush=True)
        else:
            raw = call_llm(llm_client, model, obs)
            print(f"  Action (LLM): priority={raw['priority']}, team={raw['assigned_team']}, escalate={raw['needs_escalation']}", flush=True)

        try:
            action = TriageAction(**raw)
            obs = env.step(action)
        except Exception as e:
            print(f"  ⚠️  Step failed: {e}", flush=True)
            break

        reward = getattr(obs, "reward", 0.0) or 0.0
        feedback = getattr(obs, "feedback", "")
        step_count += 1
        total_reward += reward

        print(f"  Reward: {reward:.3f}", flush=True)
        print(f"[STEP] step={step_count} reward={reward}", flush=True)
        for line in feedback.splitlines():
            print(f"    {line}", flush=True)

        if step_count > 20:
            print("  ⚠️  Safety guard: exceeded max steps", flush=True)
            break

    mean_reward = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n  ✅ Task complete. Steps={step_count}, Mean reward={mean_reward:.3f}", flush=True)
    print(f"[END] task={task_id} score={mean_reward} steps={step_count}", flush=True)
    return mean_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bug Triage Environment — Inference")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use random actions (no LLM needed)")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"],
                        choices=["easy", "medium", "hard"])
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Guard: environment must be importable
    # ------------------------------------------------------------------
    if not ENV_AVAILABLE:
        print(f"❌ Could not import BugTriageEnvironment: {_ENV_IMPORT_ERR}", flush=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build LLM client (uses hackathon-injected proxy creds first)
    # ------------------------------------------------------------------
    llm_client = None
    if not args.dry_run:
        if not OPENAI_AVAILABLE:
            print("⚠️  openai not installed — falling back to dry-run mode.", flush=True)
            args.dry_run = True
        else:
            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if not api_key:
                print("⚠️  No API key set — falling back to dry-run mode.", flush=True)
                args.dry_run = True
            else:
                llm_client = openai.OpenAI(
                    api_key=api_key,
                    **({"base_url": base_url} if base_url else {}),
                )
                print(f"✅ Using model: {args.model} via {base_url or 'OpenAI default'}", flush=True)

    # ------------------------------------------------------------------
    # Instantiate environment directly — no network, no WebSocket
    # ------------------------------------------------------------------
    env = BugTriageEnvironment()
    print("✅ BugTriageEnvironment instantiated directly.", flush=True)

    # ------------------------------------------------------------------
    # Run tasks
    # ------------------------------------------------------------------
    task_scores: dict[str, float] = {}
    for task_id in args.tasks:
        try:
            score = run_task(
                env=env,
                task_id=task_id,
                llm_client=llm_client,
                model=args.model,
                dry_run=args.dry_run,
                seed=args.seed,
            )
        except Exception as e:
            print(f"❌ Task '{task_id}' failed: {e}", flush=True)
            score = 0.0
            print(f"[END] task={task_id} score={score} steps=0", flush=True)
        task_scores[task_id] = score
        time.sleep(0.2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print("  BASELINE SCORES SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model : {'random' if args.dry_run else args.model}", flush=True)
    print(f"  Seed  : {args.seed}", flush=True)
    print(flush=True)
    for tid, sc in task_scores.items():
        bar = "█" * int(sc * 20)
        print(f"  {tid:<8} {sc:.3f}  {bar}", flush=True)
    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    print(f"\n  Overall mean: {overall:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
