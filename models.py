"""
Data models for the Bug Triage Environment.

A real-world environment that simulates GitHub/Jira-style bug triage:
agents must read bug reports and make correct triage decisions
(priority, label, assigned team, and whether it needs escalation).

Action space:
  - priority:     "critical" | "high" | "medium" | "low"
  - labels:       list of tags from a fixed vocabulary
  - assigned_team: "backend" | "frontend" | "infra" | "security" | "mobile" | "data"
  - needs_escalation: bool
  - comment:      short free-text justification (optional, used for partial credit)

Observation space:
  - bug_id:       unique identifier
  - title:        one-line bug summary
  - description:  full bug description
  - reporter:     username of reporter
  - created_at:   ISO timestamp
  - environment:  {"os": ..., "browser": ..., "version": ...}
  - attachments:  list of log snippet strings
  - task_id:      which task configuration is active
  - step_count:   how many steps in this episode
  - reward:       reward from the last step
  - done:         whether the episode is complete
  - feedback:     human-readable feedback from grader
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State

class TriageState(State):
    task_id: str = "easy"
    episode_id: str = ""
    step_count: int = 0
    bug_queue: list[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Vocabulary constants (shared between models and graders)
# ---------------------------------------------------------------------------

VALID_PRIORITIES = Literal["critical", "high", "medium", "low"]

VALID_LABELS = Literal[
    "bug",
    "regression",
    "security",
    "performance",
    "ux",
    "data-loss",
    "crash",
    "memory-leak",
    "api",
    "documentation",
    "flaky-test",
    "needs-repro",
]

VALID_TEAMS = Literal["backend", "frontend", "infra", "security", "mobile", "data"]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TriageAction(Action):
    """
    A triage decision made by the agent.

    The agent reads a bug report and fills in all fields below.
    Partial credit is given even when some fields are wrong.
    """

    priority: VALID_PRIORITIES = Field(
        ...,
        description=(
            "Severity priority: 'critical' (system down/data loss), "
            "'high' (major feature broken), 'medium' (degraded but workaround exists), "
            "'low' (cosmetic/minor)."
        ),
    )

    labels: list[VALID_LABELS] = Field(
        ...,
        min_length=1,
        max_length=4,
        description=(
            "One to four labels from the allowed vocabulary that best describe the bug."
        ),
    )

    assigned_team: VALID_TEAMS = Field(
        ...,
        description="Engineering team that should own this issue.",
    )

    needs_escalation: bool = Field(
        ...,
        description=(
            "Whether this bug should be immediately escalated to on-call / management."
        ),
    )

    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional short justification for the triage decision.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TriageObservation(Observation):
    """
    A bug report presented to the agent, plus metadata about episode progress.
    """

    # --- Bug report fields ---
    bug_id: str = Field(..., description="Unique bug identifier, e.g. 'BUG-1042'.")
    title: str = Field(..., description="One-line summary of the bug.")
    description: str = Field(..., description="Full description of the bug report.")
    reporter: str = Field(..., description="Username of the person who filed the report.")
    created_at: str = Field(..., description="ISO-8601 timestamp of report creation.")
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Runtime environment: OS, browser, app version, etc.",
    )
    attachments: list[str] = Field(
        default_factory=list,
        description="Log snippets or stack traces attached to the report.",
    )

    # --- Episode metadata ---
    task_id: str = Field(..., description="Active task identifier ('easy'|'medium'|'hard').")
    step_count: int = Field(default=0, ge=0, description="Number of steps taken so far.")
    reward: float = Field(default=0.0, description="Reward received from the last action.")
    done: bool = Field(default=False, description="True when the episode has ended.")
    feedback: str = Field(
        default="",
        description="Human-readable feedback from the grader about the last action.",
    )
