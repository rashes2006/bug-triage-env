"""
Core Environment implementation for the Bug Triage Environment.

Episode structure
-----------------
Each task presents a fixed set of bug reports (one per difficulty tier) in
order. The episode ends when all bugs in the set have been triaged.

  easy   → 1 bug  (max 1 step)
  medium → 3 bugs (max 3 steps)
  hard   → 3 bugs (max 3 steps, harder cases)

The reward returned at each step is the graded score for that individual bug.
The cumulative episode reward is the mean score across all bugs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import TriageAction, TriageObservation, TriageState
from server.bug_corpus import BUGS_BY_TASK, BugRecord
from server.grader import GraderResult, grade


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

TASK_BUG_COUNTS: dict[str, int] = {
    "easy": 1,
    "medium": 3,
    "hard": 3,
}

# Maximum steps before forced episode termination (safety)
MAX_STEPS = 10


class BugTriageEnvironment(Environment):
    """
    A real-world bug triage environment.

    Task IDs
    --------
    'easy'   : A single, clearly-described bug with unambiguous signals.
    'medium' : Three bugs requiring careful log analysis and multi-team reasoning.
    'hard'   : Three ambiguous bugs with conflicting signals and sparse evidence.
    """

    def __init__(self) -> None:
        self._state = TriageState()
        self._bugs_by_id: dict[str, BugRecord] = {}
        self._current_bug: Optional[BugRecord] = None
        self._last_result: Optional[GraderResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs,
    ) -> TriageObservation:
        """Start a new episode for the given task."""
        if task_id not in BUGS_BY_TASK:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(BUGS_BY_TASK)}")

        rng = random.Random(seed)

        # Sample the bug set for this task
        bugs_for_task = BUGS_BY_TASK[task_id]
        n = TASK_BUG_COUNTS[task_id]
        if seed is not None:
            sampled = rng.sample(bugs_for_task, min(n, len(bugs_for_task)))
        else:
            sampled = list(bugs_for_task[:n])  # deterministic by default

        self._bugs_by_id = {b.bug_id: b for b in sampled}
        queue = [b.bug_id for b in sampled]

        import uuid
        ep_id = episode_id or str(uuid.uuid4())

        self._state = TriageState(
            task_id=task_id,
            episode_id=ep_id,
            step_count=0,
            bug_queue=queue,
            cumulative_reward=0.0,
            done=False,
        )

        self._current_bug = self._bugs_by_id[queue[0]] if queue else None
        self._last_result = None

        return self._make_observation(reward=0.0, done=False, feedback="New episode started. Triage the bug report above.")

    def step(self, action: TriageAction) -> TriageObservation:
        """Triage the current bug and advance the episode."""
        if self._state.done:
            return self._make_observation(
                reward=0.0, done=True, feedback="Episode already done. Call reset() to start again."
            )
        if self._current_bug is None:
            return self._make_observation(
                reward=0.0, done=True, feedback="No bug to triage. Call reset() first."
            )

        # Grade the action
        result = grade(action.model_dump(), self._current_bug)
        self._last_result = result
        self._state.step_count += 1
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + result.reward, 4
        )

        # Advance queue
        self._state.bug_queue.pop(0)
        done = len(self._state.bug_queue) == 0 or self._state.step_count >= MAX_STEPS

        if done:
            self._state.done = True
            self._current_bug = None
            avg = round(self._state.cumulative_reward / self._state.step_count, 4)
            feedback = (
                f"{result.feedback}\n\n"
                f"Episode complete. Steps={self._state.step_count}, "
                f"Mean reward={avg:.3f}"
            )
        else:
            next_bug_id = self._state.bug_queue[0]
            self._current_bug = self._bugs_by_id[next_bug_id]
            feedback = result.feedback

        return self._make_observation(reward=result.reward, done=done, feedback=feedback)

    def state(self) -> TriageState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, reward: float, done: bool, feedback: str) -> TriageObservation:
        bug = self._current_bug
        if bug is None:
            return TriageObservation(
                bug_id="",
                title="",
                description="No active bug. Call reset() to start.",
                reporter="",
                created_at="",
                environment={},
                attachments=[],
                task_id=self._state.task_id,
                step_count=self._state.step_count,
                reward=reward,
                done=done,
                feedback=feedback,
            )
        return TriageObservation(
            bug_id=bug.bug_id,
            title=bug.title,
            description=bug.description,
            reporter=bug.reporter,
            created_at=bug.created_at,
            environment=bug.environment,
            attachments=bug.attachments,
            task_id=self._state.task_id,
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            feedback=feedback,
        )
