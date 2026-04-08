"""
Tests for the Bug Triage Environment.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import TriageAction
from server.bug_corpus import EASY_BUGS, MEDIUM_BUGS, HARD_BUGS, BUGS_BY_TASK
from server.grader import grade, GraderResult
from server.bug_triage_environment import BugTriageEnvironment


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestGrader:

    def test_perfect_easy_bug(self):
        """A perfect triage on an easy bug scores 1.0."""
        bug = EASY_BUGS[0]  # SQL injection
        action = {
            "priority": bug.gt_priority,
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": f"This is a {bug.gt_priority} {bug.gt_team} issue.",
        }
        result: GraderResult = grade(action, bug)
        assert result.reward == pytest.approx(1.0, abs=0.1)

    def test_wrong_priority_one_off_partial_credit(self):
        """Being off by one priority level gives partial credit (0.15)."""
        bug = EASY_BUGS[0]  # gt_priority = "critical"
        action = {
            "priority": "high",  # off by one
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": "",
        }
        result = grade(action, bug)
        assert result.breakdown["priority"] == pytest.approx(0.15, abs=0.01)

    def test_wrong_priority_far_off_zero(self):
        """Being two+ levels off on priority gives 0."""
        bug = EASY_BUGS[0]  # gt_priority = "critical"
        action = {
            "priority": "low",  # far off
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": "",
        }
        result = grade(action, bug)
        assert result.breakdown["priority"] == 0.0

    def test_label_f1_partial(self):
        """Partially correct labels gives proportional F1 score."""
        bug = EASY_BUGS[0]  # gt_labels = ["security", "bug"]
        action = {
            "priority": bug.gt_priority,
            "labels": ["security"],  # missing "bug"
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": "",
        }
        result = grade(action, bug)
        # F1 = 2 * (1/1) * (1/2) / (1/1 + 1/2) = 2/3 ≈ 0.667  =>  0.25 * 0.667 ≈ 0.167
        assert 0.15 < result.breakdown["labels"] < 0.25

    def test_escalation_wrong_zero(self):
        """Wrong escalation gives 0 for that component."""
        bug = EASY_BUGS[0]  # gt_escalation = True
        action = {
            "priority": bug.gt_priority,
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": False,  # wrong
            "comment": "",
        }
        result = grade(action, bug)
        assert result.breakdown["escalation"] == 0.0

    def test_comment_bonus(self):
        """A relevant comment adds a small bonus without exceeding 1.0."""
        bug = EASY_BUGS[0]  # gt_labels includes "security"
        action = {
            "priority": bug.gt_priority,
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": "This security bug needs immediate patching on the auth endpoint.",
        }
        result = grade(action, bug)
        assert result.reward <= 1.0
        assert result.reward > 0.9

    def test_total_clipped_at_one(self):
        """Reward is always ≤ 1.0."""
        bug = EASY_BUGS[0]
        action = {
            "priority": bug.gt_priority,
            "labels": bug.gt_labels,
            "assigned_team": bug.gt_team,
            "needs_escalation": bug.gt_escalation,
            "comment": "security critical backend escalation necessary.",
        }
        result = grade(action, bug)
        assert result.reward <= 1.0


# ---------------------------------------------------------------------------
# Environment unit tests
# ---------------------------------------------------------------------------

class TestBugTriageEnvironment:

    def _make_env(self) -> BugTriageEnvironment:
        return BugTriageEnvironment()

    # --- reset ---

    def test_reset_easy_returns_observation(self):
        env = self._make_env()
        obs = env.reset(task_id="easy", seed=0)
        assert obs.bug_id != ""
        assert obs.title != ""
        assert obs.task_id == "easy"
        assert obs.done is False
        assert obs.step_count == 0

    def test_reset_medium_returns_observation(self):
        env = self._make_env()
        obs = env.reset(task_id="medium", seed=0)
        assert obs.task_id == "medium"

    def test_reset_hard_returns_observation(self):
        env = self._make_env()
        obs = env.reset(task_id="hard", seed=0)
        assert obs.task_id == "hard"

    def test_reset_invalid_task_raises(self):
        env = self._make_env()
        with pytest.raises(ValueError):
            env.reset(task_id="impossible")

    def test_reset_clears_previous_state(self):
        env = self._make_env()
        obs1 = env.reset(task_id="easy", seed=0)
        # Take a step
        action = TriageAction(
            priority="critical",
            labels=["security", "bug"],
            assigned_team="security",
            needs_escalation=True,
        )
        env.step(action)
        # Reset should clear state
        obs2 = env.reset(task_id="easy", seed=0)
        assert obs2.step_count == 0
        assert obs2.done is False

    # --- step ---

    def test_step_easy_one_step_done(self):
        env = self._make_env()
        env.reset(task_id="easy", seed=0)
        action = TriageAction(
            priority="critical",
            labels=["security", "bug"],
            assigned_team="security",
            needs_escalation=True,
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.step_count == 1
        assert 0.0 <= obs.reward <= 1.0

    def test_step_medium_three_steps_done(self):
        env = self._make_env()
        env.reset(task_id="medium", seed=0)
        action = TriageAction(
            priority="high",
            labels=["bug"],
            assigned_team="backend",
            needs_escalation=False,
        )
        obs = None
        for i in range(3):
            obs = env.step(action)
        assert obs.done is True
        assert obs.step_count == 3

    def test_step_after_done_stays_done(self):
        env = self._make_env()
        env.reset(task_id="easy", seed=0)
        action = TriageAction(
            priority="critical",
            labels=["security"],
            assigned_team="security",
            needs_escalation=True,
        )
        env.step(action)
        obs2 = env.step(action)  # second step after done
        assert obs2.done is True

    def test_step_reward_in_range(self):
        env = self._make_env()
        env.reset(task_id="easy", seed=0)
        action = TriageAction(
            priority="low",
            labels=["documentation"],
            assigned_team="data",
            needs_escalation=False,
        )
        obs = env.step(action)
        assert 0.0 <= obs.reward <= 1.0

    # --- state ---

    def test_state_episode_id_set_after_reset(self):
        env = self._make_env()
        env.reset(task_id="easy", seed=0)
        state = env.state()
        assert state.episode_id != ""
        assert state.task_id == "easy"

    def test_state_step_count_increments(self):
        env = self._make_env()
        env.reset(task_id="medium", seed=0)
        action = TriageAction(
            priority="high", labels=["bug"], assigned_team="backend", needs_escalation=False
        )
        env.step(action)
        state = env.state()
        assert state.step_count == 1

    # --- reproducibility ---

    def test_same_seed_same_bug_sequence(self):
        env1 = self._make_env()
        env2 = self._make_env()
        obs1 = env1.reset(task_id="hard", seed=99)
        obs2 = env2.reset(task_id="hard", seed=99)
        assert obs1.bug_id == obs2.bug_id

    def test_deterministic_default_episodes(self):
        """No seed → always first N bugs in corpus order."""
        env = self._make_env()
        obs_a = env.reset(task_id="easy")
        env2 = self._make_env()
        obs_b = env2.reset(task_id="easy")
        assert obs_a.bug_id == obs_b.bug_id

    # --- corpus completeness ---

    def test_corpus_sizes(self):
        assert len(EASY_BUGS) >= 1
        assert len(MEDIUM_BUGS) >= 3
        assert len(HARD_BUGS) >= 3

    def test_all_bugs_have_required_fields(self):
        from server.bug_corpus import ALL_BUGS
        for bug in ALL_BUGS:
            assert bug.bug_id.startswith("BUG-")
            assert bug.title
            assert bug.description
            assert bug.gt_priority in ("critical", "high", "medium", "low")
            assert len(bug.gt_labels) >= 1
            assert bug.gt_team in ("backend", "frontend", "infra", "security", "mobile", "data")
            assert isinstance(bug.gt_escalation, bool)
