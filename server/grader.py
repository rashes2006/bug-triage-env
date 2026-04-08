"""
Programmatic graders for the Bug Triage Environment.

Each grader takes a TriageAction and a BugRecord and returns a score [0.0, 1.0]
plus a list of human-readable feedback strings.

Scoring breakdown per bug (totalling 1.0):
  - priority     : 0.35  (0 or 0.35 if exact; 0.15 if off by one level)
  - team         : 0.25  (0 or 0.25)
  - labels       : 0.25  (proportional F1 score against ground truth)
  - escalation   : 0.15  (0 or 0.15)

Episodes are single-step (one triage decision per bug), but multi-step
episodes accumulate rewards across a set of bugs (medium→3 bugs, hard→3 bugs).
"""

from __future__ import annotations

from dataclasses import dataclass

from server.bug_corpus import BugRecord

# Priority ordering for partial credit
PRIORITY_ORDER = ["critical", "high", "medium", "low"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _priority_score(predicted: str, ground_truth: str) -> tuple[float, str]:
    """Return (score, feedback) for the priority field."""
    if predicted == ground_truth:
        return 0.35, f"✅ Priority correct: '{predicted}'"
    pred_idx = PRIORITY_ORDER.index(predicted)
    gt_idx = PRIORITY_ORDER.index(ground_truth)
    if abs(pred_idx - gt_idx) == 1:
        return 0.15, (
            f"⚠️  Priority off by one level: got '{predicted}', expected '{ground_truth}'"
        )
    return 0.0, f"❌ Priority wrong: got '{predicted}', expected '{ground_truth}'"


def _team_score(predicted: str, ground_truth: str) -> tuple[float, str]:
    """Return (score, feedback) for the assigned_team field."""
    if predicted == ground_truth:
        return 0.25, f"✅ Team correct: '{predicted}'"
    return 0.0, f"❌ Team wrong: got '{predicted}', expected '{ground_truth}'"


def _labels_score(predicted: list[str], ground_truth: list[str]) -> tuple[float, str]:
    """
    Return (score, feedback) for labels using F1 against the ground truth set.

    F1 = 2 * precision * recall / (precision + recall)
    """
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    if not pred_set and not gt_set:
        return 0.25, "✅ Labels correct (both empty)"

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    score = round(0.25 * f1, 4)
    missing = gt_set - pred_set
    extra = pred_set - gt_set
    parts = []
    if tp:
        parts.append(f"correct labels: {sorted(pred_set & gt_set)}")
    if missing:
        parts.append(f"missing: {sorted(missing)}")
    if extra:
        parts.append(f"extra: {sorted(extra)}")
    if f1 == 1.0:
        feedback = f"✅ Labels perfect: {sorted(pred_set)}"
    else:
        feedback = f"⚠️  Labels F1={f1:.2f} — " + ", ".join(parts)
    return score, feedback


def _escalation_score(predicted: bool, ground_truth: bool) -> tuple[float, str]:
    """Return (score, feedback) for the needs_escalation field."""
    if predicted == ground_truth:
        return 0.15, f"✅ Escalation correct: {predicted}"
    msg = (
        "❌ Should have escalated" if ground_truth else "❌ Should NOT have escalated"
    )
    return 0.0, msg


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    reward: float          # 0.0 – 1.0
    feedback: str          # human-readable summary
    breakdown: dict[str, float]   # component scores


def grade(action_dict: dict, bug: BugRecord) -> GraderResult:
    """
    Grade one triage decision against the ground truth for `bug`.

    Parameters
    ----------
    action_dict : dict
        The action as a plain dict (keys: priority, labels, assigned_team,
        needs_escalation, comment).
    bug : BugRecord
        The bug being triaged.

    Returns
    -------
    GraderResult
        reward in [0, 1], human feedback, and component breakdown.
    """
    feedback_lines: list[str] = []

    p_score, p_fb = _priority_score(action_dict["priority"], bug.gt_priority)
    t_score, t_fb = _team_score(action_dict["assigned_team"], bug.gt_team)
    l_score, l_fb = _labels_score(action_dict["labels"], bug.gt_labels)
    e_score, e_fb = _escalation_score(action_dict["needs_escalation"], bug.gt_escalation)

    feedback_lines.extend([p_fb, t_fb, l_fb, e_fb])

    total = round(p_score + t_score + l_score + e_score, 4)
    total = min(1.0, max(0.0, total))

    # Bonus: a well-justified comment that mentions at least one key term
    key_terms = set(bug.gt_labels) | {bug.gt_team, bug.gt_priority}
    comment = (action_dict.get("comment") or "").lower()
    if any(term in comment for term in key_terms) and len(comment) > 20:
        bonus = min(0.05, round(1.0 - total, 4))  # cap so we never exceed 1.0
        total = min(1.0, total + bonus)
        feedback_lines.append(f"🎁 Comment bonus +{bonus:.2f} for relevant justification")

    return GraderResult(
        reward=total,
        feedback="\n".join(feedback_lines),
        breakdown={
            "priority": p_score,
            "team": t_score,
            "labels": l_score,
            "escalation": e_score,
        },
    )
