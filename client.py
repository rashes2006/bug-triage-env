"""
Bug Triage Environment Client.

Provides a strongly-typed client that wraps the HTTP server,
giving callers the standard OpenEnv async/sync API.

Example (async)
---------------
>>> import asyncio
>>> from bug_triage_env import BugTriageEnv, TriageAction
>>>
>>> async def main():
...     async with BugTriageEnv(base_url="http://localhost:8000") as env:
...         obs = await env.reset(task_id="easy")
...         print(obs.observation.title)
...
...         action = TriageAction(
...             priority="critical",
...             labels=["security", "bug"],
...             assigned_team="security",
...             needs_escalation=True,
...             comment="SQL injection on login endpoint — critical security issue.",
...         )
...         result = await env.step(action)
...         print(f"Reward: {result.reward}")
...
>>> asyncio.run(main())

Example (sync)
--------------
>>> from bug_triage_env import BugTriageEnv, TriageAction
>>>
>>> with BugTriageEnv(base_url="http://localhost:8000").sync() as env:
...     obs = env.reset(task_id="medium")
...     for _ in range(3):
...         if obs.observation.done:
...             break
...         action = TriageAction(
...             priority="high",
...             labels=["bug"],
...             assigned_team="backend",
...             needs_escalation=False,
...         )
...         obs = env.step(action)
...         print(obs.reward)
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import TriageAction, TriageObservation, TriageState


class BugTriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Client for the Bug Triage Environment.

    Inherits async ``reset()`` / ``step()`` / ``state()`` from ``EnvClient``.
    Call ``.sync()`` for a synchronous wrapper.
    """

    def _step_payload(self, action: TriageAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward", 0.0)
        obs = TriageObservation(**obs_data)
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> TriageState:
        return TriageState(**payload)
