"""
FastAPI application for the Bug Triage Environment.

Endpoints exposed by openenv.core.env_server.http_server.create_app:
  POST /reset          → TriageObservation
  POST /step           → TriageObservation
  GET  /state          → TriageState
  GET  /health         → {"status": "ok"}
  GET  /web            → (optional web UI, enabled via ENABLE_WEB_INTERFACE=true)

Usage
-----
  # Development (with auto-reload):
  uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

  # Production:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

  # Docker:
  docker build -t bug-triage-env . && docker run -p 8000:8000 bug-triage-env
"""

import sys
import os

# Adjust PYTHONPATH for standalone runs (so `models.py` resolves correctly)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app

from models import TriageAction, TriageObservation
from server.bug_triage_environment import BugTriageEnvironment

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug_triage_env",
)


def main() -> None:
    """Entry point for direct execution or `uv run server`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
