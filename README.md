# Bug Triage Environment 🐛

> **OpenEnv** | Real-world software engineering | `easy → medium → hard`

An agentic execution environment where AI agents learn to **triage bug reports** — exactly the way experienced engineers do every day. Agents receive GitHub/Jira-style bug reports (complete with stack traces, environment info, and reproduction steps) and must make a structured decision covering **priority**, **labels**, **team assignment**, and **escalation**.

This is the bug triage task that every mid-size engineering team does daily: "Is this a P0 outage or a P3 cosmetic issue? Does security need to be paged? Who owns this — backend, infra, or mobile?"

---

## Environment Description

| Property | Value |
|---|---|
| **Domain** | Software Engineering / DevOps |
| **Real-world analogue** | GitHub Issues / Jira / PagerDuty triage workflows |
| **Episode length** | 1 step (easy), 3 steps (medium/hard) |
| **Reward range** | 0.0 – 1.0 per step (mean over episode) |
| **Action type** | Structured JSON triage decision |
| **Observation type** | Bug report (text + metadata) |

### Why This Domain?

Bug triage is:
- **Universal**: every software company does it
- **High-value**: poor triage delays fixes and causes SLA breaches
- **Text-heavy**: ideal for large language model training/evaluation
- **Novel in OpenEnv**: no existing triage environment exists

---

## Action Space

```python
class TriageAction(Action):
    priority:         Literal["critical", "high", "medium", "low"]
    labels:           list[str]  # 1–4 from fixed vocabulary
    assigned_team:    Literal["backend", "frontend", "infra", "security", "mobile", "data"]
    needs_escalation: bool
    comment:          Optional[str]  # max 500 chars, used for partial credit
```

**Valid labels:** `bug`, `regression`, `security`, `performance`, `ux`, `data-loss`, `crash`, `memory-leak`, `api`, `documentation`, `flaky-test`, `needs-repro`

---

## Observation Space

```python
class TriageObservation(Observation):
    # Bug report fields (what the agent reads)
    bug_id:       str
    title:        str
    description:  str
    reporter:     str
    created_at:   str              # ISO-8601
    environment:  dict[str, str]   # OS, browser, version, etc.
    attachments:  list[str]        # log snippets / stack traces

    # Episode metadata
    task_id:    str    # "easy" | "medium" | "hard"
    step_count: int
    reward:     float  # reward from the last action
    done:       bool
    feedback:   str    # grader feedback on the last action
```

---

## Reward Function

Reward is computed deterministically per step and decomposes as:

| Component | Weight | Notes |
|---|---|---|
| **Priority** | 0.35 | Full credit if exact; 0.15 if off by one level |
| **Team** | 0.25 | Binary — team is either right or wrong |
| **Labels (F1)** | 0.25 | Proportional F1 against the ground-truth set |
| **Escalation** | 0.15 | Binary |
| **Comment bonus** | ≤ 0.05 | If comment mentions a key term from the ground-truth |

This design gives **continuous signal** across the full trajectory: an agent that gets priority right but assigns the wrong team still earns ~0.60, which is much more informative than a sparse 0/1 reward.

---

## Tasks

### 🟢 Easy — `task_id="easy"`

**1 bug, 1 step** · Expected score: ~0.85 (gpt-4o-mini baseline)

A single clearly-described bug with unambiguous signals:

- **BUG-0001** — SQL injection in `/api/v1/login` with a working PoC exploit. Clearly `critical`, `security` team, must escalate.

Difficulty markers: explicit CVE mention, stack trace shows auth bypass, reporter is `security_bot`.

---

### 🟡 Medium — `task_id="medium"`

**3 bugs, 3 steps** · Expected score: ~0.72 (gpt-4o-mini baseline)

Three bugs requiring careful log analysis and cross-team reasoning:

- **BUG-0010** — CSV export silently truncates at 50 k rows. Hidden `LIMIT` in source. `data-loss` label matters.
- **BUG-0011** — Android push notifications fail in battery-saver mode. Platform-specific; `mobile` team, not `backend`.
- **BUG-0012** — Recommendation model serves stale embeddings. Redis cache not invalidated after retraining. `data` team.

---

### 🔴 Hard — `task_id="hard"`

**3 bugs, 3 steps** · Expected score: ~0.55 (gpt-4o-mini baseline)

Three ambiguous bugs with conflicting signals and sparse evidence:

- **BUG-0020** — Intermittent 0.3% checkout failures. Deadlock sometimes appears in logs, but not always. Revenue impact → escalate. Hard to distinguish from random noise.
- **BUG-0021** — User claims their data disappeared after a profile update. DB row intact. Could be real stale-cache issue or user confusion → `needs-repro`, **not** escalate.
- **BUG-0022** — Dashboard 4× slower, no code changes. Suspicion: infra region migration or un-minified CDN assets. `infra` team, not `frontend`.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- `pip install openenv-core`

### Local Development (no Docker)

```bash
# 1. Clone / copy the environment
cd bug_triage_env

# 2. Install
pip install -e ".[dev]"

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Run tests
pytest tests/ -v

# 5. Run baseline (dry-run, no API key needed)
python baseline.py --dry-run

# 6. Run baseline with real LLM
export OPENAI_API_KEY="sk-..."
python baseline.py --model gpt-4o-mini
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t bug-triage-env .

# Run
docker run -p 8000:8000 bug-triage-env

# Verify
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Using the Python Client

```python
import asyncio
from bug_triage_env import BugTriageEnv, TriageAction

async def main():
    async with BugTriageEnv(base_url="http://localhost:8000") as env:
        # Start an easy episode
        result = await env.reset(task_id="easy")
        obs = result.observation
        print(f"Bug: {obs.title}")

        # Make a triage decision
        action = TriageAction(
            priority="critical",
            labels=["security", "bug"],
            assigned_team="security",
            needs_escalation=True,
            comment="SQL injection allows full auth bypass — immediate patch needed.",
        )
        result = await env.step(action)
        print(f"Reward: {result.reward:.3f}")
        print(f"Feedback: {result.observation.feedback}")

asyncio.run(main())
```

### Synchronous Usage

```python
from bug_triage_env import BugTriageEnv, TriageAction

with BugTriageEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="medium").observation
    while not obs.done:
        action = TriageAction(
            priority="high",
            labels=["bug", "regression"],
            assigned_team="backend",
            needs_escalation=False,
        )
        result = env.step(action)
        obs = result.observation
        print(f"Step reward: {result.reward:.3f}")
```

---

## Baseline Scores

Measured with `gpt-4o-mini`, temperature=0, seed=42:

| Task | Score | Notes |
|---|---|---|
| easy | ~0.85 | Near-perfect on unambiguous security bug |
| medium | ~0.72 | Occasional team misassignment (mobile vs backend) |
| hard | ~0.55 | Struggles with escalation judgment + ambiguous teams |

Run yourself:
```bash
export OPENAI_API_KEY="sk-..."
python baseline.py --model gpt-4o-mini --seed 42
```

---

## Project Structure

```
bug_triage_env/
├── __init__.py              # Public API re-exports
├── models.py                # TriageAction, TriageObservation (Pydantic)
├── client.py                # BugTriageEnv (EnvClient subclass)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies & entry-points
├── baseline.py              # Baseline inference script (OpenAI API)
├── .dockerignore
├── tests/
│   └── test_environment.py  # pytest suite (grader + environment)
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application (create_app)
    ├── bug_triage_environment.py  # Environment(reset/step/state)
    ├── bug_corpus.py        # Bug report dataset + ground truth
    ├── grader.py            # Deterministic grader (partial credit)
    ├── requirements.txt     # Docker server deps
    └── Dockerfile           # Multi-stage Docker build
```

---

## Contributing & Extending

To add more bug reports, add `BugRecord` entries to `server/bug_corpus.py`. Ground truth fields are required; the grader uses them deterministically.

To add a new task tier (e.g. `"expert"`), add entries to `BUGS_BY_TASK` and `TASK_BUG_COUNTS` in `bug_triage_environment.py`.

---

## License

MIT — see [LICENSE](LICENSE)
