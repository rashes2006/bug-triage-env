"""
Bug report corpus for the Bug Triage Environment.

Each record contains:
  - the raw bug report shown to the agent
  - the ground-truth triage decision (used by the grader)
  - the task_id it belongs to

Difficulty levels
-----------------
easy   – unambiguous, clear signals (explicit "crash on prod", obvious security, etc.)
medium – requires reading between the lines; e.g. partial log evidence, multi-team
hard   – ambiguous; conflicting signals, sparse info, triage depends on subtle context
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BugRecord:
    bug_id: str
    title: str
    description: str
    reporter: str
    created_at: str
    environment: dict[str, str]
    attachments: list[str]
    task_id: str  # "easy" | "medium" | "hard"

    # Ground-truth expected triage (used by grader)
    gt_priority: str
    gt_labels: list[str]         # set comparison (order-independent)
    gt_team: str
    gt_escalation: bool


# ---------------------------------------------------------------------------
# EASY BUGS  (task_id = "easy")
# ---------------------------------------------------------------------------

EASY_BUGS: list[BugRecord] = [
    BugRecord(
        bug_id="BUG-0001",
        title="SQL injection in /api/v1/login — auth bypass confirmed",
        description=(
            "A researcher reported that the `username` field in the login endpoint "
            "is not sanitised. Sending `' OR '1'='1` as username bypasses password "
            "check on all accounts. Tested on production. Any account on the platform "
            "can be compromised. CVE pending. Requires immediate patch."
        ),
        reporter="security_bot",
        created_at="2024-03-15T09:00:00Z",
        environment={"version": "v2.4.1", "db": "PostgreSQL 14"},
        attachments=[
            "POST /api/v1/login HTTP/1.1\n"
            "Content-Type: application/json\n"
            '{"username": "\' OR \'1\'=\'1", "password": "x"}\n'
            "HTTP/1.1 200 OK\n"
            '{"token": "eyJhbGci..."}'
        ],
        task_id="easy",
        gt_priority="critical",
        gt_labels=["security", "bug"],
        gt_team="security",
        gt_escalation=True,
    ),
    BugRecord(
        bug_id="BUG-0002",
        title="Home page crashes on Safari 17 — blank white screen",
        description=(
            "Users on Safari 17.x see a blank white screen immediately after login. "
            "The JavaScript console shows: `TypeError: Cannot read properties of undefined "
            "(reading 'map')`. This only affects the home dashboard; all other pages load fine. "
            "Reproducible 100% of the time on Safari 17. Chrome and Firefox are fine."
        ),
        reporter="qa_alice",
        created_at="2024-03-15T10:30:00Z",
        environment={"browser": "Safari 17.0", "os": "macOS Sonoma", "version": "v2.4.1"},
        attachments=[
            "TypeError: Cannot read properties of undefined (reading 'map')\n"
            "    at Dashboard.render (Dashboard.jsx:42)\n"
            "    at processChild (react-dom.development.js:3990)"
        ],
        task_id="easy",
        gt_priority="high",
        gt_labels=["bug", "crash"],
        gt_team="frontend",
        gt_escalation=False,
    ),
    BugRecord(
        bug_id="BUG-0003",
        title="Kubernetes pod OOM-kills every ~6 hours — API server downtime",
        description=(
            "The main API pod is being OOM-killed by the node approximately every 6 hours. "
            "After each kill there is a ~2-minute window of 503 errors. Memory usage climbs "
            "steadily from ~400 MB at startup to 2 GB before the kill. Heap dumps show "
            "retained EventEmitter listeners accumulating over time."
        ),
        reporter="sre_bob",
        created_at="2024-03-15T11:00:00Z",
        environment={"k8s": "1.29", "node": "Node 20 LTS", "version": "v2.4.0"},
        attachments=[
            "FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory\n"
            "OOMKiller invoked pid=7421 name=node"
        ],
        task_id="easy",
        gt_priority="high",
        gt_labels=["bug", "memory-leak"],
        gt_team="backend",
        gt_escalation=False,
    ),
]

# ---------------------------------------------------------------------------
# MEDIUM BUGS  (task_id = "medium")
# ---------------------------------------------------------------------------

MEDIUM_BUGS: list[BugRecord] = [
    BugRecord(
        bug_id="BUG-0010",
        title="Export to CSV silently drops rows when dataset > 50 k records",
        description=(
            "Users exporting analytics data to CSV notice that large datasets are "
            "silently truncated. A dataset with 75,000 rows yields a CSV with exactly "
            "50,000 rows — no error or warning is shown. The API returns HTTP 200. "
            "Discovered by a customer who noticed revenue figures didn't match. "
            "Smaller exports (<= 50 k) are fine."
        ),
        reporter="customer_success",
        created_at="2024-03-16T08:00:00Z",
        environment={"version": "v2.3.5", "db": "MySQL 8.0"},
        attachments=[
            "SELECT * FROM analytics LIMIT 50000  -- hardcoded in export_service.py:87"
        ],
        task_id="medium",
        gt_priority="high",
        gt_labels=["bug", "data-loss", "api"],
        gt_team="backend",
        gt_escalation=False,
    ),
    BugRecord(
        bug_id="BUG-0011",
        title="Push notifications not delivered on Android 14 in battery-saver mode",
        description=(
            "Several enterprise customers report their Android 14 devices on aggressive "
            "battery-saver profiles stop receiving push notifications after ~30 minutes "
            "of inactivity. iOS and older Android versions are unaffected. The FCM "
            "dashboard shows messages as 'delivered to device' but the app never surfaces "
            "them. Doze mode restrictions are suspected."
        ),
        reporter="enterprise_support_carol",
        created_at="2024-03-16T09:30:00Z",
        environment={"platform": "Android 14", "fcm_sdk": "23.4.0", "version": "v2.4.1"},
        attachments=[
            "W/FirebaseMessaging: Missing FOREGROUND_SERVICE_MANIFEST_PERMISSION for exact alarms."
        ],
        task_id="medium",
        gt_priority="medium",
        gt_labels=["bug", "ux"],
        gt_team="mobile",
        gt_escalation=False,
    ),
    BugRecord(
        bug_id="BUG-0012",
        title="Recommendation model returns stale embeddings after daily retraining",
        description=(
            "Since last week's model pipeline migration, the recommendation service "
            "appears to keep serving the previous day's embeddings even after the 03:00 "
            "retraining job completes. The feature store shows new embeddings are written, "
            "but REDIS cache is not invalidated. Users see identical recommendations all day. "
            "The retraining job exits 0 so alerting doesn't fire."
        ),
        reporter="ml_dave",
        created_at="2024-03-16T14:00:00Z",
        environment={"version": "v2.4.0", "redis": "7.2", "python": "3.11"},
        attachments=[
            "INFO  retrain_job: model saved to s3://models/rec_v2/2024-03-16/\n"
            "INFO  retrain_job: exit 0\n"
            "ERROR recommendation_server: cache TTL=86400, no invalidation hook called"
        ],
        task_id="medium",
        gt_priority="medium",
        gt_labels=["bug", "performance"],
        gt_team="data",
        gt_escalation=False,
    ),
]

# ---------------------------------------------------------------------------
# HARD BUGS  (task_id = "hard")
# ---------------------------------------------------------------------------

HARD_BUGS: list[BugRecord] = [
    BugRecord(
        bug_id="BUG-0020",
        title="Intermittent 500 errors on checkout — affects ~0.3% of transactions",
        description=(
            "Approximately 0.3% of checkout requests fail with a 500 error. The pattern "
            "seems correlated with high cart sizes (>15 items) but not consistently. "
            "Errors appear in bursts—sometimes 20 in a minute, then nothing for hours. "
            "The payment processor has confirmed no issues on their side. Logs show a "
            "Postgres deadlock occasionally but not on every 500. This has been happening "
            "since the new inventory micro-service was deployed two weeks ago, though that "
            "team says their service is healthy. Revenue impact is estimated at ~$2k/day."
        ),
        reporter="on_call_erin",
        created_at="2024-03-17T07:00:00Z",
        environment={"version": "v2.4.1", "db": "PostgreSQL 15", "services": "checkout,inventory"},
        attachments=[
            "ERROR checkout_service: pq: deadlock detected\n"
            "DETAIL: Process 34512 waits for ShareLock on transaction 9876543\n"
            "HINT: See server log for query details.\n"
            "[sometimes absent from logs]",
        ],
        task_id="hard",
        gt_priority="high",
        gt_labels=["bug", "performance", "regression"],
        gt_team="backend",
        gt_escalation=True,  # revenue impact > threshold
    ),
    BugRecord(
        bug_id="BUG-0021",
        title="User reports their account data was 'gone' after profile update",
        description=(
            "A single user (user_id=88412) contacted support claiming all their saved "
            "preferences and history have disappeared after updating their profile picture. "
            "Support cannot reproduce the issue on a test account. The user's DB row is "
            "intact. Audit log shows profile_update succeeded. The user may be confused "
            "about what data they expected. However, we did deploy a profile caching "
            "change yesterday — cache warm-up may have served stale state to this user."
        ),
        reporter="support_frank",
        created_at="2024-03-17T11:00:00Z",
        environment={"version": "v2.4.1", "cache": "Redis 7.2"},
        attachments=["audit_log: user=88412 action=profile_update status=200 ts=2024-03-17T10:47Z"],
        task_id="hard",
        gt_priority="medium",
        gt_labels=["bug", "needs-repro"],
        gt_team="backend",
        gt_escalation=False,
    ),
    BugRecord(
        bug_id="BUG-0022",
        title="Dashboard loads 4× slower than last month — no obvious code change",
        description=(
            "Multiple enterprise accounts report the main dashboard now takes 8–12 seconds "
            "to load, compared to 2–3 seconds previously. No relevant code was changed in "
            "the past 30 days according to git blame. DB query times reported by APM look "
            "normal (<100 ms). The slowdown seems to affect only accounts with > 10,000 "
            "active users. Suspicion: a recent infrastructure migration to new datacenter "
            "region (EU-West-2 → EU-Central-1) increased round-trip latency for these "
            "accounts. Alternatively, an un-tracked CDN config change may be serving "
            "un-minified assets."
        ),
        reporter="perf_grace",
        created_at="2024-03-17T14:00:00Z",
        environment={"version": "v2.4.0", "cdn": "CloudFront", "region": "EU-Central-1"},
        attachments=[
            "APM trace: total=9800ms  db=87ms  render=9400ms  network=310ms",
            "Asset sizes: main.js 4.2MB (un-minified?), main.css 1.1MB",
        ],
        task_id="hard",
        gt_priority="high",
        gt_labels=["performance", "regression"],
        gt_team="infra",
        gt_escalation=False,
    ),
]

# ---------------------------------------------------------------------------
# Combined index
# ---------------------------------------------------------------------------

ALL_BUGS: list[BugRecord] = EASY_BUGS + MEDIUM_BUGS + HARD_BUGS

BUGS_BY_TASK: dict[str, list[BugRecord]] = {
    "easy": EASY_BUGS,
    "medium": MEDIUM_BUGS,
    "hard": HARD_BUGS,
}
