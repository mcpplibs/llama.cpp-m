#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import sys
import time
from typing import NamedTuple
import urllib.error
import urllib.request

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.import_upstream import UpstreamLock, _github_repository, load_lock


POLICY_TRIGGERS = (
    "security-correctness",
    "model-quantization",
    "supported-backend",
    "consumer-api",
    "toolchain-platform",
)


class UpstreamStatus(NamedTuple):
    current_tag: str
    current_commit: str
    latest_tag: str
    latest_commit: str
    age_days: int
    candidate_reason: str


def _is_transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError),
    )


def github_json(url: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "llama.cpp-m-upstream-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except Exception as error:
            if attempt == 2 or not _is_transient(error):
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _parse_github_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GitHub timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def collect_status(
    lock: UpstreamLock,
    *,
    now: datetime | None = None,
    triggers: tuple[str, ...] = (),
    release_gate_passed: bool = False,
) -> UpstreamStatus:
    invalid = sorted(set(triggers) - set(POLICY_TRIGGERS))
    if invalid:
        raise ValueError(f"unsupported candidate trigger: {', '.join(invalid)}")

    owner, repository = _github_repository(lock.repository)
    api = f"https://api.github.com/repos/{owner}/{repository}"
    tags = github_json(f"{api}/tags?per_page=1")
    if not isinstance(tags, list) or not tags:
        raise ValueError("GitHub returned no upstream tags")
    latest = tags[0]
    latest_tag = latest["name"]
    latest_commit = latest["commit"]["sha"]

    commit = github_json(f"{api}/commits/{lock.commit}")
    committed_at = _parse_github_time(commit["commit"]["committer"]["date"])
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = max(0, (current_time - committed_at).days)

    unique_triggers = tuple(dict.fromkeys(triggers))
    if unique_triggers:
        reason = ",".join(unique_triggers)
    elif (
        latest_commit != lock.commit
        and age_days >= 30
        and release_gate_passed
    ):
        reason = "checkpoint-age"
    else:
        reason = "none"

    return UpstreamStatus(
        current_tag=lock.tag,
        current_commit=lock.commit,
        latest_tag=latest_tag,
        latest_commit=latest_commit,
        age_days=age_days,
        candidate_reason=reason,
    )


def render_report(status: UpstreamStatus) -> str:
    return "\n".join(
        (
            "# llama.cpp upstream checkpoint report",
            "",
            f"- Current: `{status.current_tag}` (`{status.current_commit}`)",
            f"- Latest: `{status.latest_tag}` (`{status.latest_commit}`)",
            f"- Current checkpoint age: {status.age_days} days",
            f"- Candidate reason: `{status.candidate_reason}`",
            "",
            "Commit count alone is informational and never triggers a release.",
        )
    )


def append_github_output(path: Path, status: UpstreamStatus) -> None:
    values = status._asdict()
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report whether the pinned llama.cpp checkpoint needs review"
    )
    parser.add_argument("--lock", type=Path, default=Path("upstream.lock"))
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--trigger", action="append", choices=POLICY_TRIGGERS, default=[])
    parser.add_argument("--release-gate-passed", action="store_true")
    args = parser.parse_args(argv)

    status = collect_status(
        load_lock(args.lock),
        triggers=tuple(args.trigger),
        release_gate_passed=args.release_gate_passed,
    )
    print(render_report(status))
    if args.github_output:
        append_github_output(args.github_output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
