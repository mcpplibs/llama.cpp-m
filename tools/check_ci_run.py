#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys


def require_latest_success(payload: dict, sha: str) -> int:
    runs = [
        run
        for run in payload.get("workflow_runs", [])
        if run.get("head_sha") == sha and isinstance(run.get("id"), int)
    ]
    if not runs:
        raise ValueError(f"no CI workflow run found for {sha}")

    latest = max(runs, key=lambda run: run["id"])
    status = str(latest.get("status")).lower()
    conclusion = str(latest.get("conclusion")).lower()
    if status != "completed" or conclusion != "success":
        raise ValueError(
            f"latest CI run {latest['id']} is {status}/{conclusion}"
        )
    return latest["id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Require the latest CI workflow run for a commit to succeed"
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.runs.read_text(encoding="utf-8"))
        run_id = require_latest_success(payload, args.sha)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"CI run {run_id} succeeded for {args.sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
