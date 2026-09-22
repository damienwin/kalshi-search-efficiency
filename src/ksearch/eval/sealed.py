"""The only way to read the sealed test slice (L11).

Every read appends an entry to data/manifests/sealed_access.jsonl, which is
committed so the history of sealed access is public: when, why, by which script,
at which commit, and the hash of the file that was read. Nothing else in the
repo opens sealed/sealed.parquet; tests/test_eval.py enforces that.

Scripts that need to know *which* events exist (sentiment, news) use the
label-free events index written by build_dataset.py instead.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from ksearch.data.manifest import REPO_ROOT, sha256_file

SEALED_PATH = os.path.join(REPO_ROOT, "sealed", "sealed.parquet")
ACCESS_LOG = os.path.join(REPO_ROOT, "data", "manifests", "sealed_access.jsonl")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def access_log(path: str = ACCESS_LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def log_access(reason: str, path: str = SEALED_PATH, log_path: str = ACCESS_LOG, **extra) -> dict:
    if not reason or len(reason.strip()) < 10:
        raise ValueError("reading the sealed slice requires a written reason (>= 10 chars)")
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "reason": reason.strip(), "script": os.path.basename(sys.argv[0]),
             "commit": _git_commit(), "sha256": sha256_file(path), **extra}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_sealed(reason: str, path: str = SEALED_PATH, log_path: str = ACCESS_LOG, **extra) -> pd.DataFrame:
    """Log the access, then return the sealed events."""
    log_access(reason, path, log_path, **extra)
    return pd.read_parquet(path)
