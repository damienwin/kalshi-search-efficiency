"""Hash manifest for raw data (leakage test L13).

Every raw file written by a pull is recorded as one JSON line in
data/manifests/<name>.jsonl: path, sha256, bytes, endpoint, params, fetched_at.
Manifests are committed; raw files are not. `verify` fails if any recorded
file is missing or its hash changed.

    python -m ksearch.data.manifest verify [data/manifests/*.jsonl]
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, file_path: str, endpoint: str, params: dict) -> dict:
        entry = {
            "path": os.path.relpath(file_path, REPO_ROOT),
            "sha256": sha256_file(file_path),
            "bytes": os.path.getsize(file_path),
            "endpoint": endpoint,
            "params": params,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def recorded_paths(self) -> set:
        return {e["path"] for e in self.entries()}


def verify(manifest_paths: list[str]) -> list[str]:
    """Return a list of problems; empty means every recorded file is intact.

    Each path is checked against its most recent record across all manifests.
    """
    latest = {}
    for mp in manifest_paths:
        for e in Manifest(mp).entries():
            prev = latest.get(e["path"])
            if prev is None or e["fetched_at"] >= prev["fetched_at"]:
                latest[e["path"]] = e  # a re-fetch, in any manifest, supersedes the earlier record
    problems = []
    for rel, e in sorted(latest.items()):
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(full):
            problems.append(f"missing: {rel}")
        elif sha256_file(full) != e["sha256"]:
            problems.append(f"hash mismatch: {rel}")
    return problems


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "verify":
        print(__doc__)
        return 2
    paths = argv[1:] or sorted(glob.glob(os.path.join(REPO_ROOT, "data", "manifests", "*.jsonl")))
    problems = verify(paths)
    n = sum(len(Manifest(p).entries()) for p in paths)
    for p in problems:
        print(p)
    print(f"manifest verify: {n} records in {len(paths)} manifest(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
