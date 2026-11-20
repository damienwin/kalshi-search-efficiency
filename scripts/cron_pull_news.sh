#!/bin/bash
# Daily resumable Guardian pull for the v2 dataset. Safe to re-run: stops
# cleanly at the daily quota, skips pages already cached on disk.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
exec .venv/bin/python scripts/pull_news.py --build-dir data/build/v2
