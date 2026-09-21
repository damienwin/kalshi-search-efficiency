from datetime import datetime, timezone


def from_iso(s: str) -> datetime:
    """Parse an ISO-8601 string (Z or offset) to an aware UTC datetime (Spring src/utils/time.py)."""
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
