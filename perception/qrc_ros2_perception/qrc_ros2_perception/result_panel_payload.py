"""Payload helpers for result panel ROS callbacks."""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(data: str) -> dict[str, Any] | None:
    """Parse a JSON object without raising into ROS callbacks."""

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
