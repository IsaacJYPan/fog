"""Visible Pokemon state shared by the modern parser and local simulator adapters."""
from typing import Any


def is_terastallized(mon: Any) -> bool:
    """Read the parser's public activation flag, with compatibility for engine adapters."""
    return bool(getattr(mon, "is_terastallized", getattr(mon, "terastallized", False)))
