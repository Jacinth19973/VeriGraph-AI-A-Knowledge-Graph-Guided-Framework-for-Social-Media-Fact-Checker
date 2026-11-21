"""
VeriGraph AI - JSON serialization utilities.

Converts NumPy / pandas / Python objects into JSON-safe
native Python types before persistence or FastAPI responses.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from enum import Enum
from typing import Any

import numpy as np


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert arbitrary pipeline values into
    JSON-compatible Python values.

    Handles:
    - numpy integer -> int
    - numpy floating -> float
    - numpy boolean -> bool
    - numpy ndarray -> list
    - numpy scalar -> Python scalar
    - datetime/date -> ISO string
    - Path -> string
    - Enum -> enum value
    - dict -> recursively converted dict
    - list/tuple -> recursively converted list
    - set -> recursively converted sorted/list representation
    """

    # -----------------------------
    # None / primitive Python types
    # -----------------------------
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    # -----------------------------
    # NumPy scalar types
    # -----------------------------
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # -----------------------------
    # NumPy arrays
    # -----------------------------
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    # -----------------------------
    # NumPy generic scalar
    # -----------------------------
    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    # -----------------------------
    # datetime/date
    # -----------------------------
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # -----------------------------
    # Path
    # -----------------------------
    if isinstance(value, Path):
        return str(value)

    # -----------------------------
    # Enum
    # -----------------------------
    if isinstance(value, Enum):
        return make_json_safe(value.value)

    # -----------------------------
    # Dictionary
    # -----------------------------
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    # -----------------------------
    # List / tuple
    # -----------------------------
    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # -----------------------------
    # Set / frozenset
    # -----------------------------
    if isinstance(value, (set, frozenset)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # -----------------------------
    # pandas-like objects
    # -----------------------------
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().isoformat()
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass

    # -----------------------------
    # Last-resort conversion
    # -----------------------------
    return str(value)