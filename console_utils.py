from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any


NAN = "NaN"


def format_console_value(value: Any) -> str:
    if value is None:
        return NAN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return NAN if not math.isfinite(numeric) else f"{numeric:.6g}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return str(value)


def format_console_line(algorithm: str, event: str, **fields: Any) -> str:
    parts = [f"{key}={format_console_value(value)}" for key, value in fields.items()]
    if parts:
        return f"[{algorithm}] {event} | " + " | ".join(parts)
    return f"[{algorithm}] {event}"


def format_console_record(algorithm: str, event: str, *, blank_after: int = 0, **fields: Any) -> str:
    return format_console_line(algorithm, event, **fields) + "\n" + ("\n" * max(0, int(blank_after)))


def emit_console(algorithm: str, event: str, *, blank_after: int = 0, **fields: Any) -> None:
    print(format_console_line(algorithm, event, **fields), flush=True)
    for _ in range(max(0, int(blank_after))):
        print("", flush=True)
