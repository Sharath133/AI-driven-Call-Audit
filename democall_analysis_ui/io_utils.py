from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class JsonParseResult:
    value: Optional[dict]
    raw_text: str
    error: Optional[str] = None


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction if the model returns extra text around JSON.
    Finds the first top-level {...} region via brace balancing.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def safe_json_parse(text: str) -> JsonParseResult:
    raw = (text or "").strip()
    if not raw:
        return JsonParseResult(value=None, raw_text=raw, error="Empty model output")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return JsonParseResult(value=parsed, raw_text=raw)
        return JsonParseResult(
            value=None,
            raw_text=raw,
            error="Expected a JSON object at top-level",
        )
    except Exception as e:
        extracted = _extract_first_json_object(raw)
        if not extracted:
            return JsonParseResult(value=None, raw_text=raw, error=f"Invalid JSON: {e}")
        try:
            parsed2 = json.loads(extracted)
            if isinstance(parsed2, dict):
                return JsonParseResult(value=parsed2, raw_text=raw)
            return JsonParseResult(
                value=None,
                raw_text=raw,
                error="Extracted JSON was not an object",
            )
        except Exception as e2:
            return JsonParseResult(
                value=None,
                raw_text=raw,
                error=f"Invalid JSON (even after extraction): {e2}",
            )


def json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False, default=str)

