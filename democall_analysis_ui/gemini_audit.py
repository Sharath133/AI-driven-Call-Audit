from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google import genai

from democall_analysis_ui.io_utils import JsonParseResult, safe_json_parse


@dataclass(frozen=True)
class AuditResult:
    audit_dict: Optional[dict]
    raw_text: str
    model: str
    parse_error: Optional[str] = None


def generate_audit_from_transcript(
    *,
    client: genai.Client,
    transcript_text: str,
    audit_prompt: str,
    model: str,
) -> AuditResult:
    """
    Request strict JSON and parse safely.
    """
    contents = [
        audit_prompt,
        f"TRANSCRIPT (English, may include timestamps/speakers):\n{transcript_text}",
    ]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config={"response_mime_type": "application/json"},
    )

    raw = (resp.text or "").strip()
    parsed: JsonParseResult = safe_json_parse(raw)
    return AuditResult(
        audit_dict=parsed.value,
        raw_text=parsed.raw_text,
        model=model,
        parse_error=parsed.error,
    )

