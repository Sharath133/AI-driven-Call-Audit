from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google import genai


@dataclass(frozen=True)
class TranscribeResult:
    transcript_text: str
    model: str


def transcribe_audio_file(
    *,
    client: genai.Client,
    audio_path: str,
    transcript_prompt: str,
    model: str,
) -> TranscribeResult:
    """
    Transcribe audio via Gemini. Uses file upload for broad format support.
    """
    uploaded = client.files.upload(file=audio_path)
    resp = client.models.generate_content(
        model=model,
        contents=[transcript_prompt, uploaded],
    )
    return TranscribeResult(transcript_text=(resp.text or "").strip(), model=model)

