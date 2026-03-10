from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from democall_analysis_ui.gemini_audit import AuditResult, generate_audit_from_transcript
from democall_analysis_ui.gemini_transcribe import (
    TranscribeResult,
    transcribe_audio_file,
)
from democall_analysis_ui.io_utils import sha256_bytes, sha256_text


# Allow importing repo-root modules (prompt.py) when running as a script or via Streamlit.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from prompt import CallAudit, prompt as AUDIT_PROMPT, transcript_prompt as TRANSCRIPT_PROMPT  # noqa: E402


@dataclass(frozen=True)
class RunMetadata:
    transcribe_model: Optional[str]
    audit_model: str
    runtime_s: float
    status: str  # "success" | "failure"


@dataclass(frozen=True)
class AnalysisResult:
    transcript_text: str
    audit_raw_text: str
    audit_validated: Optional[dict]
    validation_error: Optional[str]
    metadata: RunMetadata
    cache_key: str


class CallAuditPipeline:
    """
    Orchestrator for: audio -> transcript -> audit JSON -> Pydantic validation.
    """

    @staticmethod
    def build_client() -> genai.Client:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
        api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_GEMINI_API_KEY (set it in env or democall_analysis_ui/.env)")
        return genai.Client(api_key=api_key)

    @staticmethod
    def get_models() -> tuple[str, str]:
        # Defaults chosen to be widely available across Gemini API versions.
        # You can override these via GEMINI_TRANSCRIBE_MODEL / GEMINI_AUDIT_MODEL in .env.
        transcribe_model = os.environ.get(
            "GEMINI_TRANSCRIBE_MODEL",
            "models/gemini-1.0-pro",
        )
        audit_model = os.environ.get(
            "GEMINI_AUDIT_MODEL",
            "models/gemini-1.0-pro",
        )
        return transcribe_model, audit_model

    @staticmethod
    def run(
        *,
        client: genai.Client,
        audio_bytes: Optional[bytes],
        audio_filename: Optional[str],
        pasted_transcript: Optional[str]=None,
        force_sample_transcript: Optional[str] = None,
    ) -> AnalysisResult:
        started = time.perf_counter()
        transcribe_model, audit_model = CallAuditPipeline.get_models()

        transcript_text, used_transcribe_model, cache_key = CallAuditPipeline._get_transcript_and_cache_key(
            client=client,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            pasted_transcript=pasted_transcript,
            sample_transcript=force_sample_transcript,
            transcribe_model=transcribe_model,
        )

        audit_result = generate_audit_from_transcript(
            client=client,
            transcript_text=transcript_text,
            audit_prompt=AUDIT_PROMPT,
            model=audit_model,
        )

        validated, validation_error = CallAuditPipeline._validate_audit(audit_result)
        runtime_s = time.perf_counter() - started

        status = "success" if validated is not None else "failure"
        metadata = RunMetadata(
            transcribe_model=used_transcribe_model,
            audit_model=audit_model,
            runtime_s=runtime_s,
            status=status,
        )
        return AnalysisResult(
            transcript_text=transcript_text,
            audit_raw_text=audit_result.raw_text,
            audit_validated=validated,
            validation_error=validation_error,
            metadata=metadata,
            cache_key=cache_key,
        )

    @staticmethod
    def _get_transcript_and_cache_key(
        *,
        client: genai.Client,
        audio_bytes: Optional[bytes],
        audio_filename: Optional[str],
        pasted_transcript: Optional[str],
        sample_transcript: Optional[str],
        transcribe_model: str,
    ) -> tuple[str, Optional[str], str]:
        if sample_transcript and sample_transcript.strip():
            t = sample_transcript.strip()
            return t, None, f"sample:{sha256_text(t)}"

        if pasted_transcript and pasted_transcript.strip():
            t = pasted_transcript.strip()
            return t, None, f"paste:{sha256_text(t)}"

        if not audio_bytes:
            raise ValueError("Provide an audio file or paste a transcript.")

        audio_hash = sha256_bytes(audio_bytes)
        suffix = CallAuditPipeline._infer_suffix(audio_filename)
        transcript = CallAuditPipeline._transcribe_audio_bytes(
            client=client,
            audio_bytes=audio_bytes,
            suffix=suffix,
            transcribe_model=transcribe_model,
        )
        return transcript, transcribe_model, f"audio:{audio_hash}:{transcribe_model}"

    @staticmethod
    def _infer_suffix(audio_filename: Optional[str]) -> str:
        if not audio_filename:
            return ".bin"
        _, ext = os.path.splitext(audio_filename.lower())
        return ext if ext else ".bin"

    @staticmethod
    def _transcribe_audio_bytes(
        *,
        client: genai.Client,
        audio_bytes: bytes,
        suffix: str,
        transcribe_model: str,
    ) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            r: TranscribeResult = transcribe_audio_file(
                client=client,
                audio_path=tmp_path,
                transcript_prompt=TRANSCRIPT_PROMPT,
                model=transcribe_model,
            )
            if not r.transcript_text:
                raise RuntimeError("Empty transcript returned by model")
            return r.transcript_text
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _validate_audit(audit_result: AuditResult) -> tuple[Optional[dict], Optional[str]]:
        if audit_result.parse_error:
            return None, f"Audit JSON parse error: {audit_result.parse_error}"
        if not audit_result.audit_dict:
            return None, "Audit JSON missing or empty"

        try:
            model = CallAudit.model_validate(audit_result.audit_dict)
            return model.model_dump(mode="json"), None
        except ValidationError as e:
            return None, str(e)

