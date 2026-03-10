from __future__ import annotations

import os
import sys
import time
from typing import Optional

import streamlit as st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from democall_analysis_ui.io_utils import json_pretty, sha256_bytes, sha256_text
from democall_analysis_ui.pipeline import CallAuditPipeline, AnalysisResult


SAMPLE_TRANSCRIPT = """[00:00] Speaker 1: Hello, I’m calling from the learning team. Am I speaking with the parent?
[00:03] Speaker 2: Yes, I’m his mother. He is in class 10.
[00:08] Speaker 1: Great. Is he preparing for JEE or NEET?
[00:11] Speaker 2: JEE. We already have local tuition but we’re exploring options.
[00:18] Speaker 1: Any concerns you want us to address in the demo?
[00:21] Speaker 2: Mainly fees and schedule. Also, does your content match CBSE?
[00:29] Speaker 1: Understood. Can we schedule a demo tomorrow evening?
[00:33] Speaker 2: Not sure, I’ll confirm after discussing with his father.
"""


def _page_setup() -> None:
    st.set_page_config(page_title="AI Call Audit Demo", page_icon="🎧", layout="wide")
    st.title("AI-driven Call Audit")
    st.caption(
        "Upload an audio recording to run an automated, structured call audit JSON workflow, "
        "validated against the production `CallAudit` schema."
    )


def _render_production_expander() -> None:
    with st.expander("How this works in production (automated flow)"):
        st.markdown(
            """
This demo illustrates the **core automated call-audit workflow + schema validation** only.

**Production flow (high level):**
- **DB fetch** (Mongo/Postgres): fetch recent call activities + recording URLs
- **Download recordings** to temp storage
- **Automated audit for calls**:
  - High-priority calls may use dedicated ASR / telephony pipelines
  - Lower-priority calls are transcribed via Gemini using the transcription prompt
- **Models used here**:
  - Transcript: `gemini-2.0-flash` (configurable)
  - Audit JSON: `gemini-2.5-pro` (configurable, strict JSON)
- **Audit JSON generation** via Gemini using the call-audit prompt (**strict JSON**)
- **Validate & normalize** with `CallAudit` (Pydantic)
- **Store structured insights** (Mongo audit collection) + run metadata
- **Push to LeadSquared** (map fields to LSQ attributes)
- **Observability** with Langfuse (traces, prompts, model versions, latency, failures)

**What this demo intentionally skips:**
- No MongoDB writes
- No LeadSquared API calls
- No internal `app.external.leadsquared_api` dependencies

It’s designed to be **safe to share** while still showing the end-to-end AI audit loop.
"""
        )


@st.cache_data(show_spinner=False)
def _cached_analysis(
    *,
    cache_key: str,
    audio_bytes: Optional[bytes],
    audio_filename: Optional[str],
) -> AnalysisResult:
    client = CallAuditPipeline.build_client()
    return CallAuditPipeline.run(
        client=client,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        force_sample_transcript=None,
    )


def main() -> None:
    _page_setup()

    with st.sidebar:
        st.subheader("Inputs")
        uploaded = st.file_uploader(
            "Upload audio",
            type=["mp3", "wav", "m4a"],
            help="Upload a call recording, then click Analyze.",
        )
        # Center the Analyze button under the uploader
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            analyze = st.button("Analyze", type="primary", use_container_width=True)

        st.divider()
        st.subheader("Run metadata")
        st.caption("Models can be overridden via env vars in `democall_analysis_ui/.env`.")
        st.code(
            "\n".join(
                [
                    f"GEMINI_TRANSCRIBE_MODEL={os.environ.get('GEMINI_TRANSCRIBE_MODEL', 'models/gemini-1.0-pro')}",
                    f"GEMINI_AUDIT_MODEL={os.environ.get('GEMINI_AUDIT_MODEL', 'models/gemini-1.0-pro')}",
                ]
            )
        )

    _render_production_expander()

    audio_bytes: Optional[bytes] = uploaded.getvalue() if uploaded else None
    audio_filename: Optional[str] = uploaded.name if uploaded else None

    if not analyze:
        st.info("Upload an audio file, then click **Analyze**.")
        return

    # Build a stable cache key (so reruns don’t re-bill the model).
    if audio_bytes:
        transcribe_model = os.environ.get("GEMINI_TRANSCRIBE_MODEL", "models/gemini-1.0-pro")
        cache_key = f"audio:{sha256_bytes(audio_bytes)}:{transcribe_model}"
    else:
        st.error("Please upload an audio file before running analysis.")
        return

    with st.spinner("Running analysis (transcript → audit → schema validation)…"):
        try:
            result = _cached_analysis(
                cache_key=cache_key,
                audio_bytes=audio_bytes,
                audio_filename=audio_filename,
            )
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

    meta = result.metadata
    st.success(f"Status: {meta.status} • Runtime: {meta.runtime_s:.2f}s")

    tabs = st.tabs(["Transcript", "Audit JSON"])

    with tabs[0]:
        st.text_area("Transcript", value=result.transcript_text, height=360)
        st.download_button(
            "Download transcript.txt",
            data=result.transcript_text.encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tabs[1]:
        if result.audit_validated is not None:
            st.code(json_pretty(result.audit_validated), language="json")
            st.download_button(
                "Download audit.json",
                data=json_pretty(result.audit_validated).encode("utf-8"),
                file_name="audit.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.error("Audit failed schema validation.")
            if result.validation_error:
                st.code(result.validation_error)
            st.caption("Raw model output (for debugging / prompt iteration):")
            st.code(result.audit_raw_text)

    st.divider()
    st.subheader("Run metadata")
    st.write(
        {
            "transcribe_model": meta.transcribe_model,
            "audit_model": meta.audit_model,
            "runtime_s": round(meta.runtime_s, 3),
            "status": meta.status,
            "cache_key": result.cache_key,
        }
    )


if __name__ == "__main__":
    main()

