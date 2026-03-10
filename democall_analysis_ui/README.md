# Demo Call Analysis UI (Streamlit prototype)

A small, recruiter-friendly Streamlit prototype that demonstrates the **AI-driven call audit workflow** end-to-end:

- **Audio (.mp3/.wav/.m4a)** → Gemini transcription (using the existing `transcript_prompt`)
- **Transcript** → Gemini structured audit JSON (using the existing `prompt`)
- **Schema validation** using the existing `CallAudit` (Pydantic)

This prototype intentionally skips **DB fetch**, **Mongo writes**, and **LeadSquared API** so it’s safe to share and easy to run locally.

## Local run

1) Create a venv and install deps:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r democall_analysis_ui/requirements.txt
```

2) Set env var (recommended: copy `.env.example` → `.env`):

```bash
copy democall_analysis_ui\\.env.example democall_analysis_ui\\.env
```

Then edit `democall_analysis_ui/.env` and set `GOOGLE_GEMINI_API_KEY`.

3) Run Streamlit:

```bash
streamlit run democall_analysis_ui/app.py
```

## Deploy (Streamlit Community Cloud)

- **Repo must be on GitHub** (public or private).
- In Streamlit Cloud:
  - **App file**: `democall_analysis_ui/app.py`
  - **Python version**: 3.11+ (3.12 recommended)
  - **Secrets**: set `GOOGLE_GEMINI_API_KEY` in the Streamlit “Secrets” UI
  - **Requirements**: Streamlit will use `democall_analysis_ui/requirements.txt` (or you can set it explicitly in advanced settings if needed)

## Deploy (Render.com, quick path)

- Create a new **Web Service** from the repo
- **Build command**:

```bash
pip install -r democall_analysis_ui/requirements.txt
```

- **Start command**:

```bash
streamlit run democall_analysis_ui/app.py --server.port $PORT --server.address 0.0.0.0
```

- Add env var `GOOGLE_GEMINI_API_KEY`

## Screenshot

_(placeholder)_ Add a screenshot here once deployed.

