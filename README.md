# AI-driven-Call-Audit


#Problem
Sales teams conduct many demo calls daily, but reviewing them manually
is tough and getting insights from all calls and the quality of sales person requrie much effort and time.

This prototype demonstrates how an LLM-based workflow can automatically
analyze call transcripts and generate structured insights.

#Workflow / Architecture

Flow:
1. Recording input
2. The LLM audits the call and pushes the insights
3. Transcript generation by LLM if call is less than 2 min
4. Output stored and displayed

You can even add a simple diagram:

Recording url → LLM (Gemini) → Structured Output → Dashboard / JSON

#Tech Stack
Python
FastAPI
Google Gemini LLM
Langfuse (open source integrated)

Add your deployed link:

Live Demo:
https://ai-driven-call-audit-4cdph9mmjred3fygkap6pq.streamlit.app/

