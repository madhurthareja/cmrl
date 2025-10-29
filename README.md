# CMRL Medical Agent

## Pipeline Overview

The refreshed stack keeps the curriculum + VQA intelligence while replacing the UI and runtime surface:

```
User (text or VQA request)
    │
    ├── LibreChat-inspired React client (`frontend/`)
    │       • Dual composer for text + image questions
    │       • Shows agent badges, curriculum progress, and retrieved snippets
    │
    └── FastAPI runtime (`server/main.py`)
            │
            ├─ Domain triage and difficulty sampling (`E2HMedicalAgent`)
            │       • Routes cases to relevant mock specialists
            │       • Exposes per-role confidence metadata
            │
            ├─ Retrieval layer (`backend/retrieval/medrag_system.py`)
            │       • Curriculum-aware filtering with FAISS indices
            │       • Supplies context to both text and VQA flows
            │
            └─ Response generation
                    • Text: integrates specialist opinions and RAG evidence
                    • VQA: enriches MedGemma prompts with retrieved snippets

Responses return to the browser, which threads them into the conversation timeline and updates curriculum telemetry.
```

## Why This Pipeline Is Better

- **Unified experience:** Text consultations and image-grounded questions share the same session state, so curriculum progress, confidence tracking, and specialist routing apply consistently across modalities.
- **Curriculum-aware reasoning:** Difficulty sampling and domain triage let the agent scale responses from introductory guidance to expert-level analysis without manual tuning for every conversation.
- **Retrieval grounding:** The MedRAG subsystem supplies document snippets that surface supporting evidence to both the VQA client and the user, reducing hallucinations and increasing transparency.
- **vLLM compatibility:** The MedGemma client speaks the OpenAI chat-completions protocol, allowing drop-in replacement of the underlying model or hosting stack without UI changes.
- **Responsive front end:** The asynchronous VQA workflow previews images, validates inputs, and streams updates so users always see the context for generated answers.

## Getting Started

1. Launch the MedGemma model with vLLM (defaults to `http://localhost:8000`).
2. Install Python deps: `pip install -r requirements.txt`.
3. Start the API: `uvicorn server.main:app --reload --port 8001`.
4. Install frontend deps: `cd frontend && npm install`.
5. Boot the UI: `npm run dev` (proxies API calls to port 8001).
6. Open the Vite dev server (default `http://localhost:5173`).

## Troubleshooting

- If VQA requests fail, confirm the vLLM server is reachable from the FastAPI host.
- Adjust MedGemma parameters via `MedGemmaConfig` in `backend/models/medgemma_vqa.py`.
- Tune curriculum or retrieval logic under `backend/agents` and `backend/retrieval`.
- Experiments, finetuning, and benchmarking scripts now live in `experiments/`.
