# Medical Agent

## Pipeline Overview

The system combines curriculum-aware text consultation with medical visual question answering (VQA). The end-to-end flow is:

```
User (text or VQA request)
    │
    ├── Web front end (`frontend/templates/medical_agent.html`)
    │       • Provides chat UI, image uploader, and session controls
    │       • Packs VQA requests as `FormData` with image + question
    │
    └── Flask backend (`backend/medical_agent_app.py`)
            │
            ├─ Domain triage and difficulty prediction (`E2HMedicalAgent`)
            │       • Uses classifier heads to infer medical specialty and case difficulty
            │       • Selects curriculum policy and specialist agents
            │
            ├─ Retrieval layer (`retrieval/medrag_system.py`)
            │       • Searches FAISS index and sentence-transformer embeddings
            │       • Returns top documents serialized for the client
            │
            └─ Response generation
                    • Text-only: delegates to specialist pipelines and structured reasoners
                    • VQA: routes to `MedGemmaVQAClient` (`models/medgemma_vqa.py`)
                            - Encodes image bytes as base64 for the vLLM endpoint
                            - Sends multimodal prompt to the MedGemma model
                            - Parses chat-completions response into plain text answer

Response payloads are sent back to the browser, which updates chat history, confidence metrics, and retrieved-document summaries.
```

## Why This Pipeline Is Better

- **Unified experience:** Text consultations and image-grounded questions share the same session state, so curriculum progress, confidence tracking, and specialist routing apply consistently across modalities.
- **Curriculum-aware reasoning:** Difficulty sampling and domain triage let the agent scale responses from introductory guidance to expert-level analysis without manual tuning for every conversation.
- **Retrieval grounding:** The MedRAG subsystem supplies document snippets that surface supporting evidence to both the VQA client and the user, reducing hallucinations and increasing transparency.
- **vLLM compatibility:** The MedGemma client speaks the OpenAI chat-completions protocol, allowing drop-in replacement of the underlying model or hosting stack without UI changes.
- **Responsive front end:** The asynchronous VQA workflow previews images, validates inputs, and streams updates so users always see the context for generated answers.

## Quick Start 

1. **Clone & enter the repo**

        ```bash
        git clone https://github.com/madhurthareja/cmrl.git
        cd cmrl
        ```

2. **Create a Python environment** (Python 3.10+)

        ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

3. **Download the MedGemma GGUF + projector**

        ```bash
        mkdir -p models/gguf && cd models/gguf
        wget https://huggingface.co/SandLogicTechnologies/MedGemma-4B-IT-GGUF/resolve/main/medgemma-4b-it-Q4_K_M.gguf
        wget https://huggingface.co/SandLogicTechnologies/MedGemma-4B-IT-GGUF/resolve/main/mmproj-medgemma-4b-f16.gguf
        cd ../..
        ```

        - Use `medgemma-4b-it-Q4_K_M.gguf` on 4–6 GB GPUs (RTX 3050 friendly). Swap in `Q5_K_M` for more quality if VRAM allows.
        - The `mmproj` file is required for vision inputs.

4. **Run the llama.cpp server (OpenAI-compatible)**

        ```bash
        ./llama-server \
          -m models/gguf/medgemma-4b-it-Q4_K_M.gguf \
          --mmproj models/gguf/mmproj-medgemma-4b-f16.gguf \
          --host 0.0.0.0 --port 8000 \
          --ctx-size 4096 --n-gpu-layers 35 --threads 8
        ```

        Keep this terminal running; it exposes `http://localhost:8000/v1/chat/completions`.

5. **Start the Flask backend** (new terminal)

        ```bash
        source venv/bin/activate
        export FLASK_ENV=development               # optional, enables hot reload
        export VLLM_URL=http://localhost:8000/v1/chat/completions
        python backend/medical_agent_app.py
        ```

6. **Open the UI**

        Visit `http://127.0.0.1:5001` and try either text chat or the “Ask About Image” workflow.

### Switching or Updating Models

The MedGemma client reads its config from `backend/models/medgemma_vqa.py`:

```python
medgemma_client = MedGemmaVQAClient(
         MedGemmaConfig(
                  base_url=os.environ.get('MEDGEMMA_BASE_URL', 'http://localhost:8000'),
                  model_name=os.environ.get('MEDGEMMA_MODEL', 'medgemma-4b-it_Q4_K_M')
         )
)
```

- **Different GGUF / remote endpoint:** start llama.cpp (or vLLM) elsewhere and point the backend via `MEDGEMMA_BASE_URL` or `VLLM_URL`.
- **Switch precision (Q4 → Q5) or an entirely different model:** restart the llama server with the new weights and set `MEDGEMMA_MODEL` to the identifier you want to see in responses.
- **Tweak decoding:** edit the `MedGemmaConfig` fields (`max_tokens`, `temperature`, `top_p`) to match the new model’s sweet spot.

If you maintain multiple configs, consider creating small shell scripts (e.g., `scripts/run_llama_q4.sh`, `scripts/run_llama_q5.sh`) so you can swap models with a single command.

## Running Tests

Install dev deps and run pytest from the repo root:

```bash
pip install -r requirements.txt
pytest tests/test_medgemma_vqa.py
```

## Troubleshooting

- If VQA requests fail with connection errors, ensure the vLLM server is running and reachable.
- To adjust MedGemma parameters, edit `MedGemmaConfig` in `backend/models/medgemma_vqa.py` (base URL, model name, decoding limits).
- For curriculum or retrieval tuning, update the relevant modules under `backend/agents` and `backend/retrieval`.
