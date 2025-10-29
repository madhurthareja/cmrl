"""FastAPI runtime for the E2H curriculum + MedGemma VQA stack."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents.e2h_medical_agent import AgentResponse, E2HMedicalAgent
from backend.agents.medical_agent_core import DifficultyLevel, MedicalDomain
from backend.models.medgemma_vqa import MedGemmaVQAClient


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or prompt")
    session_id: Optional[str] = Field(None, description="Conversation identifier")


class SpecialistPayload(BaseModel):
    name: str
    confidence: float
    summary: str


class ChatResponse(BaseModel):
    answer: str
    domain: str
    difficulty_level: str
    confidence: float
    reasoning_snippet: str
    specialists: List[SpecialistPayload]
    retrieved_context: List[str]
    session: Dict[str, int]
    timestamp: datetime


class TrainingExample(BaseModel):
    question: str
    answer: str


class TrainRequest(BaseModel):
    training_examples: List[TrainingExample]


class TrainResponse(BaseModel):
    status: str
    epochs: int
    final_avg_reward: float
    curriculum_status: Dict[str, object]


class CurriculumStatus(BaseModel):
    iteration: int
    max_iterations: int
    difficulty_distribution: Dict[str, float]
    scheduler_type: str


class CurriculumResetResponse(BaseModel):
    status: str


app = FastAPI(title="CMRL Curriculum API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


medical_agent: Optional[E2HMedicalAgent] = None
medgemma_client = MedGemmaVQAClient()
conversation_memory: Dict[str, Dict[str, object]] = {}
MEMORY_WINDOW = 5


async def init_agent() -> None:
    global medical_agent
    medical_agent = E2HMedicalAgent()


@app.on_event("startup")
async def startup_event() -> None:
    await init_agent()


def _ensure_agent() -> E2HMedicalAgent:
    if medical_agent is None:
        raise HTTPException(status_code=503, detail="Medical agent is initializing")
    return medical_agent


def _seed_session(session_id: str) -> None:
    if session_id not in conversation_memory:
        conversation_memory[session_id] = {
            "messages": [],
            "total_queries": 0,
        }


def _append_message(session_id: str, role: str, content: str, metadata: Optional[Dict[str, object]] = None) -> None:
    record = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if metadata:
        record["metadata"] = metadata
    conversation_memory[session_id]["messages"].append(record)


def _build_context(session_id: str) -> str:
    messages = conversation_memory[session_id]["messages"][-MEMORY_WINDOW:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _serialize_specialists(agent_response: AgentResponse) -> List[SpecialistPayload]:
    payload = []
    for specialist in agent_response.specialist_consultations:
        payload.append(
            SpecialistPayload(
                name=specialist.specialist_type,
                confidence=round(specialist.confidence, 3),
                summary=specialist.reasoning,
            )
        )
    return payload


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    agent = _ensure_agent()

    session_id = body.session_id or "default"
    _seed_session(session_id)

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is empty")

    _append_message(session_id, "user", user_message)
    context = _build_context(session_id)

    agent_response = await agent.process_medical_query(user_message, context)

    specialist_summary = [s.specialist_type for s in agent_response.specialist_consultations]
    metadata = {
        "domain": agent_response.domain.value,
        "difficulty": agent_response.difficulty_level.value,
        "confidence": agent_response.confidence,
        "specialists": specialist_summary,
        "retrieved_docs": len(agent_response.retrieved_context),
    }
    _append_message(session_id, "assistant", agent_response.answer, metadata=metadata)

    conversation_memory[session_id]["total_queries"] += 1

    reasoning_snippet = agent_response.reasoning[:500]
    if len(agent_response.reasoning) > 500:
        reasoning_snippet += "…"

    return ChatResponse(
        answer=agent_response.answer,
        domain=agent_response.domain.value,
        difficulty_level=agent_response.difficulty_level.value,
        confidence=agent_response.confidence,
        reasoning_snippet=reasoning_snippet,
        specialists=_serialize_specialists(agent_response),
        retrieved_context=agent_response.retrieved_context,
        session={
            "total_queries": conversation_memory[session_id]["total_queries"],
            "message_count": len(conversation_memory[session_id]["messages"]),
        },
        timestamp=datetime.utcnow(),
    )


@app.post("/vqa")
async def vqa_endpoint(
    question: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> Dict[str, object]:
    agent = _ensure_agent()

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    question_text = question.strip()
    if not question_text:
        raise HTTPException(status_code=400, detail="Question is required")

    domain = agent.triage_agent.domain_classifier.classify_domain(question_text)
    difficulty = agent.difficulty_classifier.classify_difficulty_level(question_text)

    try:
        retrieved_docs = await agent.rag_system.retrieve_with_curriculum(
            question_text,
            domain,
            difficulty,
        )
    except Exception:
        retrieved_docs = []

    context = "\n".join(f"{doc.title}: {doc.content[:400]}" for doc in retrieved_docs[:3]) or None

    model_response = await medgemma_client.answer_question_async(
        payload,
        question_text,
        mime_type=image.content_type or "image/png",
        context=context,
    )

    retrieved_payload = [
        {
            "title": doc.title,
            "snippet": doc.content[:400],
            "similarity": round(doc.similarity_score, 4),
            "domain": doc.domain.value,
        }
        for doc in retrieved_docs[:5]
    ]

    if session_id:
        _seed_session(session_id)
        _append_message(session_id, "user", f"[VQA] {question_text}")
        _append_message(
            session_id,
            "assistant",
            model_response.get("answer", ""),
            metadata={
                "domain": domain.value,
                "difficulty": difficulty.value,
                "specialists": [],
                "retrieved_docs": len(retrieved_docs),
            },
        )

    return {
        "answer": model_response.get("answer", "").strip(),
        "model": model_response.get("model"),
        "usage": model_response.get("usage", {}),
        "domain": domain.value,
        "difficulty_level": difficulty.value,
        "retrieved_context": retrieved_payload,
    }


@app.get("/curriculum/status", response_model=CurriculumStatus)
async def curriculum_status() -> CurriculumStatus:
    agent = _ensure_agent()
    status = agent.get_curriculum_status()
    return CurriculumStatus(**status)


@app.post("/curriculum/reset", response_model=CurriculumResetResponse)
async def curriculum_reset() -> CurriculumResetResponse:
    agent = _ensure_agent()
    agent.curriculum_scheduler.iteration = 0
    return CurriculumResetResponse(status="Curriculum reset successfully")


@app.post("/training/grpo", response_model=TrainResponse)
async def train_grpo(body: TrainRequest) -> TrainResponse:
    agent = _ensure_agent()

    if not body.training_examples:
        raise HTTPException(status_code=400, detail="No training examples provided")

    training_pairs = [(ex.question, ex.answer) for ex in body.training_examples]
    results = await agent.train_with_curriculum(training_pairs)

    return TrainResponse(
        status="Training completed",
        epochs=results["epochs"],
        final_avg_reward=results["final_avg_reward"],
        curriculum_status=results["curriculum_status"],
    )


@app.get("/health")
async def health() -> Dict[str, object]:
    healthy = medical_agent is not None
    return {
        "ready": healthy,
        "has_medgemma": medgemma_client is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8001, reload=True)
