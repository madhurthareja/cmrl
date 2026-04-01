# Flask Web Interface for E2H Medical Agent System
# Integrates with existing Ollama setup and conversational memory

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import asyncio
import json
import logging
import traceback
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import requests

from agents.medical_agent_core import DifficultyLevel, MedicalDomain
from models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
CORS(app)

# Pipeline mode: MedGemma only
medical_agent = None
medgemma_client = MedGemmaVQAClient(
    MedGemmaConfig(
        base_url=os.environ.get('MEDGEMMA_BASE_URL', 'http://localhost:8000'),
        model_name=os.environ.get('MEDGEMMA_MODEL', 'medgemma-4b-it_Q4_K_M'),
        api_key=os.environ.get('MEDGEMMA_API_KEY') or os.environ.get('LLAMA_API_KEY'),
    )
)

# Conversation memory
conversation_memory = {}


def _extract_json_object(text: str):
    """Best-effort JSON extraction from model text."""
    if not text:
        return None

    cleaned = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _to_float_01(value, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(1.0, parsed))


def _normalize_clinical_payload(raw_text: str) -> dict:
    """Normalize model output into a stable clinical contract payload."""
    parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return {
            "json_valid": False,
            "direct_answer": raw_text.strip() if raw_text else "",
            "task_answer": "not_applicable",
            "confidence": 0.5,
            "evidence_strength": "moderate",
            "modality_limits_acknowledged": False,
            "escalation_needed": False,
            "recommended_next_step": "",
            "reasoning": "",
            "raw": raw_text or "",
        }

    task_answer = str(parsed.get("task_answer", "not_applicable")).strip().lower()
    if task_answer not in {"yes", "no", "indeterminate", "not_applicable"}:
        task_answer = "not_applicable"

    evidence_strength = str(parsed.get("evidence_strength", "moderate")).strip().lower()
    if evidence_strength not in {"weak", "moderate", "strong"}:
        evidence_strength = "moderate"

    return {
        "json_valid": True,
        "direct_answer": str(parsed.get("direct_answer", "")).strip(),
        "task_answer": task_answer,
        "confidence": _to_float_01(parsed.get("confidence", 0.5), default=0.5),
        "evidence_strength": evidence_strength,
        "modality_limits_acknowledged": _to_bool(parsed.get("modality_limits_acknowledged", False)),
        "escalation_needed": _to_bool(parsed.get("escalation_needed", False)),
        "recommended_next_step": str(parsed.get("recommended_next_step", "")).strip(),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "raw": raw_text or "",
    }


def _build_clinical_prompt(question: str, context: str = "", for_vqa: bool = False) -> str:
    mode_hint = "image-grounded" if for_vqa else "text-grounded"
    context_block = f"Context:\n{context}\n\n" if context else ""
    return (
        f"You are a {mode_hint} clinical assistant.\n"
        f"{context_block}"
        f"Question: {question}\n"
        "Return valid JSON only with EXACT keys:\n"
        "direct_answer, task_answer, confidence, evidence_strength, modality_limits_acknowledged, escalation_needed, recommended_next_step, reasoning\n"
        "Rules:\n"
        "- task_answer: yes|no|indeterminate|not_applicable\n"
        "- confidence: float between 0 and 1\n"
        "- evidence_strength: weak|moderate|strong\n"
        "- modality_limits_acknowledged: true|false\n"
        "- escalation_needed: true|false\n"
        "- direct_answer should be concise and clinically useful"
    )


def _epistemic_gate(payload: dict) -> dict:
    """Heuristic neuro-symbolic checks over structured output."""
    confidence = _to_float_01(payload.get("confidence", 0.5), default=0.5)
    evidence_strength = str(payload.get("evidence_strength", "moderate")).lower()
    escalation_needed = _to_bool(payload.get("escalation_needed", False))
    modality_ack = _to_bool(payload.get("modality_limits_acknowledged", False))

    unsupported_certainty = 1.0 if (confidence >= 0.85 and evidence_strength == "weak") else 0.0
    escalation_mismatch = 1.0 if (confidence >= 0.85 and evidence_strength == "strong" and escalation_needed) else 0.0
    modality_mismatch = 1.0 if ((not modality_ack) and confidence >= 0.8) else 0.0

    penalty = (unsupported_certainty + escalation_mismatch + modality_mismatch) / 3.0
    return {
        "unsupported_certainty_flag": unsupported_certainty,
        "escalation_mismatch_flag": escalation_mismatch,
        "modality_mismatch_flag": modality_mismatch,
        "epistemic_validity": 1.0 - penalty,
    }


def _get_session_state(session_id: str):
    """Ensure a session bucket exists for the given session_id."""
    session_key = session_id or 'default'
    if session_key not in conversation_memory:
        conversation_memory[session_key] = {
            'messages': [],
            'context': '',
            'total_queries': 0
        }
    return conversation_memory[session_key]


def _build_history_context(messages, limit: int = 6) -> str:
    """Build a compact textual summary of the last few turns."""
    recent_messages = messages[-limit:]
    return "\n".join(
        f"Previous {msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in recent_messages
    )


def _serialize_retrieved_docs(docs, limit: int = 3):
    """Convert retrieved documents to serializable form."""
    payload = []
    for doc in docs[:limit]:
        payload.append(
            {
                "title": doc.title,
                "snippet": doc.content[:400],
                "similarity_score": round(doc.similarity_score, 4),
                "domain": doc.domain.value,
            }
        )
    return payload

@app.route('/')
def index():
    return render_template('medical_agent.html')

@app.route('/api/medical_chat', methods=['POST'])
async def medical_chat():
    """Handle medical consultation requests"""
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        if medgemma_client is None:
            return jsonify({'error': 'MedGemma client unavailable'}), 500

        session_state = _get_session_state(session_id)

        timestamp = datetime.now().isoformat()
        session_state['messages'].append(
            {
                'role': 'user',
                'content': user_message,
                'timestamp': timestamp,
            }
        )

        history_context = _build_history_context(session_state['messages'])

        domain = MedicalDomain.GENERAL
        difficulty = DifficultyLevel.MEDIUM
        retrieved_docs = []
        rag_context = ''

        combined_context_parts = [history_context.strip(), rag_context.strip()]
        combined_context = "\n\n".join(part for part in combined_context_parts if part)

        structured_prompt = _build_clinical_prompt(
            user_message,
            context=combined_context if combined_context else "",
            for_vqa=False,
        )

        medgemma_response = await medgemma_client.generate_text_response_async(
            structured_prompt,
            context=combined_context if combined_context else None,
        )

        raw_answer = medgemma_response.get('answer', '').strip()
        payload = _normalize_clinical_payload(raw_answer)
        gate = _epistemic_gate(payload)

        answer_text = payload.get('direct_answer', '').strip() or raw_answer
        if not answer_text:
            raise RuntimeError('MedGemma returned an empty answer')

        confidence = payload.get('confidence', 0.65)
        assistant_metadata = {
            'domain': domain.value,
            'difficulty_level': difficulty.value,
            'confidence': round(min(confidence, 0.92), 2),
            'reasoning': payload.get('reasoning') or 'Generated by MedGemma with structured contract.',
            'specialists_consulted': [],
            'retrieved_context': _serialize_retrieved_docs(retrieved_docs),
            'model': medgemma_response.get('model', medgemma_client.config.model_name),
            'usage': medgemma_response.get('usage', {}),
            'structured_output': payload,
            'epistemic_gate': gate,
            'contract_version': 'clinical-v1',
        }

        session_state['messages'].append(
            {
                'role': 'assistant',
                'content': answer_text,
                'metadata': assistant_metadata,
                'timestamp': datetime.now().isoformat(),
            }
        )
        session_state['total_queries'] += 1
        session_state['context'] = combined_context

        assistant_metadata['curriculum_status'] = {}

        session_stats = {
            'total_queries': session_state['total_queries'],
            'conversation_length': len(session_state['messages']),
        }

        return jsonify(
            {
                'response': answer_text,
                'metadata': assistant_metadata,
                'session_stats': session_stats,
            }
        )
        
    except Exception as e:
        logger.error(f"Error in medical chat: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Medical consultation failed: {str(e)}'}), 500


@app.route('/api/medgemma_infer', methods=['POST'])
def medgemma_infer():
    """Accept an image upload (or image_url) and optional text, call local vLLM medgemma server, return reply."""
    try:
        # Prepare upload directory under the static folder
        uploads_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        image_url = None

        # If multipart upload
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return jsonify({'error': 'No image file selected'}), 400

            filename = secure_filename(image.filename)
            save_path = os.path.join(uploads_dir, filename)
            image.save(save_path)

            # Build a URL that the vLLM server can fetch (Flask serves static files)
            # Using localhost and the Flask port (default 5001)
            flask_host = request.host.split(':')[0]
            flask_port = request.host.split(':')[1] if ':' in request.host else '5001'
            image_url = f"http://{flask_host}:{flask_port}/static/uploads/{filename}"

        # Alternatively accept JSON with image_url
        elif request.is_json:
            data = request.get_json()
            image_url = data.get('image_url')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Optional user text
        user_text = request.form.get('message') or (request.get_json() or {}).get('message') or "Describe this image in one sentence."

        # Build vLLM request payload according to the example
        payload = {
            "model": "google/medgemma-4b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": user_text },
                        { "type": "image_url", "image_url": { "url": image_url } }
                    ]
                }
            ]
        }

        # Call local vLLM server
        vllm_url = os.environ.get('VLLM_URL', 'http://localhost:8000/v1/chat/completions')
        headers = {'Content-Type': 'application/json'}

        resp = requests.post(vllm_url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return jsonify({'error': f'vLLM server returned {resp.status_code}', 'detail': resp.text}), 502

        resp_json = resp.json()
        # Attempt to extract text from response (model-dependent)
        model_reply = ''
        try:
            # vLLM reply structure may have choices -> message -> content
            if isinstance(resp_json, dict) and 'choices' in resp_json:
                first = resp_json['choices'][0]
                if 'message' in first and 'content' in first['message']:
                    content = first['message']['content']
                    # content might be a list of blocks
                    if isinstance(content, list):
                        parts = []
                        for c in content:
                            if isinstance(c, dict) and 'text' in c:
                                parts.append(c['text'])
                            elif isinstance(c, str):
                                parts.append(c)
                        model_reply = '\n'.join(parts)
                    elif isinstance(content, str):
                        model_reply = content
            # Fallback
            if not model_reply:
                model_reply = resp_json.get('response') or json.dumps(resp_json)
        except Exception:
            model_reply = str(resp_json)

        return jsonify({'response': model_reply, 'image_url': image_url})

    except Exception as e:
        logger.error(f"medgemma_infer error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/curriculum_status', methods=['GET'])
def get_curriculum_status():
    """Get current curriculum learning status"""
    try:
        return jsonify({'status': 'disabled', 'reason': 'MedGemma-only mode does not use curriculum scheduler'})
        
    except Exception as e:
        logger.error(f"Error getting curriculum status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vqa', methods=['POST'])
async def medical_vqa():
    """Handle medical visual question answering requests."""
    try:
        if medgemma_client is None:
            return jsonify({'error': 'MedGemma client unavailable'}), 500

        session_id = (
            request.form.get('session_id')
            or request.args.get('session_id')
            or 'default'
        )
        session_state = _get_session_state(session_id)

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({'error': 'Uploaded image is empty'}), 400

        mime_type = image_file.mimetype or 'image/png'

        domain = MedicalDomain.GENERAL
        difficulty = DifficultyLevel.MEDIUM
        retrieved_docs = []
        context_snippets = ''

        structured_prompt = _build_clinical_prompt(question, context=context_snippets, for_vqa=True)

        model_response = await medgemma_client.answer_question_async(
            image_bytes,
            structured_prompt,
            mime_type=mime_type,
            context=context_snippets if context_snippets else None,
        )

        raw_answer = model_response.get('answer', '').strip()
        payload = _normalize_clinical_payload(raw_answer)
        gate = _epistemic_gate(payload)

        answer_text = payload.get('direct_answer', '').strip() or raw_answer
        if not answer_text:
            raise RuntimeError('MedGemma returned an empty answer')
        confidence = payload.get('confidence', 0.62)

        assistant_metadata = {
            'domain': domain.value,
            'difficulty_level': difficulty.value,
            'confidence': round(min(confidence, 0.93), 2),
            'retrieved_context': _serialize_retrieved_docs(retrieved_docs),
            'model': model_response.get('model', medgemma_client.config.model_name),
            'usage': model_response.get('usage', {}),
            'vqa': True,
            'structured_output': payload,
            'epistemic_gate': gate,
            'contract_version': 'clinical-v1',
        }

        session_state['messages'].extend(
            [
                {
                    'role': 'user',
                    'content': f"[Image Question] {question}",
                    'metadata': {'type': 'vqa', 'mime_type': mime_type},
                    'timestamp': datetime.now().isoformat(),
                },
                {
                    'role': 'assistant',
                    'content': f"[Image Analysis] {answer_text}",
                    'metadata': assistant_metadata,
                    'timestamp': datetime.now().isoformat(),
                },
            ]
        )
        session_state['total_queries'] += 1
        session_state['context'] = _build_history_context(session_state['messages'])

        session_stats = {
            'total_queries': session_state['total_queries'],
            'conversation_length': len(session_state['messages']),
        }

        return jsonify(
            {
                'answer': answer_text,
                'model': assistant_metadata['model'],
                'usage': assistant_metadata['usage'],
                'domain': domain.value,
                'difficulty_level': difficulty.value,
                'confidence': assistant_metadata['confidence'],
                'retrieved_context': assistant_metadata['retrieved_context'],
                'structured_output': payload,
                'epistemic_gate': gate,
                'session_stats': session_stats,
            }
        )

    except Exception as e:  # pragma: no cover - runtime failure
        logger.error("Error in medical VQA: %s", e)
        traceback.print_exc()
        return jsonify({'error': f'Medical VQA failed: {str(e)}'}), 500

@app.route('/api/reset_curriculum', methods=['POST'])
def reset_curriculum():
    """Reset curriculum progress"""
    try:
        return jsonify({'status': 'disabled', 'reason': 'MedGemma-only mode does not use curriculum scheduler'})
        
    except Exception as e:
        logger.error(f"Error resetting curriculum: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation_memory/<session_id>', methods=['GET'])
def get_conversation_memory(session_id):
    """Get conversation history for session"""
    try:
        if session_id in conversation_memory:
            return jsonify(conversation_memory[session_id])
        else:
            return jsonify({
                'messages': [],
                'context': '',
                'total_queries': 0
            })
    except Exception as e:
        logger.error(f"Error getting conversation memory: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_memory/<session_id>', methods=['POST'])
def clear_conversation_memory(session_id):
    """Clear conversation memory for session"""
    try:
        if session_id in conversation_memory:
            del conversation_memory[session_id]
        return jsonify({'status': 'Memory cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/medical_domains', methods=['GET'])
def get_medical_domains():
    """Get available medical domains"""
    domains = [domain.value for domain in MedicalDomain]
    return jsonify({'domains': domains})

@app.route('/api/difficulty_levels', methods=['GET'])
def get_difficulty_levels():
    """Get available difficulty levels"""
    levels = [level.value for level in DifficultyLevel]
    return jsonify({'difficulty_levels': levels})

@app.route('/api/train_model', methods=['POST'])
async def train_model():
    """Train the medical agent with provided data"""
    try:
        return jsonify({'status': 'disabled', 'reason': 'Training endpoint is unavailable in MedGemma-only mode'}), 400
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

async def init_medical_agent():
    """Initialize MedGemma-only runtime."""
    global medical_agent
    try:
        print("Initializing MedGemma-only pipeline...")
        medical_agent = None
        print("MedGemma-only pipeline initialized successfully.")
        return True
    except Exception as e:
        print(f"Failed to initialize medical agent: {e}")
        import traceback
        traceback.print_exc()
        # Create a mock agent for testing
        medical_agent = None
        return False

def initialize_app():
    """Initialize the medical agent when app starts"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_medical_agent())
    except Exception as e:
        print(f"Error during initialization: {e}")

# Initialize on module load
initialize_app()

if __name__ == '__main__':
    # Initialize the medical agent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run initialization
    success = loop.run_until_complete(init_medical_agent())
    
    if success:
        logger.info("MedGemma Flask Server Starting...")
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
    else:
        logger.error("Failed to initialize medical agent. Exiting.")
