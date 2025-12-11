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

from agents.e2h_medical_agent import E2HMedicalAgent
from agents.medical_agent_core import DifficultyLevel, MedicalDomain
from models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
CORS(app)

# Global agent instance
medical_agent = None
medgemma_client = MedGemmaVQAClient(
    MedGemmaConfig(
        base_url=os.environ.get('MEDGEMMA_BASE_URL', 'http://localhost:8000'),
        model_name=os.environ.get('MEDGEMMA_MODEL', 'medgemma-4b-it_Q4_K_M')
    )
)

# Conversation memory
conversation_memory = {}


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
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        if medical_agent is None:
            return jsonify({'error': 'Medical agent not initialized'}), 500
        
        # Initialize conversation memory for session
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                'messages': [],
                'context': '',
                'total_queries': 0
            }
        
        # Add user message to memory
        conversation_memory[session_id]['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Build context from conversation history
        context = "\n".join([
            f"Previous {msg['role']}: {msg['content']}" 
            for msg in conversation_memory[session_id]['messages'][-5:]  # Last 5 messages
        ])
        
        # Process medical query
        response = await medical_agent.process_medical_query(user_message, context)
        
        # Add response to memory
        conversation_memory[session_id]['messages'].append({
            'role': 'assistant',
            'content': response.answer,
            'metadata': {
                'domain': response.domain.value,
                'difficulty': response.difficulty_level.value,
                'confidence': response.confidence,
                'specialists': [cons.specialist_type for cons in response.specialist_consultations],
                'retrieved_docs': len(response.retrieved_context)
            },
            'timestamp': datetime.now().isoformat()
        })
        
        conversation_memory[session_id]['total_queries'] += 1
        
        # Get curriculum status
        curriculum_status = medical_agent.get_curriculum_status()
        
        return jsonify({
            'response': response.answer,
            'metadata': {
                'domain': response.domain.value,
                'difficulty_level': response.difficulty_level.value,
                'confidence': response.confidence,
                'reasoning': response.reasoning[:500] + "..." if len(response.reasoning) > 500 else response.reasoning,
                'specialists_consulted': [cons.specialist_type for cons in response.specialist_consultations],
                'retrieved_context': response.retrieved_context,
                'curriculum_status': curriculum_status
            },
            'session_stats': {
                'total_queries': conversation_memory[session_id]['total_queries'],
                'conversation_length': len(conversation_memory[session_id]['messages'])
            }
        })
        
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
        if medical_agent is None:
            return jsonify({'error': 'Medical agent not initialized'}), 500
        
        status = medical_agent.get_curriculum_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting curriculum status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vqa', methods=['POST'])
async def medical_vqa():
    """Handle medical visual question answering requests."""
    try:
        if medgemma_client is None:
            return jsonify({'error': 'MedGemma client unavailable'}), 500

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

        if medical_agent is not None:
            try:
                # Estimate domain/difficulty from question
                domain = medical_agent.triage_agent.domain_classifier.classify_domain(question)
                difficulty = medical_agent.difficulty_classifier.classify_difficulty_level(question)
                retrieved_docs = await medical_agent.rag_system.retrieve_with_curriculum(
                    question, domain, difficulty
                )
            except Exception as retrieval_error:  # pragma: no cover - fallback path
                logger.warning("VQA retrieval fallback: %s", retrieval_error)
                retrieved_docs = []

        context_snippets = "\n".join(
            f"{doc.title}: {doc.content[:400]}" for doc in retrieved_docs[:3]
        )

        model_response = await medgemma_client.answer_question_async(
            image_bytes,
            question,
            mime_type=mime_type,
            context=context_snippets if context_snippets else None,
        )

        return jsonify(
            {
                'answer': model_response.get('answer', '').strip(),
                'model': model_response.get('model'),
                'usage': model_response.get('usage', {}),
                'domain': domain.value,
                'difficulty_level': difficulty.value,
                'retrieved_context': _serialize_retrieved_docs(retrieved_docs),
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
        if medical_agent is None:
            return jsonify({'error': 'Medical agent not initialized'}), 500
        
        medical_agent.curriculum_scheduler.iteration = 0
        return jsonify({'status': 'Curriculum reset successfully'})
        
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
        data = request.get_json()
        training_examples = data.get('training_examples', [])
        
        if not training_examples:
            return jsonify({'error': 'No training examples provided'}), 400
        
        if medical_agent is None:
            return jsonify({'error': 'Medical agent not initialized'}), 500
        
        # Convert training examples to expected format
        training_data = [(ex['question'], ex['answer']) for ex in training_examples]
        
        # Run training
        results = await medical_agent.train_with_curriculum(training_data)
        
        return jsonify({
            'status': 'Training completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

async def init_medical_agent():
    """Initialize the medical agent asynchronously"""
    global medical_agent
    try:
        print("Initializing E2H Medical Agent...")
        medical_agent = E2HMedicalAgent()
        print("Medical Agent initialized successfully.")
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
        logger.info("E2H Medical Agent Flask Server Starting...")
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
    else:
        logger.error("Failed to initialize medical agent. Exiting.")
