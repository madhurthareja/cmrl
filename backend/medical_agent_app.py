# Flask Web Interface for E2H Medical Agent System
# Integrates with existing Ollama setup and conversational memory

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import asyncio
import json
import logging
import traceback
from datetime import datetime

from agents.e2h_medical_agent import E2HMedicalAgent
from agents.medical_agent_core import DifficultyLevel, MedicalDomain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
CORS(app)

# Global agent instance
medical_agent = None

# Conversation memory
conversation_memory = {}

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
        print("🏥 Initializing E2H Medical Agent...")
        medical_agent = E2HMedicalAgent()
        print("✅ Medical Agent initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize medical agent: {e}")
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
        logger.info("🏥 E2H Medical Agent Flask Server Starting...")
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
    else:
        logger.error("Failed to initialize medical agent. Exiting.")
