from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
import json
import math
import os
from dotenv import load_dotenv
import uuid

# LangChain imports for conversational memory
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Secret key for sessions
app.secret_key = 'your-secret-key-here'

class CurriculumScheduler:
    def __init__(self, max_iterations=100):
        self.iteration = 0
        self.max_iterations = max_iterations
        self.scheduler_type = 'cosine'
        self.temperature = 0.7
        self.levels = ['trivial', 'easy', 'medium', 'hard']
        self.sigma = 1.0
        self.beta = 1.0
    
    def cosine_scheduler(self):
        t = self.iteration / self.max_iterations
        probs = {}
        
        for i, level in enumerate(self.levels):
            # Cosine curriculum: start easy, progress to hard
            phase = (t * math.pi / 2) + (i * math.pi / (2 * len(self.levels)))
            probs[level] = max(0, math.cos(phase - t * math.pi / 2))
        
        return self.normalize(probs)
    
    def gaussian_scheduler(self):
        t = self.iteration / self.max_iterations
        center = self.beta * t * (len(self.levels) - 1)
        probs = {}
        
        for i, level in enumerate(self.levels):
            distance = abs(i - center)
            probs[level] = math.exp(-(distance * distance) / (2 * self.sigma * self.sigma))
        
        return self.normalize(probs)
    
    def normalize(self, probs):
        total = sum(probs.values())
        if total == 0:
            return {level: 0.25 for level in self.levels}
        return {level: prob / total for level, prob in probs.items()}
    
    def sample_difficulty(self):
        probs = self.cosine_scheduler() if self.scheduler_type == 'cosine' else self.gaussian_scheduler()
        
        # Sample based on probabilities
        import random
        rand = random.random()
        cumulative = 0
        
        for level in self.levels:
            cumulative += probs[level]
            if rand < cumulative:
                return level, probs
        
        return 'hard', probs

class ConversationalAgent:
    def __init__(self):
        # Global scheduler instance
        self.scheduler = CurriculumScheduler()
        
        # Ollama configuration
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_model = "llama3.1:8b-instruct-q4_K_M"
        
        # Initialize LangChain Ollama
        self.llm = Ollama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.7
        )
        
        # Store conversations by session ID
        self.conversations = {}
        
    def get_difficulty_template(self, difficulty):
        """Get prompt templates based on difficulty level"""
        templates = {
            'trivial': """
            You are a helpful assistant. Give very brief, simple responses.
            Current conversation:
            {history}
            Human: {input}
            AI: [Respond in maximum 10 words, use simple language]""",
            
            'easy': """
            You are a helpful assistant. Give clear, straightforward answers.
            Current conversation:
            {history}
            Human: {input}
            AI: [Give a clear answer in 1-2 sentences]""",
            
            'medium': """
            You are a knowledgeable assistant. Provide thoughtful responses with relevant details.
            Current conversation:
            {history}
            Human: {input}
            AI: [Provide a thoughtful response with details and examples, 2-3 paragraphs]""",
            
            'hard': """
            You are an expert assistant. Provide comprehensive, detailed responses.
            Current conversation:
            {history}
            Human: {input}
            AI: [Give a comprehensive response with multiple perspectives, examples, implications, and thorough analysis. Write 4+ paragraphs with in-depth coverage]"""
        }
        
        return templates.get(difficulty, templates['medium'])
    
    def get_or_create_conversation(self, session_id, difficulty):
        """Get or create a conversation chain for the session"""
        if session_id not in self.conversations:
            print(f"🆕 Creating new conversation for session: {session_id}")
            
            # Create memory with window of last 10 exchanges (20 messages)
            memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True,
                memory_key="history"
            )
            
            # Create conversation chain
            self.conversations[session_id] = {
                'memory': memory,
                'chain': None,
                'difficulty_history': []
            }
        
        # Update LLM temperature
        self.llm.temperature = self.scheduler.temperature
        
        # Create or update the conversation chain with current difficulty prompt
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=self.get_difficulty_template(difficulty)
        )
        
        self.conversations[session_id]['chain'] = ConversationChain(
            llm=self.llm,
            memory=self.conversations[session_id]['memory'],
            prompt=prompt_template,
            verbose=True
        )
        
        return self.conversations[session_id]['chain']
    
    def chat(self, session_id, user_message):
        """Process chat with conversational memory"""
        try:
            print(f"🎭 Processing chat for session: {session_id}")
            print(f"📝 User message: {user_message}")
            
            # Get current difficulty level
            difficulty, probs = self.scheduler.sample_difficulty()
            print(f"🎯 Selected difficulty: {difficulty}")
            
            # Get or create conversation chain
            conversation = self.get_or_create_conversation(session_id, difficulty)
            
            # Store difficulty history
            self.conversations[session_id]['difficulty_history'].append(difficulty)
            
            print(f"🧠 Current memory length: {len(self.conversations[session_id]['memory'].chat_memory.messages)}")
            
            # Generate response with conversation context
            response = conversation.predict(input=user_message)
            
            print(f"✅ Generated response length: {len(response)}")
            
            # Update scheduler
            self.scheduler.iteration = min(self.scheduler.iteration + 1, self.scheduler.max_iterations)
            
            return {
                'response': response,
                'difficulty': difficulty,
                'probabilities': probs,
                'iteration': self.scheduler.iteration,
                'max_iterations': self.scheduler.max_iterations,
                'memory_length': len(self.conversations[session_id]['memory'].chat_memory.messages),
                'difficulty_history': self.conversations[session_id]['difficulty_history'][-10:]  # Last 10 difficulties
            }
            
        except Exception as e:
            print(f"❌ ERROR in conversational chat: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def reset_conversation(self, session_id):
        """Reset conversation memory for a session"""
        if session_id in self.conversations:
            print(f"🔄 Resetting conversation for session: {session_id}")
            del self.conversations[session_id]
        self.scheduler.iteration = 0
        return True
    
    def get_conversation_summary(self, session_id):
        """Get conversation summary for debugging"""
        if session_id not in self.conversations:
            return {"messages": 0, "history": []}
        
        memory = self.conversations[session_id]['memory']
        messages = memory.chat_memory.messages
        
        return {
            "messages": len(messages),
            "history": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                }
                for msg in messages[-6:]  # Last 6 messages
            ],
            "difficulty_history": self.conversations[session_id].get('difficulty_history', [])[-10:]
        }

# Initialize conversational agent
agent = ConversationalAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        print("=== LangChain Conversational Chat API Called ===")
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process chat with conversational memory
        result = agent.chat(session_id, user_message)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/summary', methods=['GET'])
def conversation_summary():
    """Get conversation summary for current session"""
    try:
        session_id = session.get('session_id', 'no-session')
        summary = agent.get_conversation_summary(session_id)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation memory"""
    try:
        session_id = session.get('session_id', 'no-session')
        agent.reset_conversation(session_id)
        
        # Create new session ID
        session['session_id'] = str(uuid.uuid4())
        
        return jsonify({
            'status': 'reset',
            'iteration': 0,
            'new_session_id': session['session_id']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update scheduler settings"""
    try:
        data = request.get_json()
        
        if 'scheduler_type' in data:
            agent.scheduler.scheduler_type = data['scheduler_type']
        
        if 'temperature' in data:
            agent.scheduler.temperature = float(data['temperature'])
            agent.llm.temperature = agent.scheduler.temperature
        
        if 'max_iterations' in data:
            agent.scheduler.max_iterations = int(data['max_iterations'])
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if Ollama is running and model is available"""
    try:
        response = requests.get(f"{agent.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            model_available = agent.ollama_model in models
            return jsonify({
                'ollama_running': True,
                'model_available': model_available,
                'available_models': models,
                'selected_model': agent.ollama_model,
                'langchain_enabled': True
            })
        else:
            return jsonify({'ollama_running': False}), 500
    except Exception as e:
        return jsonify({'ollama_running': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting E2H Curriculum Learning Agent with LangChain Memory")
    print(f"📦 Model: {agent.ollama_model}")
    print(f"🔗 Ollama URL: {agent.ollama_base_url}")
    print(f"🎯 Scheduler: {agent.scheduler.scheduler_type.title()}")
    print(f"🧠 Memory: ConversationBufferWindowMemory (k=10)")
    app.run(debug=True, host='0.0.0.0', port=5000)
