class ConversationalCurriculumAgent {
    constructor() {
        this.messageCount = 0;
        this.totalResponseLength = 0;
        this.currentIteration = 0;
        this.maxIterations = 100;
        this.conversationMemory = [];
        
        this.initializeApp();
    }

    initializeApp() {
        // DOM elements
        this.chatContainer = document.getElementById('chat');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.memoryBtn = document.getElementById('memoryBtn');

        // Event listeners
        this.resetBtn.addEventListener('click', () => this.resetConversation());
        this.memoryBtn.addEventListener('click', () => this.showMemorySummary());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Settings controls
        document.getElementById('schedulerType').addEventListener('change', (e) => {
            this.updateSettings({ scheduler_type: e.target.value });
        });

        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('tempValue').textContent = e.target.value;
            this.updateSettings({ temperature: parseFloat(e.target.value) });
        });

        // Initial health check
        this.healthCheck();
        
        this.addSystemMessage('Conversational E2H Agent Ready! I remember our conversation context.');
        this.enableChat();
    }

    async healthCheck() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.ollama_running && data.model_available) {
                this.addSystemMessage(`Ollama running with ${data.selected_model} | LangChain Memory: ${data.langchain_enabled ? 'Enabled' : 'Disabled'}`);
            } else {
                this.addSystemMessage('Ollama not running or model unavailable. Please start Ollama.');
            }
        } catch (error) {
            this.addSystemMessage('Health check failed. Please ensure Ollama is running.');
        }
    }

    enableChat() {
        this.messageInput.disabled = false;
        this.sendBtn.disabled = false;
        this.messageInput.focus();
        document.getElementById('currentMode').textContent = 'LangChain + Ollama';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        this.messageInput.value = '';
        this.setLoading(true);

        // Add user message
        this.addMessage('user', message);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Add agent response
            this.addMessage('agent', data.response, data.difficulty);

            // Update UI with new data
            this.updateProgress(data);

            // Update memory info
            if (data.memory_length !== undefined) {
                document.getElementById('memoryLength').textContent = Math.floor(data.memory_length / 2); // Exchanges
                document.getElementById('memorySize').textContent = data.memory_length; // Total messages
            }

        } catch (error) {
            this.addMessage('agent', `Error: ${error.message}`, 'error');
        }

        this.setLoading(false);
        this.messageInput.focus();
    }

    async resetConversation() {
        try {
            const response = await fetch('/api/conversation/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.currentIteration = 0;
                this.messageCount = 0;
                this.totalResponseLength = 0;
                this.conversationMemory = [];
                
                // Clear chat
                this.chatContainer.innerHTML = '';
                
                this.updateUI();
                this.addSystemMessage(`Conversation reset. New session: ${data.new_session_id.substring(0, 8)}... | Fresh memory initialized.`);
                
                // Reset memory display
                document.getElementById('memoryLength').textContent = '0';
                document.getElementById('memorySize').textContent = '0';
            }
        } catch (error) {
            console.error('Reset failed:', error);
        }
    }

    async showMemorySummary() {
        try {
            const response = await fetch('/api/conversation/summary');
            const data = await response.json();
            
            let summaryText = `Conversation Summary:\n\n`;
            summaryText += `Total Messages: ${data.messages}\n`;
            summaryText += `Recent Difficulties: ${data.difficulty_history.join(', ')}\n\n`;
            summaryText += `Recent Messages:\n`;
            
            data.history.forEach((msg, index) => {
                const speaker = msg.type === 'human' ? 'User' : 'Assistant';
                summaryText += `${speaker}: ${msg.content}\n\n`;
            });
            
            this.addSystemMessage(summaryText);
        } catch (error) {
            this.addSystemMessage('Failed to get memory summary.');
        }
    }

    async updateSettings(settings) {
        try {
            await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            });
        } catch (error) {
            console.error('Settings update failed:', error);
        }
    }

    setLoading(isLoading) {
        this.sendBtn.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        
        if (isLoading) {
            this.sendBtn.textContent = 'Thinking...';
            this.chatContainer.classList.add('loading');
        } else {
            this.sendBtn.textContent = 'Send';
            this.chatContainer.classList.remove('loading');
        }
    }

    addMessage(role, content, difficulty = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (difficulty && role === 'agent') {
            const badge = document.createElement('div');
            badge.className = `difficulty-badge ${difficulty}`;
            badge.textContent = difficulty.toUpperCase();
            contentDiv.appendChild(badge);
            contentDiv.appendChild(document.createElement('br'));
        }

        // Handle multiline content
        const lines = content.split('\n');
        lines.forEach((line, index) => {
            if (index > 0) contentDiv.appendChild(document.createElement('br'));
            contentDiv.appendChild(document.createTextNode(line));
        });
        
        messageDiv.appendChild(contentDiv);
        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;

        // Update stats
        if (role === 'user') {
            this.messageCount++;
        } else if (role === 'agent' && difficulty !== 'error') {
            this.totalResponseLength += content.length;
        }
    }

    addSystemMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        messageDiv.style.whiteSpace = 'pre-wrap'; // Preserve line breaks
        messageDiv.textContent = content;
        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    updateProgress(data) {
        this.currentIteration = data.iteration;
        this.maxIterations = data.max_iterations;
        
        // Update difficulty probabilities
        if (data.probabilities) {
            for (let [level, prob] of Object.entries(data.probabilities)) {
                const percentage = Math.round(prob * 100);
                document.getElementById(`${level}Prob`).textContent = `${percentage}%`;
                document.getElementById(`${level}Bar`).style.width = `${percentage}%`;
            }
        }
        
        // Update difficulty history display
        if (data.difficulty_history) {
            const historyText = data.difficulty_history.slice(-5).join(' -> ');
            document.getElementById('difficultyHistory').textContent = historyText || 'None';
        }
        
        this.updateUI();
    }

    updateUI() {
        // Update iteration progress
        document.getElementById('iteration').textContent = this.currentIteration;
        document.getElementById('maxIterations').textContent = this.maxIterations;
        
        const progress = (this.currentIteration / this.maxIterations) * 100;
        document.getElementById('progressFill').style.width = `${progress}%`;

        // Update stats
        document.getElementById('messageCount').textContent = this.messageCount;
        
        const avgLength = this.messageCount > 0 ? 
            Math.round(this.totalResponseLength / this.messageCount) : 0;
        document.getElementById('avgLength').textContent = avgLength;
    }
}

// Initialize the Conversational application
document.addEventListener('DOMContentLoaded', () => {
    new ConversationalCurriculumAgent();
});
