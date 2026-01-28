import requests
import logging

logger = logging.getLogger(__name__)

class OllamaLLMInterface:
    """Interface to Ollama local LLM"""
    
    def __init__(self, model_name: str = "qwen3:1.7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 512)
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Unable to generate response"
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "Error: Model unavailable"
    
    def calculate_logprob(self, response: str, prompt: str) -> float:
        """Calculate log probability (mock implementation)"""
        # In practice, this would require access to model's internal states
        # For now, return a reasonable mock value based on response length
        return -len(response) * 0.1  # Longer responses get lower probability
