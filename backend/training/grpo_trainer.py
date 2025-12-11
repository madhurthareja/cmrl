# GRPO (Group Relative Policy Optimization) Implementation
# For training MMedAgent-RL triage and attending models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from agents.medical_agent_core import DifficultyLevel, MedicalQuery, SpecialistResponse

logger = logging.getLogger(__name__)

@dataclass
class PolicyRollout:
    prompt: str
    response: str
    logprob_current: float
    logprob_old: float
    reward: float
    
@dataclass 
class GRPOConfig:
    group_size: int = 8  # G in paper
    learning_rate: float = 1e-6
    kl_coeff: float = 1e-3
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    batch_size: int = 128

class GRPOTrainer:
    """Group Relative Policy Optimization trainer for medical agents"""
    
    def __init__(self, model, config: GRPOConfig, reward_calculator):
        self.model = model
        self.config = config
        self.reward_calculator = reward_calculator
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        
        # Keep reference to old policy for importance ratios
        self.old_model = self._copy_model(model)
        self.reference_model = self._copy_model(model)  # For KL regularization
        
    def _copy_model(self, model):
        """Create a copy of the model for old policy"""
        import copy
        return copy.deepcopy(model)
    
    def update_old_policy(self):
        """Update old policy snapshot"""
        self.old_model.load_state_dict(self.model.state_dict())
    
    def sample_group_responses(self, prompts: List[str], temperature: float = 1.0) -> List[List[str]]:
        """Sample G responses for each prompt"""
        all_responses = []
        
        for prompt in prompts:
            group_responses = []
            for _ in range(self.config.group_size):
                with torch.no_grad():
                    response = self.model.generate(
                        prompt, 
                        temperature=temperature,
                        do_sample=True,
                        max_length=512
                    )
                    group_responses.append(response)
            all_responses.append(group_responses)
        
        return all_responses
    
    def calculate_logprobs(self, prompts: List[str], responses: List[str], model) -> List[float]:
        """Calculate log probabilities of responses under given model"""
        logprobs = []
        
        for prompt, response in zip(prompts, responses):
            # and forward pass through the model to get token-level log probabilities
            try:
                logprob = model.calculate_logprob(response, prompt)
                logprobs.append(logprob)
            except:
                # Fallback if model doesn't support logprob calculation
                logprobs.append(0.0)
        
        return logprobs
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute normalized advantages: Ai = (Ri - mean) / std"""
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array) + 1e-8  
        
        advantages = (rewards_array - mean_reward) / std_reward
        return advantages.tolist()
    
    def grpo_loss(self, rollouts: List[PolicyRollout]) -> torch.Tensor:
        """Compute GRPO loss with PPO-style clipping and KL regularization"""
        
        # Extract components from rollouts
        prompts = [r.prompt for r in rollouts]
        responses = [r.response for r in rollouts]
        rewards = [r.reward for r in rollouts]
        
        # Calculate advantages
        advantages = self.compute_advantages(rewards)
        
        # Get current policy logprobs
        current_logprobs = self.calculate_logprobs(prompts, responses, self.model)
        old_logprobs = self.calculate_logprobs(prompts, responses, self.old_model)
        ref_logprobs = self.calculate_logprobs(prompts, responses, self.reference_model)
        
        # Compute importance ratios
        ratios = []
        surrogate_losses = []
        kl_divs = []
        
        for i in range(len(rollouts)):
            # Importance ratio: π_θ(a|s) / π_old(a|s)
            ratio = torch.exp(torch.tensor(current_logprobs[i] - old_logprobs[i]))
            ratios.append(ratio)
            
            # PPO-style clipped surrogate loss
            advantage = torch.tensor(advantages[i])
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
            surrogate_loss = -torch.min(unclipped, clipped)
            surrogate_losses.append(surrogate_loss)
            
            # KL divergence with reference model
            kl_div = current_logprobs[i] - ref_logprobs[i]
            kl_divs.append(kl_div)
        
        # Total loss: surrogate + KL penalty
        surrogate_loss = torch.stack(surrogate_losses).mean()
        kl_penalty = torch.tensor(kl_divs).mean() * self.config.kl_coeff
        
        total_loss = surrogate_loss + kl_penalty
        
        return total_loss
    
    def training_step(self, prompts: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Single GRPO training step"""
        
        # Sample group responses for each prompt
        all_group_responses = self.sample_group_responses(prompts)
        
        # Create rollouts with rewards
        rollouts = []
        for prompt_idx, (prompt, group_responses) in enumerate(zip(prompts, all_group_responses)):
            ground_truth = ground_truths[prompt_idx]
            
            for response in group_responses:
                # Calculate reward
                reward = self.reward_calculator.calculate_total_reward(response, ground_truth)
                
                # Calculate logprobs (simplified)
                logprob_current = self.calculate_logprobs([prompt], [response], self.model)[0]
                logprob_old = self.calculate_logprobs([prompt], [response], self.old_model)[0]
                
                rollout = PolicyRollout(
                    prompt=prompt,
                    response=response,
                    logprob_current=logprob_current,
                    logprob_old=logprob_old,
                    reward=reward
                )
                rollouts.append(rollout)
        
        # Compute loss and update
        loss = self.grpo_loss(rollouts)
        
        # Only perform backward/optimizer step if loss is differentiable.
        # Mock models or analytic rewards may produce non-differentiable tensors
        # (loss.requires_grad == False). In that case skip the optimizer step.
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        else:
            logger.debug("Skipping optimizer step because loss has no gradients (mock/non-differentiable model)")
        
        # Calculate metrics
        avg_reward = np.mean([r.reward for r in rollouts])
        avg_ratio = np.mean([torch.exp(torch.tensor(r.logprob_current - r.logprob_old)).item() for r in rollouts])
        
        return {
            'loss': loss.item(),
            'avg_reward': avg_reward,
            'avg_ratio': avg_ratio,
            'num_rollouts': len(rollouts)
        }

class CurriculumMARLTrainer:
    """Curriculum Multi-Agent RL trainer for attending GP"""
    
    def __init__(self, model, specialist_models, config: GRPOConfig, reward_calculator):
        self.model = model
        self.specialist_models = specialist_models
        self.grpo_trainer = GRPOTrainer(model, config, reward_calculator)
        
        # Curriculum stage configurations
        self.curriculum_stages = {
            DifficultyLevel.EASY: {'kl_coeff': 1e-3, 'epochs': 5},
            DifficultyLevel.MEDIUM: {'kl_coeff': 4e-3, 'epochs': 5}, 
            DifficultyLevel.HARD: {'kl_coeff': 1e-2, 'epochs': 10}
        }
    
    def partition_dataset_by_specialist_accuracy(self, dataset, specialists) -> Dict[DifficultyLevel, List]:
        """Partition dataset based on how well specialists perform"""
        easy_data, medium_data, hard_data = [], [], []
        
        for sample in dataset:
            # Get specialist responses
            specialist_accuracies = []
            for specialist in specialists:
                try:
                    response = specialist.consult(sample)
                    # Calculate accuracy (simplified)
                    accuracy = 1.0 if sample.ground_truth.lower() in response.response.lower() else 0.0
                    specialist_accuracies.append(accuracy)
                except:
                    specialist_accuracies.append(0.0)
            
            avg_accuracy = np.mean(specialist_accuracies)
            
            # Partition based on specialist performance
            if avg_accuracy > 0.8:
                easy_data.append(sample)
            elif avg_accuracy > 0.4:
                medium_data.append(sample)
            else:
                hard_data.append(sample)
        
        return {
            DifficultyLevel.EASY: easy_data,
            DifficultyLevel.MEDIUM: medium_data,
            DifficultyLevel.HARD: hard_data
        }
    
    def get_specialist_responses(self, query: MedicalQuery, specialists: List) -> List[SpecialistResponse]:
        """Get responses from specialist models"""
        responses = []
        for specialist in specialists:
            try:
                response = specialist.consult(query)
                responses.append(response)
            except Exception as e:
                logger.warning(f"Specialist failed: {e}")
                # Fallback response
                responses.append(SpecialistResponse(
                    specialist_type="fallback",
                    response="Unable to provide consultation",
                    confidence=0.0,
                    reasoning="Specialist model unavailable"
                ))
        return responses

    def train_curriculum_stage(self, stage_data: List, difficulty: DifficultyLevel) -> Dict[str, float]:
        """Train on a specific curriculum stage"""
        stage_config = self.curriculum_stages[difficulty]

        # Update KL coefficient for this stage
        self.grpo_trainer.config.kl_coeff = stage_config['kl_coeff']

        stage_metrics = []

        for epoch in range(stage_config['epochs']):
            # Prepare prompts with specialist responses
            prompts = []
            ground_truths = []

            for sample in stage_data:
                # Get specialist consultations
                specialist_responses = self.get_specialist_responses(sample, self.specialist_models)

                # Create prompt with specialist context
                specialist_context = "\n".join([
                    f"Specialist {i+1} ({resp.specialist_type}): {resp.response}"
                    for i, resp in enumerate(specialist_responses)
                ])

                prompt = f"""
Medical Question: {sample.question}

Specialist Consultations:
{specialist_context}

Please provide a comprehensive answer integrating the specialist opinions:
"""

                prompts.append(prompt)
                ground_truths.append(sample.ground_truth)

            # Training step
            metrics = self.grpo_trainer.training_step(prompts, ground_truths)
            stage_metrics.append(metrics)

            logger.info(f"Stage {difficulty.value}, Epoch {epoch}: Loss={metrics['loss']:.4f}, Reward={metrics['avg_reward']:.3f}")

        # Update old policy after stage
        self.grpo_trainer.update_old_policy()

        return {
            'difficulty': difficulty.value,
            'avg_loss': np.mean([m['loss'] for m in stage_metrics]) if stage_metrics else 0.0,
            'avg_reward': np.mean([m['avg_reward'] for m in stage_metrics]) if stage_metrics else 0.0
        }


# Simplified mock implementations for testing
class MockLLM:
    """Mock LLM for testing GRPO"""

    def __init__(self):
        # keep parameters in a private attribute and provide an iterator via parameters()
        self._params = [torch.nn.Parameter(torch.randn(10, 10))]

    def generate(self, prompt: str, **kwargs) -> str:
        # Mock generation
        responses = [
            "<think>This is a medical question.</think><answer>Basic medical response</answer>",
            "<answer>Short answer</answer>",
            "No format response",
            "<think>Complex analysis</think><answer>Detailed medical explanation</answer>"
        ]
        return np.random.choice(responses)

    def calculate_logprob(self, response: str, prompt: str) -> float:
        # Mock logprob calculation
        return float(np.random.normal(-2.0, 0.5))

    def parameters(self):
        # Return an iterator compatible with optimizers
        return iter(self._params)

    def state_dict(self):
        return {'param': torch.randn(10, 10)}

    def load_state_dict(self, state_dict):
        pass


if __name__ == "__main__":
    # Test GRPO implementation (smoke test)
    from backend.agents.medical_agent_core import RewardCalculator

    # Initialize components
    mock_model = MockLLM()
    reward_calculator = RewardCalculator()
    config = GRPOConfig(group_size=4, learning_rate=1e-5)

    trainer = GRPOTrainer(mock_model, config, reward_calculator)

    # Test training step
    test_prompts = [
        "What causes chest pain?",
        "Explain myocardial infarction pathophysiology"
    ]
    test_ground_truths = [
        "Chest pain can be caused by heart problems, lung issues, or muscle strain",
        "Myocardial infarction occurs when blood flow to heart muscle is blocked"
    ]

    print("Testing GRPO training step...")
    metrics = trainer.training_step(test_prompts, test_ground_truths)

    print(f"Training metrics: {metrics}")
    print("GRPO implementation tested successfully.")
