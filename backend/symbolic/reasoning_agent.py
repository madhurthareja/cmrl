import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from symbolic.engine import SymbolicEngine
from agents.llm_interface import OllamaLLMInterface

logger = logging.getLogger(__name__)

class SymbolicReasoningAgent:
    """
    Agent that reasons within the constraints of a SymbolicEngine.
    """
    def __init__(self, workflow_path: str, model_name: str = "qwen3:1.7b", retriever = None):
        self.engine = SymbolicEngine(workflow_path)
        self.llm = OllamaLLMInterface(model_name=model_name)
        self.retriever = retriever
        
    def _construct_prompt(self, state: str, context: str = "", frozen_diagnosis: Optional[str] = None) -> str:
        """
        Constructs the strict prompt for the specific state.
        """
        prompt = f"""You are a medical reasoning module operating downstream of a high-precision VQA system.
        
Current state: {state}

DIAGNOSTIC AUTHORITY CONTRACT:
1. VISUAL SUPREMACY: The "Visual Findings" provided in the context are OBSERVATIONAL FACTS. You do not have independent vision. You must ACCEPT these findings as the ground truth.
2. NO RE-LITIGATION: Do not output phrases like "suggestive of", "possible", or "cannot rule out" when referring to the primary visual finding. It IS present.
3. CONSTRAINED DIFFERENTIALS: Your differential diagnosis must be a subset of the visual findings, not a superset of general possibilities. Do not re-introduce low-probability alternatives that the VQA has already implicitly filtered.
4. MANAGEMENT FOCUS: Your job is to reason about *consequences* and *management* of the diagnosed condition, not to question the diagnosis itself.

Output Discipline & Diagnostic Commitment Guidelines:
1. Single Primary Finding: Always state one most likely diagnosis when imaging evidence is sufficient.
2. Hierarchical Differentials: List alternatives only as secondary and explicitly lower probability.
3. Stage-Dependent Certainty:
   - Descriptive sections: neutral, observational
   - Differential generation: broad
   - Diagnostic planning: narrow decisively
   - Treatment / disposition: conditional on the primary diagnosis
4. Action Requires Commitment: If management or disposition is discussed, uncertainty must be resolved or branched.
5. Avoid Defensive Redundancy: Do not repeat identical diagnostic phrases across sections.
6. Escalation Thresholds: Recommend advanced imaging or referral only with justification.
7. Safety Without Silence: Prefer calibrated confidence over maximal hedging.

Your task:
- Produce reasoning ONLY for this state.
- Do NOT jump ahead to future states.
- Output valid JSON only.
- Ensure all strings in the JSON are properly escaped.

Context content if any:
{context}

"""
        if state == "HISTORY_OF_PRESENT_ILLNESS":
            prompt += """
IMPORTANT: Your analysis must include characterization of the symptoms or findings (onset, duration, description, or visual findings).
"""
        elif state == "MEDICAL_HISTORY_AND_RISK":
            prompt += """
IMPORTANT: You MUST explicitly review the patient's past medical history and potential risk factors. If not provided, state that they are unknown.
"""
        elif state == "PHYSICAL_EXAM":
             prompt += """
IMPORTANT: You MUST explicitly mention 'vital signs', 'physical examination', 'signs', or 'observations' in your text. If not available, state they are not reported.
"""
        elif state == "DIFFERENTIAL_GENERATION":
            prompt += """
IMPORTANT: You MUST explicitly list differential diagnoses or potential causes being considered.
"""
        elif state == "DISPOSITION":
            prompt += """
IMPORTANT: Provide the final disposition (e.g., Admit, Discharge, Refer) and a summary plan. Keep your content concise.
"""
        
        if frozen_diagnosis:
            prompt += f"""
DIAGNOSTIC FINALITY PROTOCOL (ACTIVE):
The diagnosis was established in the previous stage and is now FROZEN.
ESTABLISHED DIAGNOSIS: {frozen_diagnosis}

CRITICAL RULES:
1. You must NOT re-evaluate probabilities, re-rank, or question this diagnosis.
2. You must NOT introduce new differential diagnoses.
3. Your task is ONLY to provide management, treatment, or disposition based on this established fact.
4. Any "prediction" you make must match the established diagnosis exactly.
"""

        prompt += f"""Schema:
{{
  "state": "{state}",
  "content": "<your reasoning and findings here>",
  "confidence": <float 0.0-1.0>,
  "next_recommended_action": "<short phrase>"
}}
"""
        return prompt

    async def step(self, context: str = "", frozen_diagnosis: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes one step of the reasoning process for the current state.
        """
        current_state = self.engine.current_state
        logger.info(f"Step executing for state: {current_state}")
        
        # Retrieval with intent (Step 5)
        # OPTIMIZATION: Only fetch external knowledge for knowledge-intensive states
        # to reduce latency and redundant API calls.
        retrieved_context = ""
        knowledge_heavy_states = ["DIFFERENTIAL_GENERATION", "DIAGNOSTIC_PLANNING", "INITIAL_TREATMENT"]
        
        if self.retriever and current_state in knowledge_heavy_states:
            # We use the current context/patient info to query the knowledge base
            # We pass the full context so the retriever can extract relevant medical keywords
            query = context
            
            docs = self.retriever.retrieve(query, current_state)
            if docs:
                retrieved_context = "\nRelevant Guidelines/Literature:\n"
                for d in docs:
                    retrieved_context += f"- {d.get('title')}: {d.get('abstract')[:200]}...\n"

        full_context = context + "\n" + retrieved_context

        prompt = self._construct_prompt(current_state, full_context, frozen_diagnosis)
        
        # Call LLM
        response_text = await self.llm.generate(prompt, temperature=0.2, max_tokens=1024)
        
        # Parse JSON
        try:
            # Robust cleanup for json wrapped in markdown
            cleaned_response = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```" in cleaned_response:
                # regex or simple split to extract content between first ```(json)? and last ```
                import re
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(1)
                else:
                    # Fallback: remove just the markers
                    cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
            
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response_text}")
            return {
                "state": current_state,
                "status": "error",
                "message": "LLM failed to produce valid JSON",
                "raw_response": response_text
            }
            
        # Validate Rule Content
        validation = self.engine.validate_content(current_state, data.get('content', ''))
        
        if not validation['valid']:
            logger.warning(f"Validation failed for state {current_state}: {validation['violations']}")
            return {
                "status": "violation",
                "state": current_state,
                "violations": validation['violations'],
                "content": data.get('content')
            }
            
        # Result for this step
        result = {
            "status": "success",
            "state": current_state,
            "content": data.get('content'),
            "confidence": data.get('confidence'),
            "next_action": data.get('next_recommended_action')
        }
        
        return result

    def advance(self, next_state: str) -> bool:
        """
        Manually advance the state machine if valid.
        """
        success = self.engine.transition(next_state)
        if success:
            logger.info(f"Transitioned to {next_state}")
        else:
            logger.error(f"Invalid transition from {self.engine.current_state} to {next_state}")
        return success

    async def run_full_workflow(self, initial_context: str, max_steps: int = 15):
        """
        Run the agent through the steps.
        For automation, we might need a controller to pick the next state.
        For now, we just pick the first available allowed transition?
        Or we stick to the linear path if only one exists.
        """
        history = []
        step_count = 0
        
        # Accumulate context as we progress
        current_context = initial_context
        frozen_diagnosis = None
        
        while self.engine.current_state != "END" and step_count < max_steps:
            step_result = await self.step(context=current_context, frozen_diagnosis=frozen_diagnosis)
            history.append(step_result)
            
            if step_result['status'] == 'success':
                # FREEZE DIAGNOSIS after DIAGNOSTIC_PLANNING
                if self.engine.current_state == "DIAGNOSTIC_PLANNING":
                    frozen_diagnosis = step_result.get('content')
                    logger.info(f"Diagnosis FROZEN: {frozen_diagnosis[:50]}...")

                # Append the agent's reasoning to the context for the next step
                current_context += f"\n\n[Determined at {self.engine.current_state}]: {step_result.get('content')}"
                
                # Determine next state
                # Strategy: For this workflow, it is mostly linear.
                # We check allowed next states.
                allowed = self.engine.get_allowed_next_states(self.engine.current_state)
                if not allowed:
                    logger.info("No further transitions possible.")
                    break
                
                # Logic: Just take the first one for now (Linear assumption)
                # In a real agent, the LLM might suggest the next state, or we evaluate criteria.
                next_st = allowed[0]
                self.advance(next_st)
            elif step_result['status'] == 'violation':
                # In real RL, we'd penalize and retry.
                # Here, we stop or retry. Let's stop to show the constraint working.
                logger.info("Stopping due to rule violation.")
                break
            
            step_count += 1
            
        return history
