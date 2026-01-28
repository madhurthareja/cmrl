import yaml
import json
import logging
from typing import List, Dict, Optional, Set, Any

logger = logging.getLogger(__name__)

class SymbolicEngine:
    """
    Finite-State Machine Engine for enforcing medical workflows.
    """
    def __init__(self, workflow_path: str):
        self.workflow_path = workflow_path
        self.states = self._load_yaml(f"{workflow_path}/states.yaml")['states']
        self.transitions = self._load_yaml(f"{workflow_path}/transitions.yaml")['allowed_transitions']
        try:
            self.rules = self._load_yaml(f"{workflow_path}/rules.yaml")['rules']
        except FileNotFoundError:
            logger.warning(f"No rules.yaml found at {workflow_path}, proceeding with NO validation rules.")
            self.rules = []
        
        self.current_state = self.states[0]  # Start at the first state
        logger.info(f"Symbolic Engine initialized. Start state: {self.current_state}")

    def _load_yaml(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_allowed_next_states(self, state: str) -> List[str]:
        return self.transitions.get(state, [])

    def is_valid_transition(self, current_state: str, next_state: str) -> bool:
        allowed = self.get_allowed_next_states(current_state)
        return next_state in allowed

    def validate_content(self, state: str, content: Any) -> Dict[str, Any]:
        """
        Check if the content generated for a specific state violates any rules.
        Returns:
            {
                "valid": bool,
                "violations": List[str] (reasons)
            }
        """
        violations = []
        
        # Ensure content is string for validation
        if isinstance(content, list):
            content_str = " ".join([str(x) for x in content])
        elif isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = str(content)
            
        lower_content = content_str.lower()
        
        state_rules = [r for r in self.rules if r['state'] == state]

        for rule in state_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'mandatory_content':
                # Check if at least one keyword is present (or all? usually at least one for 'topics')
                # Let's be strict: if it lists specific mandatory items, we check them.
                # For this implementation, let's say "keywords" represents a set where we need SOME coverage.
                # Actually, "keywords" in the yaml are specific concepts.
                
                # Simple check: define "missing" if none of the keywords appear? 
                # Or if specific ones are expected? 
                # Let's treat the list as "Must contain at least one of these to show addressed topic"
                found = any(kw.lower() in lower_content for kw in rule['keywords'])
                if not found:
                    violations.append(f"Rule '{rule['id']}': Missing required discussion on {rule['keywords']}")

            elif rule_type == 'forbidden_content':
                found_forbidden = [kw for kw in rule['keywords'] if kw.lower() in lower_content]
                if found_forbidden:
                    violations.append(f"Rule '{rule['id']}': Forbidden terms found: {found_forbidden}")

            elif rule_type == 'dependency':
                # Check for "if_missing" then "forbid"
                # Example: If missing "ECG", forbid "Catheterization"
                missing_prereqs = [req for req in rule.get('if_missing', []) if req.lower() not in lower_content]
                
                # If we are missing prereqs (meaning we didn't mention them), we must NOT mention the forbidden stuff
                if len(missing_prereqs) == len(rule.get('if_missing', [])): # Logic: if ANY or ALL? Let's say if we haven't done the prereq.
                    # Simplified: If the text doesn't mention ECG, it shouldn't mention Cath.
                    # Implementation: Check if forbidden terms exist
                    found_forbidden = [term for term in rule.get('forbid', []) if term.lower() in lower_content]
                    if found_forbidden:
                        violations.append(f"Rule '{rule['id']}': Cannot jump to {found_forbidden} without establishing {rule['if_missing']}")

        return {
            "valid": len(violations) == 0,
            "violations": violations
        }

    def transition(self, next_state: str):
        if self.is_valid_transition(self.current_state, next_state):
            self.current_state = next_state
            return True
        return False
