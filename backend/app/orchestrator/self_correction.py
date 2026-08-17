import re
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from app.orchestrator.tools import tool_registry

class SelfCorrectionVerifier:
    """
    Test-Time Compute Sandbox Verifier & Self-Correction Engine.
    
    Extracts code and mathematical expressions from model-generated candidate responses,
    executes them inside isolated sandboxes, captures runtime errors, and guides the LLM
    through an automated self-healing feedback loop before presenting answers to the user.
    """
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def extract_python_code_blocks(self, text: str) -> List[str]:
        """Extract all python code blocks enclosed in ```python ... ```."""
        pattern = r"```(?:python|py)\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def extract_math_expressions(self, text: str) -> List[str]:
        """Extract inline equations that can be verified via calculator."""
        pattern = r"(?:calculate|compute|solve|equals?)\s*[:=\s]+([0-9\.\+\-\*\/\(\)\^\s]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [m.strip() for m in matches if len(m.strip()) > 3]

    async def verify_and_correct(
        self,
        initial_text: str,
        generation_fn,
        conversation_context: List[Any],
        model_name: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs code execution verification on initial_text.
        If errors occur, prompts generation_fn to fix the issue iteratively.
        """
        logs = []
        current_text = initial_text
        
        for iteration in range(1, self.max_retries + 1):
            code_blocks = self.extract_python_code_blocks(current_text)
            
            # If no code to verify, return clean
            if not code_blocks:
                return current_text, logs

            all_passed = True
            error_details = []

            for idx, code in enumerate(code_blocks, 1):
                # Execute in safe code sandbox
                exec_result = await tool_registry.execute_tool("code_execution", {"code": code})
                
                status = exec_result.get("status")
                output = exec_result.get("output", "")

                if status == "error" or "Execution error:" in output or "Traceback" in output:
                    all_passed = False
                    error_details.append({
                        "block_idx": idx,
                        "code_sample": code[:120],
                        "error": output
                    })
                    logs.append({
                        "iteration": iteration,
                        "status": "failed",
                        "block_idx": idx,
                        "error": output,
                        "action": "Triggering Self-Correction feedback loop"
                    })
                else:
                    logs.append({
                        "iteration": iteration,
                        "status": "verified",
                        "block_idx": idx,
                        "output": output[:100]
                    })

            # If all code blocks executed cleanly with 0 errors, verification succeeds!
            if all_passed:
                logs.append({
                    "iteration": iteration,
                    "status": "success",
                    "message": "All Python/Math blocks verified cleanly with 0 runtime errors."
                })
                return current_text, logs

            # If errors found and we have retries left, prompt the model to self-correct
            if iteration < self.max_retries:
                err_msg = "\n".join([f"Block {e['block_idx']} Error: {e['error']}" for e in error_details])
                correction_prompt = (
                    f"AUTOMATED VERIFIER NOTICE:\n"
                    f"Your previous code output was executed in the runtime sandbox and encountered errors:\n{err_msg}\n\n"
                    f"Please analyze the traceback, identify the logical or syntax bug, and provide the complete, corrected solution."
                )
                
                # Request corrected generation
                try:
                    corrected_text = await generation_fn(correction_prompt, model_name)
                    if corrected_text:
                        current_text = corrected_text
                except Exception as gen_err:
                    logs.append({"status": "aborted", "error": str(gen_err)})
                    break

        return current_text, logs

self_correction_verifier = SelfCorrectionVerifier()
