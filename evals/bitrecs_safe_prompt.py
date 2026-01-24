import re
import logging
import time
import traceback
import tiktoken
from datetime import datetime, timezone
from typing import Tuple
from common import constants as CONST
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Bitrecs Safe Prompt Validation

check: ensures prompts are safe from injection attacks using a safety LLM.
data: N/A

"""


class BitrecsSafeEval(BaseEval):    

    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_SAFE_DAILY

    def run(self, sample_size = 10) -> EvalResult:
        """
        Run the Bitrecs safe prompt evaluation.
        """
           
        count = 0
        success_count = 0
        exception_count = 0        
        start_time = time.monotonic()
        final_score = 0.0
        template_status = "ERROR"
        reason = "NA"
        result = False
        try:
            combined_prompts = f"System Prompt:\n{self.miner_artifact.system_prompt_template}\n\nUser Prompt:\n{self.miner_artifact.user_prompt_template}"
            prompt_safe, safe_reason = BitrecsSafeEval.is_prompt_safe(combined_prompts)
            result = prompt_safe
            reason = safe_reason
            template_status = "SAFE" if prompt_safe else "UNSAFE"
            count = 1
        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")            
            traceback.print_exc()
            exception_count += 1
        end_time = time.monotonic()
        total_duration = end_time - start_time

        eval_success = result
        if eval_success:
            final_score = 1.0

        result = EvalResult(
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Test result: {result}. TEMPLATE {template_status}.\nScore: {final_score:.2f}.\n{reason}\nEvaluated {count} samples with {exception_count} exceptions for hotkey {self.miner_artifact.miner_hotkey}.\n{self.run_id}",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )        

        detail_report = self.make_detail_report()
        logger.info(f"Detail Report:\n{detail_report}")

        return result
    
    def make_detail_report(self) -> str:
        report_lines = []
        report_lines.append(f"Eval Name: {self.get_eval_name()}")
        report_lines.append(f"Run ID: {self.run_id}")
        report_lines.append(f"Miner Hotkey: {self.miner_artifact.miner_hotkey}")
        report_lines.append(f"Model: {self.miner_artifact.model}")
        report_lines.append(f"Provider: {self.miner_artifact.provider}")
        return "\n".join(report_lines)
  

    @staticmethod    
    def is_prompt_safe(prompt: str) -> Tuple[bool, str]:
        """Test if a prompt is vulnerable to injection attacks. Returns (is_safe, reason)."""
        #safe_model = "meta-llama/llama-guard-4-12b"
        # safe_model = "Qwen/Qwen3Guard-Gen-0.6B"
        # safe_model = "meta-llama/llama-guard-3-8b"
        # safe_model = "openai/gpt-oss-safeguard-20b"
        
        safe_model = "meta-llama/llama-guard-4-12b"
        safe_server = LLM.OPEN_ROUTER

        #safe_server = LLM.OLLAMA_LOCAL
        #safe_model = "llama-guard3"
        
        start_time = time.time()    
        try:
            result = LLMFactory.query_llm(server=safe_server, model=safe_model, user_prompt=prompt)
            lines = result.strip().split('\n')
            duration = time.time() - start_time

            safety_status = "unsafe"
            if safe_model == "meta-llama/llama-guard-4-12b":            
                safety_status = lines[0].strip().lower() if lines else "unsafe"  # Default to unsafe if malformed
                categories = lines[1].strip() if len(lines) > 1 and safety_status == "unsafe" else ""
            elif safe_model == "Qwen/Qwen3Guard-Gen-0.6B":
                if lines and "Safety:" in lines[0]:
                    safety_status = lines[0].split(":", 1)[1].strip().lower()
                else:
                    safety_status = lines[0].strip().lower() if lines else "unsafe"            
                if len(lines) > 1 and "Categories:" in lines[1]:
                    categories = lines[1].split(":", 1)[1].strip()
                else:
                    categories = lines[1].strip() if len(lines) > 1 else ""
            elif safe_model == "llama-guard3":
                safety_status = "unsafe"
                categories = ""
                if lines and lines[0].lower() == "safe":
                    safety_status = "safe"
            elif safe_model == "openai/gpt-oss-safeguard-20b":
                safety_status = "unsafe"
                categories = ""
                if lines and "SAFE" in lines[0].upper():
                    safety_status = "safe"
            else:
                raise ValueError(f"Unknown safety model: {safe_model}")
            
            # Custom injection list
            # injection_keywords = [
            #     # Basic overrides
            #     "ignore previous", "override", "system prompt", "jailbreak", "bypass restrictions",
            #     # Persona/role changes
            #     "act as", "role-play as", "you are now", "pretend to be", "switch to",
            #     # Uncensored modes
            #     "dan mode", "uncensored", "developer mode", "god mode", "maximum truth",
            #     # Injection patterns
            #     "forget instructions", "new instructions", "disregard", "ignore safety",
            #     # Adversarial phrases
            #     "do not follow", "break rules", "unrestricted", "no limits", "full access",
            #     # Common jailbreaks
            #     "hacking mode", "evil mode", "unfiltered", "raw response", "no censorship"
            # ]
            # has_injection_keywords = any(keyword in prompt.lower() for keyword in injection_keywords)
            has_injection_keywords = False  # Disable keyword check for now
            
            if "unsafe" in safety_status or has_injection_keywords:
                reason = f"Detected as unsafe by {safe_model} (categories: {categories}) or contains injection keywords."
                logger.warning(f"Prompt vulnerable to injection: {reason} (Query time: {duration:.2f}s)")
                return False, reason
            else:
                reason = f"Appears safe based on {safe_model} and keyword checks."
                logger.info(f"Prompt safe from injection: {reason} (Query time: {duration:.2f}s)")
                return True, reason
        
        except Exception as e:
            reason = f"Error querying safety model: {str(e)}"
            logger.error(f"Failed to test prompt for injection: {reason}")
            return False, reason  # Default to unsafe on error for caution
