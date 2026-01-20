import json
import re
import logging
import time
import traceback
import tiktoken
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple
from common import constants as CONST
from db.models.eval import Miner, MinerResponse, db
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from jinja2 import Template, TemplateSyntaxError, Environment, nodes

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Bitrecs Basic Template Validation

check: ensures templates are valid Jinja2 templates and only use allowed variables.
data: N/A

"""
MIN_PROMPT_TOKENS = 100
MAX_PROMPT_TOKENS = 50_000
MAX_SYSTEM_PROMPT_TOKENS = 10_000

VALID_TEMPLATE_VARIABLES = {
    'current_date',
    'product_catalog',
    'sku',
    'sku_info',
    'num_recs',    
    'persona',
    'cart_json',
    'order_json'
}

class BitrecsBasicEval(BaseEval):    

    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_BASIC_DAILY

    def run(self, sample_size=10) -> EvalResult:
        """
        Run the Bitrecs prompt evaluation.
        """
           
        count = 0
        success_count = 0
        exception_count = 0
        result = False
        start_time = time.monotonic()
        final_score = 0.0
        template_status = "ERROR"
        reason = "NA"
        try:
            result, reason = self.validate_template()
            hotkey_valid = BitrecsBasicEval.is_hotkey_valid(self.miner_artifact.miner_hotkey)
            if not hotkey_valid:
                reason = "miner_hotkey is not a valid S58 address"
                result = False

            if result:
                system_prompt_safe, system_prompt_safety_reason = BitrecsBasicEval.is_prompt_safe(self.miner_artifact.system_prompt_template)   
                if not system_prompt_safe:
                    result = False
                    reason += f" | System Prompt Safety Check Failed: {system_prompt_safety_reason}"
                else:
                    reason += f" | System Prompt Safety Check Passed."
            if result:
                user_prompt_safe, user_prompt_safety_reason = BitrecsBasicEval.is_prompt_safe(self.miner_artifact.user_prompt_template)   
                if not user_prompt_safe:
                    result = False
                    reason += f" | User Prompt Safety Check Failed: {user_prompt_safety_reason}"
                else:
                    reason += f" | User Prompt Safety Check Passed."

        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")
            traceback.print_exc()
            exception_count += 1
        end_time = time.monotonic()
        total_duration = end_time - start_time

        eval_success = result
        if eval_success:
            template_status = "OK"
            # Add variance based on variable usage count
            all_vars = set()
            all_vars.update(self.get_template_variables(self.miner_artifact.system_prompt_template))
            all_vars.update(self.get_template_variables(self.miner_artifact.user_prompt_template))
            valid_vars_used = all_vars & VALID_TEMPLATE_VARIABLES
            variable_count = len(valid_vars_used)
            max_vars = len(VALID_TEMPLATE_VARIABLES)  # 8
            variable_score = variable_count / max_vars if max_vars > 0 else 0.0
            final_score = 0.5 + (variable_score * 0.5)  # Base 0.5 for passing validation, plus up to 0.5 for variables
        else:
            final_score = 0.0  # Fail validation = 0.0            

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
    
    def validate_template(self) -> Tuple[bool, str]:
        validated, reason = BitrecsBasicEval.validate_artifact_template(self.miner_artifact)
        logger.info(f"Template validation result: {validated}, Reason: {reason}")        
        return validated, reason

    def get_eval_name(self) -> str:
        this_type = self.eval_type()
        name = str(this_type)
        return name
    
    @staticmethod
    def is_hotkey_valid(hotkey: str) -> bool:
        if not isinstance(hotkey, str) or len(hotkey) != 48:
            return False
        # regex s58 address
        pattern = r"^5[1-9A-HJ-NP-Za-km-z]{47}$"
        if re.match(pattern, hotkey):
            return True
        return False

    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    @staticmethod
    def get_template_variables(template_str: str) -> set:
        env = Environment()
        ast = env.parse(template_str)
        variables = set()
        for node in ast.find_all(nodes.Name):
            variables.add(node.name)
        return variables
    
    @staticmethod
    def validate_artifact_template(agent: Artifact) -> Tuple[bool, str]:    
        # if agent.agent_id is not None:
        #     return False, "agent_id must not be set by the client" 
        
        if len(agent.miner_hotkey) == 0:
            return False, "miner_hotkey must not be empty"
        # if len(agent.name) == 0:
        #     return False, "name must not be empty"
        # if agent.version_num <= 0:
        #     return False, "version_num must be greater than 0"
        if agent.miner_uid <= 0:
            return False, "miner_uid must be greater than 0"
        if len(agent.provider) == 0:
            return False, "provider must not be empty"
        if len(agent.model) == 0:
            return False, "model must not be empty"
        if len(agent.system_prompt_template) == 0:
            return False, "system_prompt_template must not be empty"
        if len(agent.user_prompt_template) == 0:
            return False, "user_prompt_template must not be empty"    
        if BitrecsBasicEval.get_token_count(agent.system_prompt_template) > 10_000:
            return False, "system_prompt_template exceeds maximum token count"
        if BitrecsBasicEval.get_token_count(agent.user_prompt_template) > 100_000:
            return False, "user_prompt_template must not exceed maximum token count"
        
        if BitrecsBasicEval.get_token_count(agent.user_prompt_template) < MIN_PROMPT_TOKENS:
            return False, "user_prompt_template is too short to be valid"
        
        # if agent.status != 'screening_1':
        #     return False, "status must be 'screening_1' upon submission"

        if LLM.is_valid(agent.provider) == False:
            return False, f"provider '{agent.provider}' is not a valid LLM provider"
        
        try:
            Template(agent.system_prompt_template)
        except TemplateSyntaxError as e:
            return False, f"system_prompt_template is not a valid Jinja2 template: {e}"
        try:
            Template(agent.user_prompt_template)
        except TemplateSyntaxError as e:
            return False, f"user_prompt_template is not a valid Jinja2 template: {e}"
        
        env = Environment()
        matched_vars = set()    
        for template_str, template_name in [(agent.system_prompt_template, "system_prompt_template"), (agent.user_prompt_template, "user_prompt_template")]:
            try:
                ast = env.parse(template_str)
                variables_used = set()
                for node in ast.find_all(nodes.Name):
                    variables_used.add(node.name)            
                
                invalid_vars = variables_used - VALID_TEMPLATE_VARIABLES
                if invalid_vars:
                    return False, f"{template_name} contains invalid variable(s): {', '.join(invalid_vars)}. Allowed variables are: {', '.join(sorted(VALID_TEMPLATE_VARIABLES))}"
                
                matched_vars.update(variables_used)
            except Exception as e:
                return False, f"Error parsing variables in {template_name}: {e}"

        if len(matched_vars) == 0:
            return False, "No valid template variables found in either prompt template"
        
        logger.info(f"\033[32mTemplate validation successful. Used variables: {', '.join(sorted(matched_vars))} \033[0m")
        return True, f"Valid template. Used variables: {', '.join(sorted(matched_vars))}"
    

    @staticmethod    
    def is_prompt_safe(prompt: str) -> Tuple[bool, str]:
        """Test if a prompt is vulnerable to injection attacks. Returns (is_safe, reason)."""
        #safe_model = "meta-llama/llama-guard-4-12b"
        safe_model = "Qwen/Qwen3Guard-Gen-0.6B"
        safe_server = LLM.CHUTES
        
        start_time = time.time()    
        try:
            result = LLMFactory.query_llm(server=safe_server, model=safe_model, user_prompt=prompt)
            lines = result.strip().split('\n')
            duration = time.time() - start_time
            
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
            else:
                raise ValueError(f"Unknown safety model: {safe_model}")
            
            # Expanded injection keywords (sourced from SecLists, OpenAI, and community lists)
            injection_keywords = [
                # Basic overrides
                "ignore previous", "override", "system prompt", "jailbreak", "bypass restrictions",
                # Persona/role changes
                "act as", "role-play as", "you are now", "pretend to be", "switch to",
                # Uncensored modes
                "dan mode", "uncensored", "developer mode", "god mode", "maximum truth",
                # Injection patterns
                "forget instructions", "new instructions", "disregard", "ignore safety",
                # Adversarial phrases
                "do not follow", "break rules", "unrestricted", "no limits", "full access",
                # Common jailbreaks
                "hacking mode", "evil mode", "unfiltered", "raw response", "no censorship"
            ]
            has_injection_keywords = any(keyword in prompt.lower() for keyword in injection_keywords)
            
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
