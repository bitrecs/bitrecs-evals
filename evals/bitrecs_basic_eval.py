import re
import logging
import time
import traceback
import tiktoken
import langdetect
from datetime import datetime, timezone
from typing import Tuple
from common import constants as CONST
from evals.eval_result import EvalResult
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
check: ensures prompt lengths are within specified token limits.
check: ensures miner_hotkey is a valid S58 address.
data: N/A

"""

MAX_PROMPT_TOKENS = 10_000
MAX_SYSTEM_PROMPT_TOKENS = 5_000

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

# Variables that must only appear once in the template
VARIABLE_COUNT_RESTRICTIONS = {
    'product_catalog': 1,
    'order_json': 1,
    'cart_json': 1    
}



class BitrecsBasicEval(BaseEval):    

    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_BASIC_DAILY

    def run(self, sample_size = 10) -> EvalResult:
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
            hotkey_valid = BitrecsBasicEval.is_hotkey_valid_format(self.miner_artifact.miner_hotkey)
            if not hotkey_valid:
                reason = "miner_hotkey is not a valid S58 address"
                result = False
            system_language = BitrecsBasicEval.get_language_code(self.miner_artifact.system_prompt_template)
            user_language = BitrecsBasicEval.get_language_code(self.miner_artifact.user_prompt_template)
            if system_language != "en" or user_language != "en":
                reason = f"One or both prompts are not in English (system: {system_language}, user: {user_language})"
                result = False
            
            count = 1
        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")
            traceback.print_exc()
            exception_count += 1
        end_time = time.monotonic()
        total_duration = end_time - start_time

        eval_success = result
        if eval_success:
            template_status = "OK"            
            all_vars = set()
            all_vars.update(self.get_template_variables(self.miner_artifact.system_prompt_template))
            all_vars.update(self.get_template_variables(self.miner_artifact.user_prompt_template))
            valid_vars_used = all_vars & VALID_TEMPLATE_VARIABLES
            variable_count = len(valid_vars_used)
            max_vars = len(VALID_TEMPLATE_VARIABLES)  # 8
            variable_score = variable_count / max_vars if max_vars > 0 else 0.0
            final_score = 0.5 + (variable_score * 0.5)  # Base 0.5 for passing validation, plus up to 0.5 for variables
        else:
            final_score = 0.0

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
    
    @staticmethod
    def is_hotkey_valid_format(hotkey: str) -> bool:
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
    def get_language_code(text: str) -> str:
        try:
            lang = langdetect.detect(text)
            return lang
        except langdetect.lang_detect_exception.LangDetectException:
            return "unknown"
    
    @staticmethod    
    def validate_artifact_template(agent: Artifact) -> Tuple[bool, str]:
        if len(agent.name) == 0:
            return False, "name must not be empty"
        if len(agent.name) == 0 or len(agent.name) > 30:
            return False, "name must not be empty and must not exceed 30 characters"
        if not re.match(r'^[a-zA-Z0-9 ]{1,30}$', agent.name):
            return False, "name must be 1-30 alphanumeric characters only (letters and numbers, no special characters)"
        if agent.version_num <= 0:
            return False, "version_num must be greater than 0" 
        if len(agent.provider) == 0:
            return False, "provider must not be empty"
        if len(agent.model) == 0:
            return False, "model must not be empty"
        if len(agent.system_prompt_template) == 0:
            return False, "system_prompt_template must not be empty"
        if len(agent.user_prompt_template) == 0:
            return False, "user_prompt_template must not be empty"    
        if BitrecsBasicEval.get_token_count(agent.system_prompt_template) > MAX_SYSTEM_PROMPT_TOKENS:
            return False, "system_prompt_template exceeds maximum token count"
        if BitrecsBasicEval.get_token_count(agent.user_prompt_template) > MAX_PROMPT_TOKENS:
            return False, "user_prompt_template must not exceed maximum token count"
        
        if agent.status != 'screening_1':
            return False, "status must be 'screening_1' upon submission"
        
        if LLM.is_valid(agent.provider) == False:
            return False, f"provider '{agent.provider}' is not a supported LLM provider"
        
        provider = LLM.try_parse(agent.provider)
        ALLOWED_PROVIDERS = [LLM.CHUTES, LLM.OPEN_ROUTER]
        if provider not in ALLOWED_PROVIDERS:
            return False, f"provider '{agent.provider}' is currently not supported. Supported providers are: {', '.join(ALLOWED_PROVIDERS)}"
        
        if ":free" in agent.model.lower():
            return False, "Free models are not supported"
        
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
        variable_counts = {var: 0 for var in VARIABLE_COUNT_RESTRICTIONS}
        for template_str, template_name in [
            (agent.system_prompt_template, "system_prompt_template"),
            (agent.user_prompt_template, "user_prompt_template")
        ]:
            try:
                ast = env.parse(template_str)
                variables_used = set()
                for node in ast.find_all(nodes.Name):
                    variables_used.add(node.name)
                    if node.name in VARIABLE_COUNT_RESTRICTIONS:
                        variable_counts[node.name] += 1

                invalid_vars = variables_used - VALID_TEMPLATE_VARIABLES
                if invalid_vars:
                    return False, (
                        f"{template_name} contains invalid variable(s): {', '.join(invalid_vars)}. "
                        f"Allowed variables are: {', '.join(sorted(VALID_TEMPLATE_VARIABLES))}"
                    )

                matched_vars.update(variables_used)
            except Exception as e:
                return False, f"Error parsing variables in {template_name}: {e}"

        # Check variable count restrictions
        for var, max_count in VARIABLE_COUNT_RESTRICTIONS.items():
            if variable_counts[var] > max_count:
                return False, (
                    f"Variable '{var}' appears {variable_counts[var]} times, "
                    f"but may only appear {max_count} time(s) in the templates."
                )

        if len(matched_vars) == 0:
            return False, "No valid template variables found in either prompt template"
        
        logger.info(f"\033[32mTemplate validation successful. Used variables: {', '.join(sorted(matched_vars))} \033[0m")
        return True, f"Valid template. Used variables: {', '.join(sorted(matched_vars))}"  
   