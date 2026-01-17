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
from llm.llm_provider import LLM
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from jinja2 import Template, TemplateSyntaxError, Environment, nodes

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Evaluate template schema validity for Bitrecs artifacts.

"""

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

    min_row_count = 3

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
        try:
            result = self.validate_template()    
        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")
            traceback.print_exc()
            exception_count += 1
        end_time = time.monotonic()
        total_duration = end_time - start_time

        eval_success = result
        if eval_success:
            final_score = 1.0        
            template_status = "OK"

        result = EvalResult(           
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Test result: {result}. TEMPLATE {template_status}.\nEvaluated {count} samples with {exception_count} exceptions for hotkey {self.miner_artifact.miner_hotkey}.\n{self.run_id}",
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
    
    def validate_template(self) -> bool:
        validated, reason = BitrecsBasicEval.validate_artifact_template(self.miner_artifact)
        logger.info(f"Template validation result: {validated}, Reason: {reason}")        
        return validated

    def get_eval_name(self) -> str:
        this_type = self.eval_type()
        name = str(this_type)
        return name

    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
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