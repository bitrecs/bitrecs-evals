import logging
import time
import tiktoken
import pandas as pd

from datetime import datetime, timezone
from typing import Tuple
from common import constants as CONST
from db.models.eval import Miner, MinerResponse, db
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from jinja2 import Template, TemplateSyntaxError, Environment, nodes

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Basic Bitrecs Evaluation Class
-run validate_template

"""

class BitrecsBasicEval(BaseEval):

    min_row_count = 3

    def __init__(self, run_id: str, miner_artifact: Artifact = None):      
        super().__init__(run_id, miner_artifact)

        self.this_artifact = miner_artifact
        if not self.this_artifact:
            raise ValueError("Miner artifact is required for basic evaluation.")        


    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_BASIC_DAILY

    def run(self, sample_size=10) -> EvalResult:
        """
        Run the Bitrecs prompt evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0       


        self.validate_template()
        
        end_time = time.monotonic()
        total_duration = end_time - start_time        
        final_score = success_count / count if count > 0 else 0.0
        eval_success = False


        result = EvalResult(           
            eval_name=self.get_eval_name(),  # Use base method
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"AMAZON TESTS RESULT DETAILS",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )        

        return result
    
    def validate_template(self) -> bool:


        return False
    

    def log_miner_response(self, run_id: str, query: str, num_recs: int, recommended_skus: list, duration: float):
        """
        Log the miner response to the database.
        """
        try:
            db.connect()
            db.create_tables([Miner, MinerResponse], safe=True)  # Ensure tables exist

            # Get or create Miner
            miner, created = Miner.get_or_create(hotkey=self.miner_artifact.miner_hotkey)

            # Create MinerResponse record
            MinerResponse.create(
                run_id=run_id,
                miner=miner,
                hotkey=self.miner_artifact.miner_hotkey,
                query=query,
                num_recs=num_recs,
                response=str(recommended_skus),
                model_name=self.miner_artifact.model,
                provider_name=self.miner_artifact.provider,
                temperature=self.miner_artifact.sampling_params.temperature,
                duration_seconds=duration         
            )
            logger.info("Miner response logged to DB.")
        except Exception as e:
            logger.error(f"Failed to log miner response to DB: {e}")
        finally:
            db.close()


    def get_eval_name(self) -> str:
        return "Bitrecs Basic Daily Eval"    
    

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
        if len(agent.name) == 0:
            return False, "name must not be empty"
        if agent.version_num <= 0:
            return False, "version_num must be greater than 0"
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
        
        if agent.status != 'screening_1':
            return False, "status must be 'screening_1' upon submission"
        
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