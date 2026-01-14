import logging
import time
import traceback
import pandas as pd
from datetime import datetime, timezone
from common import constants as CONST
from db.models.eval import Miner, MinerResponse, db
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from common.utils import rec_list_to_set
from evals.base_eval import BaseEval


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Evaluates prompt effectiveness for product recommendations.
checks if the ground truth SKU is in the LLM's recommended SKUs.

"""

class BitrecsPromptEval(BaseEval):

    min_sample_size = 3

    def __init__(self, run_id: str, miner_artifact: Artifact = None):      
        super().__init__(run_id, miner_artifact)
        holdout_df = self.get_latest_holdout()
        self.holdout_df = holdout_df
        logger.info(f"Loaded holdout set with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.min_sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_sample_size}")        
        self.debug_prompts = False

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_PROMPT_DAILY

    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the Bitrecs prompt evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0
        for idx, row in self.holdout_df.iterrows():
            if idx >= max_iterations:
                break
            try:               
                logger.info(f"\033[34mATTEMPT {idx+1}/{len(self.holdout_df)}...\033[0m")
                st = time.monotonic()
                ctx = row.get('context', '')
                logger.info(f"Context length: {len(str(ctx))} characters")
                eval_result = self.evaluate_row(row)                
                et = time.monotonic()
                duration = et - st

                logger.info(f"\033[32m Row {idx} evaluation took {duration:.2f}s \033[0m")
                if eval_result:
                    logger.info(f"\033[32m Row {idx} SUCCESS \033[0m")
                    success_count += 1
                else:
                    logger.info(f"\033[31m Row {idx} FAILURE \033[0m")
                    
                logger.info(f"Current score: {success_count}/{idx + 1} = {success_count / (idx + 1):.2f}")
            except Exception as e:
                error_message = traceback.format_exc()                
                logger.error(f"Error evaluating row {idx}: {e} \n{error_message}")
                exception_count += 1
                continue
            finally:
                 count += 1

        
        end_time = time.monotonic()
        total_duration = end_time - start_time
        #final_score = success_count / len(self.holdout_df)  # Standardized: float between 0 and 1
        final_score = success_count / count if count > 0 else 0.0  # Standardized: float between 0 and 1
        
        #eval_success = (success_count == len(self.holdout_df))
        eval_success = False
        if success_count >= max_iterations:
            eval_success = True

        eval_success = True

        result = EvalResult(           
            eval_name=self.get_eval_name(),  # Use base method
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {len(self.holdout_df)} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )        

        return result
    
    def evaluate_row(self, row: pd.Series) -> bool:
        """
        Evaluate a single row from the holdout set.
        """
        
        created_at = row.get('created_at', '')
        query = row.get('query', '')
        ground_truth_sku = row.get('ground_truth_sku', '')
        #provider = row.get('provider', '')
        provider = self.miner_artifact.provider
        winning_response = row.get('winning_response', '')
        context = row.get('context', '')

        query = str(query)
        ground_truth_sku = str(ground_truth_sku)
        provider = str(provider)        

        # Decode double-encoded JSON string
        products = self.decode_context(context)
        winning_products = self.decode_context(winning_response)
        num_recs = len(winning_products)

        prompt_factory = PromptFactory(
            miner_artifact=self.miner_artifact,
            sku=query,
            products=products,
            num_recs=num_recs,
            debug=self.debug_prompts
        )
        
        temperature = self.miner_artifact.sampling_params.temperature
        model = self.miner_artifact.model
        
        #provider = "OLLAMA_LOCAL"

        #model = "gemma3"
        #model = "qwen3-next:latest"
        #model = "qwen3:30b-a3b-instruct-2507-q4_K_M"
        #model = "mistral-nemo"
        #model = "qwen3-next:80b-a3b-instruct-q4_K_M"
        st = time.monotonic()
        system_prompt, user_prompt = prompt_factory.generate_prompt()
        server = LLM.try_parse(provider)
        llm_output = LLMFactory.query_llm(server=server,
                                            model=model,
                                            system_prompt=system_prompt,
                                            user_prompt=user_prompt,
                                            temp=temperature)
        recommended_skus = PromptFactory.tryparse_llm(llm_output)
        logger.info(f"LLM Output: {llm_output}")
        logger.info(f"Recommended SKUs: {recommended_skus}")
        et = time.monotonic()
        durtion = et - st

        rec_set = rec_list_to_set(recommended_skus)
        if ground_truth_sku in rec_set:
            logger.info(f"\033[32mGround truth SKU {ground_truth_sku} found in recommendations: {recommended_skus}\033[0m")
            result = True
        else:
            result = False      

        self.log_miner_response(
            run_id=self.run_id,
            query=query,
            num_recs=num_recs,
            recommended_skus=recommended_skus,
            duration=durtion
        )

        return result    
    

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