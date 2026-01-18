
import time
import json
import traceback
from typing import List
import pandas as pd
import logging
from datetime import datetime, timezone
from common import constants as CONST
from common.utils import rec_list_to_set
from db.models.eval import Miner, MinerResponse, db
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from datasets import load_dataset


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Amazon Prompt 500 Evaluation
check: evaluates prompts on Amazon dataset for recommendation accuracy.
data: Amazon recommendation dataset (500)

"""

class AmazonPromptEval500(BaseEval):

    min_sample_size = 3
    pass_threshold = 0.3

    def __init__(self, run_id: str, miner_artifact: Artifact = None):
        super().__init__(run_id, miner_artifact)
        
        sample_data = self.sample_dataset(sample_size=self.
                                          min_sample_size)
        self.holdout_df = pd.DataFrame(sample_data)
        logger.info(f"Loaded holdout set with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.min_sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_sample_size}")            


    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.AMAZON_PROMPT_100    
    

    def sample_dataset(self, sample_size = 5) -> List:        
        #ds = load_dataset("reallybigmouse4/dense_core_amazon2023", split="train", streaming=True)
        ds = load_dataset("reallybigmouse4/fashion-eval-recsys", split="train", streaming=True)        
        small_sample = ds.take(sample_size)        
        small_list = list(small_sample)        
        #pretty_json = json.dumps(small_list, indent=2)
        #print(pretty_json)
        count = 0
        for thing in small_list:
            #print(thing)
            count += 1

        assert count == len(small_list)

        return small_list

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
        final_score = success_count / count if count > 0 else 0.0                
        eval_success = False
        if final_score >= self.pass_threshold:
            eval_success = True     

        result = EvalResult(           
            eval_name=self.get_eval_name(),
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
            num_recs=num_recs            
        )
        
        temperature = self.miner_artifact.sampling_params.temperature
        model = self.miner_artifact.model        
       
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
   