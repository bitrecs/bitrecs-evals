import time
import math
import traceback
import pandas as pd
import logging
from datetime import datetime, timezone
from common import constants as CONST
from common.hf_utils import sample_dataset
from common.utils import rec_list_to_set
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.amazon_size import AmazonDatasetSize
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from models.product import Product

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

class AmazonAllBeautyNDCG1000(BaseEval):
    def __init__(self, run_id: str, miner_artifact: Artifact = None):
        super().__init__(run_id, miner_artifact)
        size = AmazonDatasetSize(1000)
        folder_name = "All_Beauty"
        # dataset format: amazon_all_beauty_universe_{universe_size}.csv
        sample_data = sample_dataset(folder_name=folder_name, size=size.value, sample_size=20)
        self.holdout_df = sample_data
        logger.info(f"Loaded All_Beauty_1000 holdout set with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.sample_size}")

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.AMAZON_ALL_BEAUTY_1000

    def calculate_ndcg_score(self, rank_index: int, k=10) -> float:
        """Calculates rank-aware score. index is 0-indexed position."""
        if rank_index is None or rank_index >= k:
            return 0.0
        return 1.0 / math.log2(rank_index + 2)

    def run(self, max_iterations=10) -> EvalResult:
        start_time = time.monotonic()        
        total_ndcg = 0.0
        count = 0
        success_count = 0 # Not strictly used for NDCG score but good for logging
        exception_count = 0
        
        for idx, row in self.holdout_df.iterrows():
            if idx >= max_iterations:
                break
            try:               
                logger.info(f"\033[34mATTEMPT {idx+1}/{len(self.holdout_df)}...\033[0m")
                
                st = time.monotonic()
                rank_index = self.evaluate_row(row)
                et = time.monotonic()
                duration = et - st
                
                ndcg_score = self.calculate_ndcg_score(rank_index)
                total_ndcg += ndcg_score
                
                logger.info(f"\033[32m Row {idx} evaluation took {duration:.2f}s, NDCG Score: {ndcg_score:.4f} \033[0m")
                
                if ndcg_score > 0:
                    success_count += 1
                    
                logger.info(f"Current Avg NDCG: {total_ndcg / (idx + 1):.4f}")
                
            except Exception as e:
                error_message = traceback.format_exc()                
                logger.error(f"Error evaluating row {idx}: {e} \n{error_message}")
                exception_count += 1
                continue
            finally:
                 count += 1

        end_time = time.monotonic()
        total_duration = end_time - start_time        
        final_score = total_ndcg / count if count > 0 else 0.0                
        
        # Pass threshold logic (optional, keeping it consistent with BaseEval expectations)
        eval_success = final_score >= self.pass_threshold

        result = EvalResult(           
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {len(self.holdout_df)} rows with {exception_count} exceptions (max_iterations {max_iterations}). NDCG Score.",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )
        return result

    def evaluate_row(self, row: pd.Series) -> int:
        """
        Evaluate a single row and return the rank index of the ground truth SKU.
        Returns None if not found or on error.
        """
        created_at = row.get('created_at', '')
        query = row.get('query', '')
        ground_truth_sku = row.get('ground_truth_sku', '')
        provider = self.miner_artifact.provider
        winning_response = row.get('winning_response', '')
        context = row.get('context', '')

        query = str(query)
        ground_truth_sku = str(ground_truth_sku)
        provider = str(provider)        

        # Decode double-encoded JSON string
        products = self.decode_context(context)
        
        # Verify if Ground Truth is in the context
        ground_truth_lower = ground_truth_sku.lower()
        if not any(p.sku.lower() == ground_truth_lower for p in products):
            logger.warning(f"\033[31mIMPOSSIBLE TASK: Ground Truth SKU {ground_truth_sku} is NOT in the provided context/catalog of {len(products)} items.\033[0m")
        else:
            logger.info(f"\033[32mGround Truth SKU {ground_truth_sku} IS present in the context.\033[0m")

        num_recs = 10 # Force 10 recommendations for NDCG@10
        
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
        duration = et - st

        ground_truth_lower = ground_truth_sku.lower()
        
        # Find index
        rank_index = None
        for i, sku in enumerate(recommended_skus):
            if sku.lower() == ground_truth_lower:
                rank_index = i
                break
        
        if rank_index is not None:
            logger.info(f"\033[32mGround truth SKU {ground_truth_sku} found at rank {rank_index}\033[0m")
        else:
            logger.info(f"\033[31mGround truth SKU {ground_truth_sku} NOT found in recommendations.\033[0m")

        self.log_miner_response(
            run_id=self.run_id,
            query=query,
            num_recs=num_recs,
            recommended_skus=recommended_skus,
            duration=duration
        )

        return rank_index

if __name__ == "__main__":
    # Mock Artifact for testing
    from models.miner_artifact import Artifact, SamplingParams
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")

    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    mock_artifact = Artifact(
        name="test_artifact",
        miner_hotkey="test_hotkey",
        miner_uid=1,
        provider="open_router", # Valid enum value
        model="moonshotai/kimi-k2.5",
        system_prompt_template="You are a recommender. Context: {{ product_catalog }}",
        user_prompt_template="Recommend {{ num_recs }} products for query: {{ sku }}. Return ONLY a JSON array of strings, e.g. ['B00...', 'B01...'].",
        sampling_params=SamplingParams(temperature=0.7)
    )
    
    eval_instance = AmazonAllBeautyNDCG1000(run_id="test_run", miner_artifact=mock_artifact)
    result = eval_instance.run(max_iterations=20)
    print(result)