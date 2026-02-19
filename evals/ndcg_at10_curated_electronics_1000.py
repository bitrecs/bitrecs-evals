import time
import math
import traceback
import pandas as pd
import logging
from datetime import datetime, timezone
from common import constants as CONST
from common.hf_utils import sample_from_url
from common.utils import rec_list_to_set
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from models.product import Product

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

class NdcgAt10CuratedElectronics1000(BaseEval):
    def __init__(self, run_id: str, miner_artifact: Artifact = None):
        super().__init__(run_id, miner_artifact)
        
        dataset_url = "https://huggingface.co/datasets/reallybigmouse4/ndgc_amazon_curated/resolve/main/Electronics/amazon_curated_electronics_universe_1000.csv"
        
        self.holdout_df = sample_from_url(dataset_url=dataset_url, sample_size=25)
        
        logger.info(f"Loaded Curated holdout set from {dataset_url} with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.sample_size}")

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.NDCG_AT10_CURATED_ELECTRONICS_1000
        
    @property
    def pass_threshold(self) -> float:
        return 0.1

    def calculate_ndcg_score(self, rank_index: int, k=10) -> float:
        """Calculates rank-aware score. index is 0-indexed position."""
        if rank_index is None or rank_index >= k:
            return 0.0
        return 1.0 / math.log2(rank_index + 2)

    def run(self, max_iterations=25) -> EvalResult:
        start_time = time.monotonic()        
        total_ndcg = 0.0
        count = 0
        success_count = 0 
        exception_count = 0
        
        for idx, row in self.holdout_df.iterrows():
            if idx >= max_iterations:
                break
            try:               
                logger.info(f"[34mATTEMPT {idx+1}/{len(self.holdout_df)}...[0m")
                
                st = time.monotonic()
                rank_index = self.evaluate_row(row)
                et = time.monotonic()
                duration = et - st
                
                ndcg_score = self.calculate_ndcg_score(rank_index)
                total_ndcg += ndcg_score
                
                logger.info(f"[32m Row {idx} evaluation took {duration:.2f}s, NDCG Score: {ndcg_score:.4f} [0m")
                
                if ndcg_score > 0:
                    success_count += 1
                    
                logger.info(f"Current Avg NDCG: {total_ndcg / (idx + 1):.4f}")
                
            except Exception as e:
                error_message = traceback.format_exc()                
                logger.error(f"Error evaluating row {idx}: {e}\n{error_message}")
                exception_count += 1
                continue
            finally:
                 count += 1

        end_time = time.monotonic()
        total_duration = end_time - start_time        
        final_score = total_ndcg / count if count > 0 else 0.0                
        
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
        logger.info(f"Context loaded with {len(products)} products.")
        
        num_recs = 10 
        
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
        logger.debug(f"LLM Output: {llm_output}")
        logger.debug(f"Recommended SKUs: {recommended_skus}")
        
        # Normalize recommended_skus to list of strings
        normalized_skus = []
        for item in recommended_skus:
            if isinstance(item, dict):
                s = item.get("sku")
                if s:
                    normalized_skus.append(str(s))
            elif isinstance(item, str):
                normalized_skus.append(item)
        recommended_skus = normalized_skus
        
        # Helper map for SKU -> Product Name
        sku_to_name = {p.sku.lower(): p.name for p in products}

        # Log Query Name
        query_name = sku_to_name.get(query.lower(), "Unknown SKU")
        logger.info(f"Query Product: {query} - {query_name}")

        # Log Ground Truth Name
        gt_name = sku_to_name.get(ground_truth_sku.lower(), "Unknown SKU")
        logger.info(f"Ground Truth Product: {ground_truth_sku} - {gt_name}")

        # Log Top 3 Recommended Names
        top_3 = recommended_skus[:3]
        for i, sku in enumerate(top_3):
            name = sku_to_name.get(sku.lower(), "Unknown SKU")
            logger.info(f"Rec #{i+1}: {sku} - {name}")

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
            logger.info(f"[32mGround truth SKU {ground_truth_sku} found at rank {rank_index}[0m")
        else:
            logger.info(f"[31mGround truth SKU {ground_truth_sku} NOT found in recommendations.[0m")

        self.log_miner_response(
            run_id=self.run_id,
            query=query,
            num_recs=num_recs,
            recommended_skus=recommended_skus,
            duration=duration
        )

        return rank_index
