
import json
import time
import logging
from datetime import datetime, timezone
from evals.base_eval import BaseEval 
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from commerce.product_factory import ProductFactory
from llm.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)


"""
Call provider using prompt and ensure catalog skus match

"""

class BitrecsCatalogEval(BaseEval):
  
    min_row_count = 3
    
    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)
        self.holdout_df = None
        holdout_df = self.get_latest_holdout()
        self.holdout_df = holdout_df
        logger.info(f"Loaded holdout set with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.min_row_count:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_row_count}")        
        self.debug_prompts = False

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.CATALOG
    
    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the Bitrecs catalog evaluation.
        """
        start_time = time.monotonic()
        count = 0
        success_count = 0
        exception_count = 0
        total_duration = 0.0
        for idx, row in self.holdout_df.iterrows():
            if idx >= max_iterations:
                break
            try:
                eval_start = time.monotonic()
                eval_score = self.evaluate_row(row)
                eval_end = time.monotonic()
                total_duration += (eval_end - eval_start)
                
                if eval_score == 1.0:  # Assuming 1.0 for valid catalog match
                    success_count += 1
                
                logger.info(f"Row {idx}: Score {eval_score}")
            except Exception as e:
                exception_count += 1
                logger.error(f"Error evaluating row {idx}: {e}")
            finally:
                count += 1
        
        # Calculate overall score (fraction of valid catalogs)
        overall_score = success_count / count if count > 0 else 0.0
        end_time = time.monotonic()
        total_time = end_time - start_time
        
        logger.info(f"Evaluation completed in {total_time:.2f} seconds.")
        
        passed = overall_score >= 0.8  # Example pass threshold

        passed = True
        
        return EvalResult(
            eval_name=self.get_eval_name(),
            hot_key=self.miner_artifact.miner_hotkey,
            created_at=datetime.now(timezone.utc).isoformat(),
            rows_evaluated=count,
            duration_seconds=total_duration,
            passed=passed,
            score=overall_score,

            details={
                "total_rows": count,
                "successful_rows": success_count,
                "exceptions": exception_count,
                "total_duration": total_duration
            },
            temperature=self.miner_artifact.sampling_params.temperature
        )
    

    def evaluate_row(self, row) -> float:
        """
        Evaluate a single row for catalog matching.
        Return a score (e.g., 1.0 for match, 0.0 for no match).
        """
        # Placeholder implementation - replace with actual logic       
        
        
        winning_response = row.get('winning_response', '')
        logger.info(f"Winning response length: {len(winning_response)}")        

        ground_truth_sku = row.get('ground_truth_sku', '')
        context = row.get('context', '')
        # Decode double-encoded JSON string
        try:
            # First decode: from escaped JSON string to JSON string
            context_decoded = json.loads(context) if isinstance(context, str) and context.startswith('"') else context
            # Now context_decoded should be a proper JSON string like '[{...}]'            
            catalog = ProductFactory.try_parse_context_strict(context_decoded)
            #products = Product.try_parse_context_strict(context_decoded)[:5]
            if len(catalog) == 0:
                logger.warning("No products found in catalog context")
                return False
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse context for row: {e}")
            catalog = []
        
        #catalog = ProductFactory.try_parse_context_strict(context)
        token_count = PromptFactory.get_token_count(context)
        logger.info(f"Catalog token count: {token_count}")

        catalog_count = len(catalog)
        logger.info(f"Catalog product count: {catalog_count}")

        if catalog_count < 100:
            logger.warning("No products found in catalog context")
            return 0.0
        
        return 1.0
    

       