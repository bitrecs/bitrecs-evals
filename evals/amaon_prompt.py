import logging
import time
import pandas as pd
from datetime import datetime, timezone
from common import constants as CONST
from db.models.eval import Miner, MinerResponse, db
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
checks for sku exist in pair of bought together items with amazon data

"""

class AmazonPromptEval(BaseEval):

    min_row_count = 3

    def __init__(self, run_id: str, miner_artifact: Artifact = None):      
        super().__init__(run_id, miner_artifact)

        # download hugginface?

        # holdout_df = self.get_latest_holdout()
        # self.holdout_df = holdout_df
        # logger.info(f"Loaded holdout set with {len(self.holdout_df)} records.")
        # if len(self.holdout_df) < self.min_row_count:
        #     raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_row_count}")        


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
    
    def evaluate_row(self, row: pd.Series) -> bool:
        """
        Evaluate a single row from the holdout set.
        """

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