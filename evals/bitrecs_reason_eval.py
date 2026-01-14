
import time
import logging
from datetime import datetime, timezone
from evals.base_eval import BaseEval 
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact

logger = logging.getLogger(__name__)


"""
Evaluate individual sku reason statements for recommendations.
Each reason statement is checked for validity using an LLM.

"""

class BitrecsReasonEval(BaseEval):
  
    min_row_count = 3
    
    def __init__(self,  run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)
        self.holdout_df = None
        holdout_df = self.get_latest_holdout()
        self.holdout_df = holdout_df
        logger.info(f"Loaded holdout set with {len(self.holdout_df)} records.")
        if len(self.holdout_df) < self.min_row_count:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_row_count}")        
        self.debug_prompts = False

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.REASON
    
    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the Bitrecs reason evaluation.
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
                
                if eval_score == 1.0:  # Assuming 1.0 for valid reason
                    success_count += 1
                
                logger.info(f"Row {idx}: Score {eval_score}")
            except Exception as e:
                exception_count += 1
                logger.error(f"Error evaluating row {idx}: {e}")
            finally:
                count += 1
        
        # Calculate overall score (fraction of valid reasons)
        score = success_count / len(self.holdout_df) if len(self.holdout_df) > 0 else 0.0
        eval_success = score >= 1.0  # Pass if all reasons are valid (adjust threshold as needed)
        
        end_time = time.monotonic()
        total_duration = end_time - start_time

        eval_success = True
        
        result = EvalResult(
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=score,
            passed=eval_success,
            rows_evaluated=len(self.holdout_df),
            details=f"Evaluated {count} of {len(self.holdout_df)} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature            
        )
        return result
    
    def evaluate_row(self, row) -> float:
        """
        Evaluate a single row: Use LLM to check if the reason statement is valid for the recommendation.
        Returns 1.0 if valid, 0.0 if invalid.
        """
        # recommendation = row.get('recommendation', '')
        # reason_statement = row.get('reason_statement', '')
        
        # if not recommendation or not reason_statement:
        #     raise ValueError("Row missing 'recommendation' or 'reason_statement'")
        
        # # Create prompt for LLM evaluation
        # system_prompt = PromptFactory.create_system_prompt(self.miner_artifact)
        # user_prompt = f"""
        # Evaluate the following reason statement for the recommendation. Is the reason logical, accurate, and well-supported? Respond with 'VALID' if yes, 'INVALID' if no, and explain briefly.

        # Recommendation: {recommendation}
        # Reason Statement: {reason_statement}
        # """
        
        # # Query LLM
        # server = self.miner_artifact.provider.lower()
        # model = self.miner_artifact.model
        # temp = self.miner_artifact.sampling_params.temperature
        
        # llm_output = LLMFactory.query_llm(server=server, model=model, system_prompt=system_prompt, user_prompt=user_prompt, temp=temp)
        
        # # Parse LLM response (simple keyword check; enhance with regex or JSON parsing if needed)
        # if "VALID" in llm_output.upper():
        #     return 1.0
        # elif "INVALID" in llm_output.upper():
        #     return 0.0
        # else:
        #     logger.warning(f"Ambiguous LLM response: {llm_output}")
        #     return 0.0  # Default to invalid on ambiguity
        print("Evaluating reason statement...")  # Placeholder
        #print(row)  # Show row being evaluated
        return 0 # Placeholder implementation; replace with actual LLM evaluation logic