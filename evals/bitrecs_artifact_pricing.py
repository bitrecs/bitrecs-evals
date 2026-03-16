import os
import time
import traceback
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval

logger = logging.getLogger(__name__)

@dataclass
class CostResult:
    input: float
    output: float

class BitrecsArtifactPricing(BaseEval):
    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_ARTIFACT_PRICING

    @property
    def max_input_cost_per_million(self) -> float:
        """
        Maximum allowed cost per million input tokens in dollars.
        Models costing > $1.00 per million input tokens will fail.
        """
        return 1.00

    def get_cost_result(self) -> CostResult:
        MODEL_COST_INPUT = os.getenv("MODEL_COST_INPUT")
        MODEL_COST_OUTPUT = os.getenv("MODEL_COST_OUTPUT")

        # Temporarily hardcoded for testing; remove in production.
        #MODEL_COST_INPUT = 2.02  # Cost per million input tokens
        #MODEL_COST_OUTPUT = 0.04  # Not used, but kept for compatibility

        if not MODEL_COST_INPUT:
            raise ValueError("MODEL_COST_INPUT environment variable must be set.")
        if not MODEL_COST_OUTPUT:
            logger.warning("MODEL_COST_OUTPUT environment variable is not set.")            
        
        input_cost_per_million = float(MODEL_COST_INPUT)
        output_cost_per_million = float(MODEL_COST_OUTPUT)
        return CostResult(input=input_cost_per_million, output=output_cost_per_million)

    def run(self, sample_size=10) -> EvalResult:
        """
        Check if the model's input cost per million tokens exceeds the threshold.
        """
        start_time = time.monotonic()
        exception_count = 0
        result = False
        final_score = 0.0
        reason = "NA"
        provider = self.miner_artifact.provider
        model_name = self.miner_artifact.model        
        try:
            cost_result = self.get_cost_result()
            input_cost_per_million = cost_result.input
            logger.info(f"Model '{model_name}' (Provider: {provider}) input cost: ${input_cost_per_million:.2f} per million tokens")            
            if input_cost_per_million > self.max_input_cost_per_million:
                final_score = 0.0
                result = False
                reason = f"FAIL: Input cost ${input_cost_per_million:.2f}/1M tokens exceeds maximum allowed ${self.max_input_cost_per_million:.2f}. (Provider: {provider}, Model: {model_name})"
            else:
                final_score = 1.0
                result = True
                reason = f"PASS: Input cost ${input_cost_per_million:.2f}/1M tokens is within the allowed threshold of ${self.max_input_cost_per_million:.2f}. (Provider: {provider}, Model: {model_name})"
        
        except Exception as e:
            logger.error(f"Exception during pricing evaluation: {e}")
            traceback.print_exc()
            exception_count += 1
            reason = str(e)
        
        end_time = time.monotonic()
        total_duration = end_time - start_time
        
        temperature = 0.0
        if getattr(self.miner_artifact, 'sampling_params', None) and getattr(self.miner_artifact.sampling_params, 'temperature', None):
            temperature = self.miner_artifact.sampling_params.temperature
        
        return EvalResult(
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=result,
            rows_evaluated=1,
            details=f"Test result: {result}.\\nScore: {final_score:.2f}.\\n{reason}\\nExceptions: {exception_count}\\nRun ID: {self.run_id}",
            duration_seconds=total_duration,
            temperature=temperature,
            model_name=model_name,
            provider_name=provider,
            run_id=self.run_id
        )
