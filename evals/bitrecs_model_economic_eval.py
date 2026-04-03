import os
import math
import time
import traceback
import logging
from datetime import datetime, timezone
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval

logger = logging.getLogger(__name__)


class BitrecsModelEconomicEval(BaseEval):
    """
    Scores models on cost-efficiency using an exponential decay curve.
    Cheaper models score higher. Always passes — score reflects cost penalty.

    Score = exp(-k * weighted_cost)
    where weighted_cost = (INPUT_WEIGHT * input_cost) + (OUTPUT_WEIGHT * output_cost)

    At $0.00/M tokens → score ≈ 1.0
    At $1.00/M weighted → score ≈ 0.05
    """

    INPUT_WEIGHT = 0.8
    OUTPUT_WEIGHT = 0.2
    DECAY_RATE = 3.0  # k — controls how aggressively cost is penalized

    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_MODEL_ECONOMIC_EVAL

    def _get_costs(self) -> tuple:
        MODEL_COST_INPUT = os.getenv("MODEL_COST_INPUT")
        MODEL_COST_OUTPUT = os.getenv("MODEL_COST_OUTPUT")

        if not MODEL_COST_INPUT:
            raise ValueError("MODEL_COST_INPUT environment variable must be set.")

        input_cost = float(MODEL_COST_INPUT)
        output_cost = float(MODEL_COST_OUTPUT) if MODEL_COST_OUTPUT else 0.0
        return input_cost, output_cost

    @staticmethod
    def economic_score(input_cost: float, output_cost: float,
                       input_weight: float = 0.8, output_weight: float = 0.2,
                       decay_rate: float = 3.0) -> float:
        weighted_cost = (input_weight * input_cost) + (output_weight * output_cost)
        score = math.exp(-decay_rate * weighted_cost)
        return round(max(0.0, min(1.0, score)), 6)

    def run(self, sample_size=10) -> EvalResult:
        start_time = time.monotonic()
        exception_count = 0
        final_score = 0.0
        reason = "NA"
        provider = self.miner_artifact.provider
        model_name = self.miner_artifact.model

        try:
            input_cost, output_cost = self._get_costs()
            weighted_cost = (self.INPUT_WEIGHT * input_cost) + (self.OUTPUT_WEIGHT * output_cost)
            final_score = self.economic_score(
                input_cost, output_cost,
                self.INPUT_WEIGHT, self.OUTPUT_WEIGHT, self.DECAY_RATE
            )
            logger.info(
                f"\033[36mEconomic Eval — Model: {model_name} (Provider: {provider}) "
                f"input: ${input_cost:.6f}/M, output: ${output_cost:.6f}/M, "
                f"weighted: ${weighted_cost:.6f}/M, score: {final_score:.6f}\033[0m"
            )
            reason = (
                f"Input cost: ${input_cost:.6f}/M tokens, "
                f"Output cost: ${output_cost:.6f}/M tokens, "
                f"Weighted cost: ${weighted_cost:.6f}/M tokens, "
                f"Score: {final_score:.6f} "
                f"(Provider: {provider}, Model: {model_name})"
            )
        except Exception as e:
            logger.error(f"Exception during economic evaluation: {e}")
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
            passed=True,
            rows_evaluated=1,
            details=f"Test result: True.\nScore: {final_score:.6f}.\n{reason}\nExceptions: {exception_count}\nRun ID: {self.run_id}",
            duration_seconds=total_duration,
            temperature=temperature,
            model_name=model_name,
            provider_name=provider,
            run_id=self.run_id,
            inference_data=BaseEval.load_inference_data(self.run_id)
        )
