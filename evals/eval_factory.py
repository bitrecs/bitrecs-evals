import logging
from typing import List, Dict
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from common import constants as CONST
from datetime import datetime, timezone
from evals.bitrecs_catalog_eval import BitrecsCatalogEval
from evals.bitrecs_prompt_eval import BitrecsPromptEval
from evals.bitrecs_reason_eval import BitrecsReasonEval



logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EvalFactory:
    """
    Factory for creating and running evals.
    Supports dynamic registration of new evals.
    """
    
    _registry: Dict[BitrecsEvaluationType, type] = {
        BitrecsEvaluationType.PROMPT: BitrecsPromptEval,
        BitrecsEvaluationType.REASON: BitrecsReasonEval,
        BitrecsEvaluationType.CATALOG: BitrecsCatalogEval
    }
    
    @classmethod
    def register_eval(cls, name: BitrecsEvaluationType, eval_class: type):
        """Register a new eval class."""
        cls._registry[name] = eval_class
    
    @classmethod
    def run_eval(cls, eval_type: BitrecsEvaluationType, miner_artifact: Artifact, run_id: str, max_iterations: int = 10) -> EvalResult:
        """Create and run a specific eval."""
        if eval_type not in cls._registry:
            raise ValueError(f"Unknown eval type: {eval_type}")
        
        eval_instance = cls._registry[eval_type](run_id, miner_artifact)
        return eval_instance.run(max_iterations)
    
    @classmethod
    def run_all_evals(cls, run_id: str, miner_artifact: Artifact, eval_types: List[BitrecsEvaluationType] = None, max_iterations: int = 10) -> List[EvalResult]:
        """Run multiple evals and return aggregated results."""
        if eval_types is None:
            raise ValueError("eval_types must be provided")
            #eval_types = list(cls._registry.keys())
        
        results = []        
        for eval_type in eval_types:
            try:
                logger.debug(f"\033[34mRunning eval type: {eval_type}\033[0m")
                result = cls.run_eval(eval_type, miner_artifact, run_id, max_iterations)
                results.append(result)
            except Exception as e:
                # Log error and continue (don't fail all evals)
                logger.error(f"Failed to run {eval_type} eval: {e}")
                results.append(EvalResult(
                    eval_name=f"{eval_type} Eval",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    hot_key=miner_artifact.miner_hotkey,
                    score=0.0,
                    passed=False,
                    rows_evaluated=0,
                    details=f"FAIL - Error: {e}",
                    duration_seconds=0.0,
                    run_id=run_id                    
                ))
        return results