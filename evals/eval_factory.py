import logging
from typing import List, Dict
from common import constants as CONST
from datetime import datetime, timezone

from evals.amazon_all_beauty_100 import AmazonAllBeauty100
from evals.amazon_all_beauty_1000 import AmazonAllBeauty1000
from evals.amazon_all_beauty_500 import AmazonAllBeauty500
from evals.amazon_appliances_100 import AmazonAppliances100
from evals.amazon_appliances_1000 import AmazonAppliances1000
from evals.amazon_appliances_500 import AmazonAppliances500
from evals.amazon_arts_crafts_and_sewing_100 import AmazonArtsCraftsAndSewing100
from evals.amazon_arts_crafts_and_sewing_1000 import AmazonArtsCraftsAndSewing1000
from evals.amazon_arts_crafts_and_sewing_500 import AmazonArtsCraftsAndSewing500
from evals.amazon_automotive_100 import AmazonAutomotive100
from evals.amazon_automotive_1000 import AmazonAutomotive1000
from evals.amazon_automotive_500 import AmazonAutomotive500
from evals.amazon_baby_products_100 import AmazonBabyProducts100
from evals.amazon_baby_products_1000 import AmazonBabyProducts1000
from evals.amazon_baby_products_500 import AmazonBabyProducts500
from evals.amazon_beauty_and_personal_care_100 import AmazonBeautyAndPersonalCare100
from evals.amazon_beauty_and_personal_care_1000 import AmazonBeautyAndPersonalCare1000
from evals.amazon_beauty_and_personal_care_500 import AmazonBeautyAndPersonalCare500
from evals.amazon_fashion_100 import AmazonFashion100
from evals.amazon_fashion_500 import AmazonFashion500
from evals.amazon_fashion_1000 import AmazonFashion1000

from evals.bitrecs_safe_prompt import BitrecsSafeEval
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact

from evals.eval_result import EvalResult
from evals.bitrecs_basic_eval import BitrecsBasicEval
from evals.bitrecs_sku_eval import BitrecsSkuEval

from evals.bitrecs_prompt_eval import BitrecsPromptEval
from evals.bitrecs_reason_eval import BitrecsReasonEval
from evals.amazon_fashion_100 import AmazonFashion100

from evals.amazon_video_games_100 import AmazonVideoGames100
from evals.amazon_video_games_500 import AmazonVideoGames500
from evals.amazon_video_games_1000 import AmazonVideoGames1000



logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EvalFactory:
    """
    Factory for creating and running evals.
    Supports dynamic registration of new evals.
    """
    
    _registry: Dict[BitrecsEvaluationType, type] = {
        BitrecsEvaluationType.BITRECS_BASIC_DAILY: BitrecsBasicEval,
        BitrecsEvaluationType.BITRECS_SAFE_DAILY: BitrecsSafeEval,
        BitrecsEvaluationType.BITRECS_PROMPT_DAILY: BitrecsPromptEval,
        BitrecsEvaluationType.BITRECS_REASON_DAILY: BitrecsReasonEval,
        BitrecsEvaluationType.BITRECS_SKU_DAILY: BitrecsSkuEval,        
        
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_100: AmazonAllBeauty100,
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_500: AmazonAllBeauty500,  
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_1000: AmazonAllBeauty1000,

        BitrecsEvaluationType.AMAZON_FASHION_100: AmazonFashion100,
        BitrecsEvaluationType.AMAZON_FASHION_500: AmazonFashion500,
        BitrecsEvaluationType.AMAZON_FASHION_1000: AmazonFashion1000,

        BitrecsEvaluationType.AMAZON_APPLIANCES_100: AmazonAppliances100,
        BitrecsEvaluationType.AMAZON_APPLIANCES_500: AmazonAppliances500,
        BitrecsEvaluationType.AMAZON_APPLIANCES_1000: AmazonAppliances1000,

        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_100: AmazonArtsCraftsAndSewing100,
        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_500: AmazonArtsCraftsAndSewing500,
        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_1000: AmazonArtsCraftsAndSewing1000,

        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_100: AmazonAutomotive100,
        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_500: AmazonAutomotive500,
        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_1000: AmazonAutomotive1000,

        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_100: AmazonBabyProducts100,
        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_500: AmazonBabyProducts500,
        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_1000: AmazonBabyProducts1000,

        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_100: AmazonBeautyAndPersonalCare100,
        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_500: AmazonBeautyAndPersonalCare500,
        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_1000: AmazonBeautyAndPersonalCare1000,

        
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_100: AmazonVideoGames100,        
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_500: AmazonVideoGames500,
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_1000: AmazonVideoGames1000,

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