import json
import os
import logging
from typing import List
import pandas as pd
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from db.models.eval import db, Miner, MinerResponse
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from common import constants as CONST
from models.product import Product

logger = logging.getLogger(__name__)

class BaseEval(ABC):
    """
    Abstract base class for all evaluation suites.
    Ensures standardized invocation and result reporting.
    """
    @property
    def sample_size(self) -> int:        
        return 3
    
    @property
    def pass_threshold(self) -> float:
        return 0.3
    
    def __init__(self, run_id: str, miner_artifact: Artifact):      
        if not run_id:
            raise ValueError("Run ID is required")
        self.run_id = run_id
        if not miner_artifact:
            raise ValueError("Miner artifact is required")
        self.miner_artifact = miner_artifact
        logger.info(f"Initialized {self.miner_artifact.model} eval with run ID: {self.run_id}")
        logger.info(f"Artifact {self.miner_artifact.provider} ")
        logger.info(f"Artifact {self.miner_artifact.created_at} ")
        logger.info(f"Artifact {self.miner_artifact.miner_hotkey} ")
    
    @abstractmethod
    def run(self, max_iterations: int = 10) -> EvalResult:
        """
        Run the evaluation and return a standardized EvalResult.
        Subclasses must implement this.
        """
        pass
    
    @abstractmethod
    def eval_type(self) -> BitrecsEvaluationType:
        """Return the type of evaluation being performed."""
        pass
    
    def get_eval_name(self) -> str:        
        this_type = self.eval_type()
        name = str(this_type)
        return name
    
    def get_latest_holdout(self, specific_file: str = None) -> pd.DataFrame:
        """
        Get latest holdout set from data/holdout directory by parsing date/time from filenames.
        
        Filenames are expected in format: prompt_holdout_YYYYMMDD_HHMMSS.csv
        Sorts by the embedded datetime to find the most recent.
        """
        #specific_file = "fashion_data_holdout_20260102.csv"
        #specific_file = "amazon_ranking_100_strict_sample_20.csv"
        holdout_dir = os.path.join(CONST.ROOT_DIR, "data", "holdout")
        holdout_files = [f for f in os.listdir(holdout_dir) if f.endswith('.csv')]
        if not holdout_files:
            raise FileNotFoundError("No holdout files found.")
        if specific_file:
            specific_path = os.path.join(holdout_dir, specific_file)
            if not os.path.exists(specific_path):
                raise FileNotFoundError(f"Specified holdout file {specific_file} not found.")
            logger.info(f"Using specified holdout file: {specific_path}")
            return pd.read_csv(specific_path)
        
        # Parse datetime from filename and sort
        file_datetime_pairs = []
        for f in holdout_files:
            try:
                # Extract datetime part: e.g., '20251213_150551' from 'prompt_holdout_20251213_150551.csv'
                datetime_str = f.split('_')[-2] + '_' + f.split('_')[-1].replace('.csv', '')
                dt = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
                file_datetime_pairs.append((f, dt))
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse datetime from filename {f}: {e}")
                continue
        
        if not file_datetime_pairs:
            raise FileNotFoundError("No valid holdout files with parseable dates found.")
        
        # Sort by datetime descending (most recent first)
        file_datetime_pairs.sort(key=lambda x: x[1], reverse=True)
        latest_file = file_datetime_pairs[0][0]
        latest_path = os.path.join(holdout_dir, latest_file)
        logger.info(f"Using latest holdout file: {latest_path}")       
        return pd.read_csv(latest_path)
    
    def decode_context(self, context: str) -> List[str]:
        """Context is double encoded from holdout sets """
        try:
            # First decode: from escaped JSON string to JSON string
            context_decoded = json.loads(context) if isinstance(context, str) and context.startswith('"') else context
            # Now context_decoded should be a proper JSON string like '[{...}]'            
            products = Product.try_parse_context_strict(context_decoded)            
            if len(products) == 0:
                logger.warning(f"No products parsed from context.")
                return []
            return products
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse context for row: {e}")
            products = []
    

    def log_miner_response(self, run_id: str, query: str, num_recs: int, recommended_skus: list, duration: float):
        """
        Log the miner response to the database.
        """
        try:
            db.connect()
            db.create_tables([Miner, MinerResponse], safe=True)            
            miner, created = Miner.get_or_create(hotkey=self.miner_artifact.miner_hotkey)            
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


    def null_eval(self) -> EvalResult:
        """
        Return a null EvalResult indicating the evaluation could not be performed.
        """
        result = EvalResult(           
            eval_name=self.get_eval_name(),  # Use base method
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=0.0,
            passed=False,
            rows_evaluated=0,
            details="FAILED: Evaluation could not be performed due to data initialization failure.",
            duration_seconds=0.0,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )        
        return result
    