import time
import logging
import secrets
import pandas as pd
from datetime import datetime, timezone
from db.models.eval import db
from evals.base_eval import BaseEval 
from evals.eval_result import EvalResult
from evals.scoring.sku_scorer import SKURelevanceScorer
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact

logger = logging.getLogger(__name__)


"""
Evaluate how relevant each SKU recommendation is to the task
check: if the recommended SKUs are relevant to the query SKU.
data: loads fresh miner responses for the given miner artifact.

"""

class BitrecsSkuEval(BaseEval):
  
    min_sample_size = 2

    pass_threshold = 0.3
    
    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

        db.connect()
        db.close()     
        db_path = db.database  

        self.judge_llm_provider = "OPEN_ROUTER"
        self.judge_llm_model = "mistralai/mistral-small-3.2-24b-instruct"
        
        self.sku_scorer = SKURelevanceScorer(source_db=db_path, 
                                             judge_provider=self.judge_llm_provider, 
                                             judge_model=self.judge_llm_model, 
                                             is_debug=False,
                                             run_id=run_id)
        self.debug_prompts = False

        df = self.load_recent_answers()
        self.holdout_df = df

        if len(self.holdout_df) == 0:
            if 1==1:
                self.init_baseline_reasons()                
                self.holdout_df = self.load_recent_answers()
                if len(self.holdout_df) == 0:
                    logger.error(f"No data for hotkey {self.miner_artifact.miner_hotkey}")
                    raise ValueError(f"No recent miner responses found for {self.miner_artifact.miner_hotkey}")
        
        if len(self.holdout_df) < self.min_sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.min_sample_size}")
        
        self.holdout_df = self.holdout_df.head(50)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_SKU_DAILY
    
    def load_recent_answers(self) -> pd.DataFrame:
        try:
            db.connect()
            sql = "Select * From miner_responses where hotkey = ? Order By created_at Desc Limit 100"
            df = pd.read_sql_query(sql, db.connection(), params=(self.miner_artifact.miner_hotkey,))
            return df
        finally:
            db.close()

    def init_baseline_reasons(self):
        rows = 5
        for idx in range(rows):
            reason = f"This is a baseline iteration number {idx+1}."
            logger.info(f"Reason Baseline {idx+1}: {reason}")
            
            num_recs = 5
            #random_product = random.choice(self.sku_scorer.product_catalog)
            random_product = secrets.choice(self.sku_scorer.product_catalog)            
            query = random_product.sku
            
            prompt_factory = PromptFactory(
                miner_artifact=self.miner_artifact,
                sku=query,
                products=self.sku_scorer.product_catalog,
                num_recs=num_recs,
                debug=self.debug_prompts
            )
            system_prompt, user_prompt = prompt_factory.generate_prompt()
            tokens = PromptFactory.get_token_count(system_prompt + user_prompt)
            logger.info(f"Prompt Tokens: {tokens}")

            temperature = self.miner_artifact.sampling_params.temperature
            model = self.miner_artifact.model
            provider = self.miner_artifact.provider

            st = time.monotonic()
            system_prompt, user_prompt = prompt_factory.generate_prompt()
            server = LLM.try_parse(provider)
            llm_output = LLMFactory.query_llm(server=server,
                                                model=model,
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                temp=temperature)
            recommended_skus = PromptFactory.tryparse_llm(llm_output)
            #logger.info(f"LLM Output: {llm_output}")
            logger.info(f"Query : {query}")
            logger.info(f"Recommended SKUs: {recommended_skus}")
            et = time.monotonic()
            durtion = et - st

            self.log_miner_response(
                run_id=self.run_id,
                query=query,
                num_recs=num_recs,
                recommended_skus=recommended_skus,
                duration=durtion
            )

    
    def run(self, max_iterations = 10) -> EvalResult:
        """
        Run the Bitrecs reason evaluation.
        """
        
        count = 0
        success_count = 0
        exception_count = 0
        total_duration = 0.0
        start_time = time.monotonic()
        eval_score = 0.0        
        hotkey = self.miner_artifact.miner_hotkey        
        try:
            top = 3
            miner_score = self.sku_scorer.score_miner(hot_key=hotkey, top=top)            
            eval_score = miner_score
            count = top
        except Exception as e:
            logger.error(f"Exception during evaluation: {e}")
            exception_count += 1     
        end_time = time.monotonic()
        total_duration = end_time - start_time        
        eval_success = eval_score >= self.pass_threshold        

        result = EvalResult(
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=eval_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {len(self.holdout_df)} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature            
        )
        return result    
  
   