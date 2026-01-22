import secrets
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from db.models.eval import db
from evals.base_eval import BaseEval 
from evals.eval_result import EvalResult
from evals.scoring.rules_scorer import RulesScorer
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact


logger = logging.getLogger(__name__)


"""
Evaluate individual reason statements for recommendations.

check: validity of reason statements for recommended SKUs against rules.
data: loads fresh miner responses for the given miner artifact.

"""

class BitrecsReasonEval(BaseEval):  
   
    
    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

        db.connect()
        db.close()     
        db_path = db.database  
     
        self.rules_scorer = RulesScorer(db_full_path=db_path, max_workers=4, debug=True, run_id=run_id)        
        self.debug_prompts = False

        #print(df.head())
        if 1==1:
            self.init_baseline_reasons()

        df = self.load_recent_answers()
        self.holdout_df = df

        if len(self.holdout_df) == 0:
            logger.error(f"No data for hotkey {self.miner_artifact.miner_hotkey}")
            raise ValueError(f"No recent miner responses found for {self.miner_artifact.miner_hotkey}")
        
        if len(self.holdout_df) < self.sample_size:
            raise ValueError(f"Holdout set size {len(self.holdout_df)} is less than minimum required {self.sample_size}")

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_REASON_DAILY
    
    def load_recent_answers(self) -> pd.DataFrame:
        try:
            db.connect()
            sql = "Select * From miner_responses where hotkey = ? Order By created_at Desc Limit 100"
            df = pd.read_sql_query(sql, db.connection(), params=(self.miner_artifact.miner_hotkey,))
            return df
        finally:
            db.close()

    def init_baseline_reasons(self):
        max_examples = 5
        for idx in range(max_examples):
            reason = f"This is a baseline iteration number {idx+1}."
            logger.info(f"Reason Baseline {idx+1}: {reason}")
            
            random_product = secrets.choice(self.rules_scorer.product_catalog)
            num_recs = 5
            query = random_product.sku
            
            prompt_factory = PromptFactory(
                miner_artifact=self.miner_artifact,
                sku=query,
                products=self.rules_scorer.product_catalog,
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
            duration = et - st

            self.log_miner_response(
                run_id=self.run_id,
                query=query,
                num_recs=num_recs,
                recommended_skus=recommended_skus,
                duration=duration
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
            report = self.rules_scorer.score_miner(miner_hotkey=hotkey, days_ago=7, min_success=1)
            logger.info(f"Notes for miner {hotkey}:")
            for note in report.evaluator_notes:
                logger.info(f"  - {note}")
            logger.info(f"R Score: {report.r_score}")
            logger.info(f"S Score: {report.s_score}")        
            logger.info(f"F Score: {report.f_score}")
            eval_score = report.r_score
            count = report.num_requests_evaluated
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
  
   