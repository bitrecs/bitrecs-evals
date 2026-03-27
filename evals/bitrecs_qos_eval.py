import logging
import secrets
import time
from datetime import datetime, timezone
from commerce.product_factory import CatalogProvider, ProductFactory
from common import constants as CONST
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from models.product import Product


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Evaluates quality of service (QoS) for product recommendations.

check: ensure Model / Provider meets QoS standards.
data: local product catalog

"""

class BitrecsQoSEval(BaseEval):

    @property
    def accuracy_threshold(self) -> float:
        return 0.6  #Minimum accuracy score to pass
    
    @property
    def duration_threshold(self) -> float:
        return 0.15  #Lower = slower models allowed
    
    @property
    def tolerance_seconds_per_query(self) -> float:
        return 15.0  #max expected duration per rec

    @property
    def num_recs(self) -> int:
        return 5

    def __init__(self, run_id: str, miner_artifact: Artifact = None):
        super().__init__(run_id, miner_artifact)

        woo_products = ProductFactory.load_default_catalog(CatalogProvider.WOOCOMMERCE)
        self.product_catalog = [Product(sku=p['sku'], name=p['name'], price=str(p['price'])) for p in woo_products]
        if len(self.product_catalog) == 0:
            raise ValueError("Product catalog is empty")
       

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_QOS_DAILY

    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the QoS evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0
        durations = []
        inference_data = []
        for idx in range(self.sample_size):
            reason = f"This is a QoS evaluation iteration number {idx+1}."
            logger.info(f"QoS Eval {idx+1}: {reason}")
            
            random_product = secrets.choice(self.product_catalog)
            num_recs = self.num_recs
            query = random_product.sku
            
            try:
                prompt_factory = PromptFactory(
                    miner_artifact=self.miner_artifact,
                    sku=query,
                    products=self.product_catalog,
                    num_recs=num_recs,
                    debug=False
                )
                system_prompt, user_prompt = prompt_factory.generate_prompt()
                tokens = PromptFactory.get_token_count(system_prompt + user_prompt)
                logger.info(f"Prompt Tokens: {tokens}")

                temperature = self.miner_artifact.sampling_params.temperature
                model = self.miner_artifact.model
                provider = self.miner_artifact.provider

                st = time.monotonic()
                server = LLM.try_parse(provider)
                inference = LLMFactory.query_llm_with_usage(server=server,
                                                    model=model,
                                                    system_prompt=system_prompt,
                                                    user_prompt=user_prompt,
                                                    temp=temperature)
                llm_output = inference.response
                inference_data.append(inference.data)

                et = time.monotonic()
                duration = et - st
                durations.append(duration)
                recommended_skus = PromptFactory.tryparse_llm(llm_output)
                #logger.info(f"LLM Output: {llm_output}")
                logger.info(f"Query : {query}")
                logger.info(f"Duration : {duration:.2f} seconds")
                
                # Simple accuracy check            
                if len(recommended_skus) == num_recs:
                    success_count += 1
                    logger.info(f"QoS Eval Passed: Received {num_recs} valid recommendations.")
                else:
                    logger.warning(f"QoS Eval Failed: Expected {num_recs} valid recommendations, got {len(recommended_skus)}")
                count += 1

                self.log_miner_response(
                    run_id=self.run_id,
                    query=query,
                    num_recs=num_recs,
                    recommended_skus=recommended_skus,
                    duration=duration
                )
                self.log_inference_data(run_id=self.run_id, data=inference.data)

            except Exception as e:
                exception_count += 1
                logger.error(f"Exception in QoS Eval {idx+1}: {e}")
                count += 1

        
        end_time = time.monotonic()
        total_duration = end_time - start_time        
        accuracy_score = success_count / count if count > 0 else 0.0        
        
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        logger.info(f"Average query duration: {avg_duration:.2f} seconds")
        # Score: 1.0 if avg <= tolerance, else penalize linearly but cap at 0
        duration_score = max(0.0, 1.0 - (avg_duration / self.tolerance_seconds_per_query))
        
        final_score = accuracy_score * duration_score
        eval_success = (accuracy_score >= self.accuracy_threshold) and (duration_score >= self.duration_threshold)
        
        result = EvalResult(           
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {self.sample_size} rows with {exception_count} exceptions (max_iterations {max_iterations}). Accuracy: {accuracy_score:.2f}, Avg Duration: {avg_duration:.2f}s, Duration Score: {duration_score:.2f}.",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id,
            inference_data=inference_data
        )        

        return result



