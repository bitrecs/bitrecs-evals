import json
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
Evaluates needle in a haystack of sku in catalog

check: ensure Model / Provider can meet selection standards.
data: local product catalog
note: This does not test the user defined prompt but only the provider/model

"""

class BitrecsHaystackEval(BaseEval):
   
    @property
    def pass_threshold(self) -> float:
        return 0.6

    def __init__(self, run_id: str, miner_artifact: Artifact = None):
        super().__init__(run_id, miner_artifact)

        woo_products = ProductFactory.load_default_catalog(CatalogProvider.WOOCOMMERCE)
        self.product_catalog = [Product(sku=p['sku'], name=p['name'], price=str(p['price'])) for p in woo_products]
        if len(self.product_catalog) == 0:
            raise ValueError("Product catalog is empty")            
       

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_HAYSTACK_DAILY

    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the QoS evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0
        inference_data = []
        for idx in range(self.sample_size):
            reason = f"This is a Haystack evaluation iteration number {idx+1}."
            logger.info(f"Haystack Eval {idx+1}: {reason}")
            
            random_product = secrets.choice(self.product_catalog)            
            query = random_product.sku
            
            try:

                system_prompt = "You are a helpful assistant."
                products = json.dumps([{'sku': p.sku, 'name': p.name, 'price': p.price} for p in self.product_catalog])
                
                user_prompt = f"""
                # Instructions:

                Return the element with sku = {query} from the following list of products: {products}
                \n
                \n
                # Return Format:

                [{{"sku": "<sku>", "name": "<name>", "price": "<price>"}}, ...] """

                tokens = PromptFactory.get_token_count(system_prompt + user_prompt)
                logger.info(f"Prompt Tokens: {tokens}")
                
                if 1==2:
                    logger.info(f"System Prompt: {system_prompt}")
                    logger.info(f"User Prompt: {user_prompt}")

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
                matched_sku = PromptFactory.tryparse_llm(llm_output)
                logger.info(f"Query : {query}")
                logger.info(f"Matched SKU: {matched_sku}")
                logger.info(f"Duration : {et - st:.2f} seconds")
                
                if len(matched_sku) == 1 and matched_sku[0]["sku"] == query:
                    success_count += 1
                    logger.info(f"Haystack Eval Passed: Received 1 valid recommendation.")
                else:
                    logger.warning(f"Haystack Eval Failed: Expected 1 valid recommendation, got {len(matched_sku)}")
                count += 1

                self.log_miner_response(
                    run_id=self.run_id,
                    query=query,
                    num_recs=1,
                    recommended_skus=matched_sku,
                    duration=duration
                )
                self.log_inference_data(run_id=self.run_id, data=inference.data)
            except Exception as e:
                exception_count += 1
                logger.error(f"Exception in Haystack Eval {idx+1}: {e}")
                count += 1
        
        end_time = time.monotonic()
        total_duration = end_time - start_time        
        final_score = success_count / count if count > 0 else 0.0                
        eval_success = False

        if final_score >= self.pass_threshold:
            eval_success = True    
        
        result = EvalResult(           
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {self.sample_size} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id,
            inference_data=inference_data
        )        

        return result



