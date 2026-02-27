import logging
import time
import traceback
import httpx
from datetime import datetime, timezone

from common import constants as CONST
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from evals.bitrecs_basic_eval import BitrecsBasicEval

logger = logging.getLogger(__name__)

class BitrecsGetArtifactPricing(BaseEval):
    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_GET_ARTIFACT_PRICING

    def get_pricing(self, provider: str, model_name: str) -> dict:
        """
        Fetch pricing for a given model based on the provider.
        """
        if provider.upper() == "CHUTES":
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get("https://api.chutes.ai/chutes/")
                    response.raise_for_status()
                    data = response.json()
                    
                    target_id = model_name.lower()
                    target_base = target_id.split("/")[-1]
                    
                    for item in data.get("items", []):
                        item_name_lower = item.get("name", "").lower()
                        if item_name_lower == target_id or item_name_lower.endswith("/" + target_base):
                            price_info = item.get("current_estimated_price", {}).get("per_million_tokens", {})
                            input_price = price_info.get("input", {}).get("usd", 0.0)
                            output_price = price_info.get("output", {}).get("usd", 0.0)
                            return {
                                "prompt": input_price / 1_000_000,
                                "completion": output_price / 1_000_000
                            }
            except Exception as e:
                logger.error(f"Error fetching pricing from Chutes: {e}")
            return {}
            
        # Default to OpenRouter
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get("https://openrouter.ai/api/v1/models")
                response.raise_for_status()
                models = response.json().get("data", [])
                
                target_id = model_name.lower()
                target_base = target_id.split("/")[-1]
                
                # First pass: lowercased match
                for model in models:
                    model_id_lower = model["id"].lower()
                    if model_id_lower == target_id or model_id_lower.endswith("/" + target_base):
                        return model.get("pricing", {})
                        
                # Second pass: remove '-instruct'
                fallback_target = target_id.replace("-instruct", "")
                fallback_base = fallback_target.split("/")[-1]
                for model in models:
                    model_id_lower = model["id"].lower().replace("-instruct", "")
                    if model_id_lower == fallback_target or model_id_lower.endswith("/" + fallback_base):
                        return model.get("pricing", {})
                        
        except Exception as e:
            logger.error(f"Error fetching pricing from OpenRouter: {e}")
            
        return {}

    def run(self, sample_size = 10) -> EvalResult:
        """
        Compute the token cost for the given artifact based on OpenRouter's pricing.
        """
        start_time = time.monotonic()
        exception_count = 0
        result = False
        final_score = 0.0
        reason = "NA"
        
        provider = self.miner_artifact.provider
        model_name = self.miner_artifact.model
        
        try:
            pricing = self.get_pricing(provider, model_name)
            if pricing:
                prompt_cost_per_token = float(pricing.get("prompt", 0.0))
                completion_cost_per_token = float(pricing.get("completion", 0.0))
                
                # Compute token counts from the prompt templates
                system_tokens = BitrecsBasicEval.get_token_count(self.miner_artifact.system_prompt_template) if getattr(self.miner_artifact, 'system_prompt_template', None) else 0
                user_tokens = BitrecsBasicEval.get_token_count(self.miner_artifact.user_prompt_template) if getattr(self.miner_artifact, 'user_prompt_template', None) else 0
                prompt_tokens = system_tokens + user_tokens
                
                # Default completion token estimation if not provided by the artifact's max_tokens
                completion_tokens = 1000
                if getattr(self.miner_artifact, 'sampling_params', None) and getattr(self.miner_artifact.sampling_params, 'max_tokens', None):
                    completion_tokens = self.miner_artifact.sampling_params.max_tokens
                
                total_cost = (prompt_tokens * prompt_cost_per_token) + (completion_tokens * completion_cost_per_token)
                
                prompt_cost_per_million = prompt_cost_per_token * 1_000_000
                
                if prompt_cost_per_million > 2.0:
                    final_score = 0.0
                    result = False
                    reason = f"FAIL: Input cost ${prompt_cost_per_million:.2f}/1M tokens exceeds maximum allowed $2.00 threshold. (Provider: {provider}, Model: {model_name})"
                else:
                    final_score = 1.0
                    result = True
                    reason = f"PASS: Provider: {provider}, Model: {model_name}. Tokens (Prompt: {prompt_tokens}, Completion: {completion_tokens}). Cost: {total_cost:.6f} (Prompt: {prompt_cost_per_token}/token, Comp: {completion_cost_per_token}/token)"
            else:
                reason = f"Could not find pricing proxy for model '{model_name}' (Provider: {provider})"
                result = False
                final_score = 0.0
                
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
