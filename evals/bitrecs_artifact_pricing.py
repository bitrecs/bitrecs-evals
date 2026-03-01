import time
import traceback
import logging
from datetime import datetime, timezone
from evals.eval_result import EvalResult
from llm.inference_coster import InferenceCoster
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from evals.bitrecs_basic_eval import BitrecsBasicEval

logger = logging.getLogger(__name__)


class BitrecsArtifactPricing(BaseEval):

    def __init__(self, run_id: str, miner_artifact: Artifact):
        super().__init__(run_id, miner_artifact)

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_ARTIFACT_PRICING

    @property
    def cost_threshold(self) -> float:
        """
        Define the maximum allowed cost per million tokens for the prompt.
        """
        return 1.10  # $1.1 per million tokens as the threshold        

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
            coster = InferenceCoster(provider, model_name)
            pricing = coster.fetch_cost()
            if pricing:

                prompt_cost_per_token = pricing.input
                completion_cost_per_token = pricing.output
                
                # Compute token counts from the prompt templates
                system_tokens = BitrecsBasicEval.get_token_count(self.miner_artifact.system_prompt_template) if getattr(self.miner_artifact, 'system_prompt_template', None) else 0
                user_tokens = BitrecsBasicEval.get_token_count(self.miner_artifact.user_prompt_template) if getattr(self.miner_artifact, 'user_prompt_template', None) else 0
                prompt_tokens = system_tokens + user_tokens
                
                # Default completion token estimation if not provided by the artifact's max_tokens
                completion_tokens = 1000
                if getattr(self.miner_artifact, 'sampling_params', None) and getattr(self.miner_artifact.sampling_params, 'max_tokens', None):
                    completion_tokens = self.miner_artifact.sampling_params.max_tokens

                total_cost = coster.calculate_cost(prompt_tokens, completion_tokens)
                prompt_cost_per_million = pricing.input
                
                if prompt_cost_per_million >= self.cost_threshold:
                    final_score = 0.0
                    result = False
                    reason = f"FAIL: Input cost ${prompt_cost_per_million:.2f}/1M tokens exceeds maximum allowed ${self.cost_threshold:.2f} threshold. (Provider: {provider}, Model: {model_name})"
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
