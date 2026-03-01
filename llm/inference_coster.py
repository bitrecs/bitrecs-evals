import httpx
import logging
from typing import Optional
from dataclasses import dataclass
from llm.llm_provider import LLM

logger = logging.getLogger(__name__)


@dataclass  
class CostResult:
    """
    Represents the cost result for a given model, including input and output costs.
    Costs are expected to be in USD per million tokens.
    """
    input: float
    output: float


class InferenceCoster:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        if not LLM.is_valid(self.provider):
            raise ValueError(f"Unsupported provider: {self.provider}")

    def fetch_cost(self) -> Optional[CostResult]:
        """
        Fetch pricing for a given model based on the provider.      
        """
        if self.provider.upper() == "CHUTES":
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get("https://api.chutes.ai/chutes/")
                    response.raise_for_status()
                    data = response.json()
                    
                    target_id = self.model_name.lower()
                    target_base = target_id.split("/")[-1]
                    
                    for item in data.get("items", []):
                        item_name_lower = item.get("name", "").lower()
                        if item_name_lower == target_id or item_name_lower.endswith("/" + target_base):
                            price_info = item.get("current_estimated_price", {}).get("per_million_tokens", {})
                            input_price = price_info.get("input", {}).get("usd", 0.0)
                            output_price = price_info.get("output", {}).get("usd", 0.0)
                            return CostResult(
                                input=input_price, #chutes already factored by 1M
                                output=output_price,
                            )
            except Exception as e:
                logger.error(f"Error fetching pricing from Chutes: {e}")
                return None
            
        elif self.provider.upper() == "OPEN_ROUTER":
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get("https://openrouter.ai/api/v1/models")
                    response.raise_for_status()
                    models = response.json().get("data", [])
                    
                    target_id = self.model_name.lower()
                    target_base = target_id.split("/")[-1]
                    
                    # First pass: lowercased match
                    for model in models:
                        model_id_lower = model["id"].lower()
                        if model_id_lower == target_id or model_id_lower.endswith("/" + target_base):
                            pricing = model.get("pricing", {})
                            input = float(pricing.get("prompt", 0.0)) * 1e6 # openrouter provides per token, convert to per million tokens
                            output = float(pricing.get("completion", 0.0)) * 1e6
                            return CostResult(
                                input=input,
                                output=output
                            )
                            
                    # Second pass: remove '-instruct'
                    fallback_target = target_id.replace("-instruct", "")
                    fallback_base = fallback_target.split("/")[-1]
                    for model in models:
                        model_id_lower = model["id"].lower().replace("-instruct", "")
                        if model_id_lower == fallback_target or model_id_lower.endswith("/" + fallback_base):
                            pricing = model.get("pricing", {})
                            input = float(pricing.get("prompt", 0.0)) * 1e6
                            output = float(pricing.get("completion", 0.0)) * 1e6
                            return CostResult(
                                input=input,
                                output=output
                            )
                            
            except Exception as e:
                logger.error(f"Error fetching pricing from OpenRouter: {e}")
                return None

        else:
            logger.warning(f"Provider {self.provider} not supported for pricing fetch.")
            return None
            
        
    def calculate_cost(self, input_tokens: float, output_tokens: float) -> Optional[float]:
        pricing = self.fetch_cost()
        if pricing is None:
            logger.warning("Pricing information not available, cannot calculate cost.")
            return None
        total_cost = (input_tokens / 1e6) * pricing.input + (output_tokens / 1e6) * pricing.output
        return total_cost