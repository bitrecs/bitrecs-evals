import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass  
class CostReport:  
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float

    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int, input_price_per_million_tokens: float, output_price_per_million_tokens: float) -> 'CostReport':
        total_tokens = input_tokens + output_tokens
        cost = (input_tokens / 1_000_000) * input_price_per_million_tokens + (output_tokens / 1_000_000) * output_price_per_million_tokens
        return CostReport(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost
        )
    
    @staticmethod
    def calculate_cost_from_report(report: dict, input_price_per_million_tokens: float, output_price_per_million_tokens: float) -> 'CostReport':
        if "inference_data" in report:
            total_prompt_tokens = sum(item["prompt_tokens"] for item in report["inference_data"])
            total_completion_tokens = sum(item["completion_tokens"] for item in report["inference_data"])         
            return CostReport.calculate_cost(total_prompt_tokens, total_completion_tokens, input_price_per_million_tokens, output_price_per_million_tokens)
        else:
            logger.error("No inference_data found in report.")
            return CostReport(0, 0, 0, 0.0)  # Return zero cost if no data
