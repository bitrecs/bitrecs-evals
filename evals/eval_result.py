
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalResult:    
    eval_name: str
    created_at: str = ""
    hot_key: str = ""    
    passed: bool = False
    score: float = 0.0
    duration_seconds: float = 0.0
    details: str = ""
    rows_evaluated: int = 0
    model_name: str = ""
    provider_name: str = ""
    temperature: float = 0.0
    run_id: str = ""
    inference_data: List[Dict[str, Any]] = None

    @staticmethod
    def calculate_overall_score(results: List["EvalResult"]) -> float:      
        if not results:
            return 0.0
        total_score = sum(r.score for r in results)
        return total_score / len(results)
    