
from dataclasses import dataclass


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
    