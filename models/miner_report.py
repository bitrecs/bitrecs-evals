from typing import List


class MinerReport:
    run_id: str
    created_at: str
    scored_at: str
    miner_hotkey: str
    miner_uid: str
    miner_ip: str
    num_success: int
    num_failures: int
    
    num_requests_evaluated: int
    total_unique_products: int
    avg_response_time: float

    r_score: float # Rules Score (Reasoning Quality)
    s_score: float # SKU Relevance Score (Reasoning Relevance)
    f_score: float # Final Score

    report_card: str
    models_used: List[str] = []    
    evaluator_notes: List[str] = []
    rank: int = -1    

    def to_dict(self):
        return {
            'created_at': self.created_at,
            'scored_at': self.scored_at,
            'miner_hotkey': self.miner_hotkey,
            'miner_uid': self.miner_uid,
            'miner_ip': self.miner_ip,
            'num_success': self.num_success,
            'num_failures': self.num_failures,
            'num_requests_evaluated': self.num_requests_evaluated,
            'total_unique_products': self.total_unique_products,
            'avg_response_time': self.avg_response_time,
            'r_score': self.r_score,
            's_score': self.s_score,
            'f_score': self.f_score,
            'report_card': self.report_card,
            'models_used': self.models_used,
            'evalator_notes': self.evaluator_notes,
            'rank': self.rank
        }
