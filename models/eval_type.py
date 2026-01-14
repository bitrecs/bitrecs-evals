
from enum import Enum


class BitrecsEvaluationType(str, Enum):
    PROMPT = "prompt"
    REASON = "reason"
    CATALOG = "catalog"
    RECALL = "recall"
    RANKING = "ranking"