
from enum import Enum


class BitrecsEvaluationType(str, Enum):
    BITRECS_SAFE_DAILY = "bitrecs_safe_daily"
    BITRECS_BASIC_DAILY = "bitrecs_basic_daily"
    BITRECS_PROMPT_DAILY = "bitrecs_prompt_daily"
    BITRECS_REASON_DAILY = "bitrecs_reason_daily"
    BITRECS_SKU_DAILY = "bitrecs_sku_daily"

    AMAZON_PROMPT_100 = "amazon_prompt_100"
    AMAZON_PROMPT_500 = "amazon_prompt_500"
    AMAZON_PROMPT_1000 = "amazon_prompt_1000"