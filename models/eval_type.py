from enum import Enum

class BitrecsEvaluationType(str, Enum):
    BITRECS_SAFE_DAILY = "bitrecs_safe_daily"
    BITRECS_BASIC_DAILY = "bitrecs_basic_daily"
    BITRECS_PROMPT_DAILY = "bitrecs_prompt_daily"
    BITRECS_REASON_DAILY = "bitrecs_reason_daily"
    BITRECS_SKU_DAILY = "bitrecs_sku_daily"    

    AMAZON_ALL_BEAUTY_100 = "amazon_all_beauty_100"    
    AMAZON_ALL_BEAUTY_500 = "amazon_all_beauty_500"
    AMAZON_ALL_BEAUTY_1000 = "amazon_all_beauty_1000"

    AMAZON_FASHION_100 = "amazon_fashion_100"    
    AMAZON_FASHION_500 = "amazon_fashion_500"
    AMAZON_FASHION_1000 = "amazon_fashion_1000"    

    AMAZON_APPLIANCES_100 = "amazon_appliances_100"    
    AMAZON_APPLIANCES_500 = "amazon_appliances_500"
    AMAZON_APPLIANCES_1000 = "amazon_appliances_1000"

    AMAZON_ARTS_CRAFTS_AND_SEWING_100 = "amazon_arts_crafts_and_sewing_100"    
    AMAZON_ARTS_CRAFTS_AND_SEWING_500 = "amazon_arts_crafts_and_sewing_500"
    AMAZON_ARTS_CRAFTS_AND_SEWING_1000 = "amazon_arts_crafts_and_sewing_1000"

    AMAZON_AUTOMOTIVE_100 = "amazon_automotive_100"    
    AMAZON_AUTOMOTIVE_500 = "amazon_automotive_500"
    AMAZON_AUTOMOTIVE_1000 = "amazon_automotive_1000"

    AMAZON_BABY_PRODUCTS_100 = "amazon_baby_products_100"
    AMAZON_BABY_PRODUCTS_500 = "amazon_baby_products_500"
    AMAZON_BABY_PRODUCTS_1000 = "amazon_baby_products_1000"

    AMAZON_BEAUTY_AND_PERSONAL_CARE_100 = "amazon_beauty_and_personal_care_100"    
    AMAZON_BEAUTY_AND_PERSONAL_CARE_500 = "amazon_beauty_and_personal_care_500"
    AMAZON_BEAUTY_AND_PERSONAL_CARE_1000 = "amazon_beauty_and_personal_care_1000"

    AMAZON_BOOKS_100 = "amazon_books_100"    
    AMAZON_BOOKS_500 = "amazon_books_500"
    AMAZON_BOOKS_1000 = "amazon_books_1000"

    AMAZON_VIDEO_GAMES_100 = "amazon_video_games_100"    
    AMAZON_VIDEO_GAMES_500 = "amazon_video_games_500"
    AMAZON_VIDEO_GAMES_1000 = "amazon_video_games_1000"
    
