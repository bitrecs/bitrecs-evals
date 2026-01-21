import logging
from datetime import datetime, timezone
from typing import List, Dict

from common import constants as CONST
from evals.amazon_all_beauty_100 import AmazonAllBeauty100
from evals.amazon_all_beauty_1000 import AmazonAllBeauty1000
from evals.amazon_all_beauty_500 import AmazonAllBeauty500
from evals.amazon_appliances_100 import AmazonAppliances100
from evals.amazon_appliances_1000 import AmazonAppliances1000
from evals.amazon_appliances_500 import AmazonAppliances500
from evals.amazon_arts_crafts_and_sewing_100 import AmazonArtsCraftsAndSewing100
from evals.amazon_arts_crafts_and_sewing_1000 import AmazonArtsCraftsAndSewing1000
from evals.amazon_arts_crafts_and_sewing_500 import AmazonArtsCraftsAndSewing500
from evals.amazon_automotive_100 import AmazonAutomotive100
from evals.amazon_automotive_1000 import AmazonAutomotive1000
from evals.amazon_automotive_500 import AmazonAutomotive500
from evals.amazon_baby_products_100 import AmazonBabyProducts100
from evals.amazon_baby_products_1000 import AmazonBabyProducts1000
from evals.amazon_baby_products_500 import AmazonBabyProducts500
from evals.amazon_beauty_and_personal_care_100 import AmazonBeautyAndPersonalCare100
from evals.amazon_beauty_and_personal_care_1000 import AmazonBeautyAndPersonalCare1000
from evals.amazon_beauty_and_personal_care_500 import AmazonBeautyAndPersonalCare500
from evals.amazon_books_100 import AmazonBooks100
from evals.amazon_books_1000 import AmazonBooks1000
from evals.amazon_books_500 import AmazonBooks500
from evals.amazon_cds_and_vinyl_100 import AmazonCdsAndVinyl100
from evals.amazon_cds_and_vinyl_1000 import AmazonCdsAndVinyl1000
from evals.amazon_cds_and_vinyl_500 import AmazonCdsAndVinyl500
from evals.amazon_cellphones_and_accessories_100 import AmazonCellPhonesAndAccessories100
from evals.amazon_cellphones_and_accessories_1000 import AmazonCellPhonesAndAccessories1000
from evals.amazon_cellphones_and_accessories_500 import AmazonCellPhonesAndAccessories500
from evals.amazon_clothing_shoes_and_jewelry_100 import AmazonClothingShoesAndJewelry100
from evals.amazon_clothing_shoes_and_jewelry_1000 import AmazonClothingShoesAndJewelry1000
from evals.amazon_clothing_shoes_and_jewelry_500 import AmazonClothingShoesAndJewelry500
from evals.amazon_digital_music_100 import AmazonDigitalMusic100
from evals.amazon_digital_music_1000 import AmazonDigitalMusic1000
from evals.amazon_digital_music_500 import AmazonDigitalMusic500
from evals.amazon_electronics_100 import AmazonElectronics100
from evals.amazon_electronics_1000 import AmazonElectronics1000
from evals.amazon_electronics_500 import AmazonElectronics500
from evals.amazon_fashion_100 import AmazonFashion100
from evals.amazon_fashion_1000 import AmazonFashion1000
from evals.amazon_fashion_500 import AmazonFashion500
from evals.amazon_grocery_and_gourmet_food_100 import AmazonGroceryAndGourmetFood100
from evals.amazon_grocery_and_gourmet_food_1000 import AmazonGroceryAndGourmetFood1000
from evals.amazon_grocery_and_gourmet_food_500 import AmazonGroceryAndGourmetFood500
from evals.amazon_handmade_products_100 import AmazonHandmadeProducts100
from evals.amazon_handmade_products_1000 import AmazonHandmadeProducts1000
from evals.amazon_handmade_products_500 import AmazonHandmadeProducts500
from evals.amazon_health_and_household_100 import AmazonHealthAndHousehold100
from evals.amazon_health_and_household_1000 import AmazonHealthAndHousehold1000
from evals.amazon_health_and_household_500 import AmazonHealthAndHousehold500
from evals.amazon_health_and_personal_care_100 import AmazonHealthAndPersonalCare100
from evals.amazon_health_and_personal_care_1000 import AmazonHealthAndPersonalCare1000
from evals.amazon_health_and_personal_care_500 import AmazonHealthAndPersonalCare500
from evals.amazon_home_and_kitchen_100 import AmazonHomeAndKitchen100
from evals.amazon_home_and_kitchen_1000 import AmazonHomeAndKitchen1000
from evals.amazon_home_and_kitchen_500 import AmazonHomeAndKitchen500
from evals.amazon_industrial_and_scientific_100 import AmazonIndustrialAndScientific100
from evals.amazon_industrial_and_scientific_1000 import AmazonIndustrialAndScientific1000
from evals.amazon_industrial_and_scientific_500 import AmazonIndustrialAndScientific500
from evals.amazon_movies_and_tv_100 import AmazonMoviesAndTV100
from evals.amazon_movies_and_tv_1000 import AmazonMoviesAndTV1000
from evals.amazon_movies_and_tv_500 import AmazonMoviesAndTV500
from evals.amazon_musical_instruments_100 import AmazonMusicalInstruments100
from evals.amazon_musical_instruments_1000 import AmazonMusicalInstruments1000
from evals.amazon_musical_instruments_500 import AmazonMusicalInstruments500
from evals.amazon_office_products_100 import AmazonOfficeProducts100
from evals.amazon_office_products_1000 import AmazonOfficeProducts1000
from evals.amazon_office_products_500 import AmazonOfficeProducts500
from evals.amazon_patio_lawn_and_garden_100 import AmazonPatioLawnAndGarden100
from evals.amazon_patio_lawn_and_garden_1000 import AmazonPatioLawnAndGarden1000
from evals.amazon_patio_lawn_and_garden_500 import AmazonPatioLawnAndGarden500
from evals.amazon_pet_supplies_100 import AmazonPetSupplies100
from evals.amazon_patio_lawn_and_garden_1000 import AmazonPatioLawnAndGarden1000
from evals.amazon_pet_supplies_1000 import AmazonPetSupplies1000
from evals.amazon_pet_supplies_500 import AmazonPetSupplies500
from evals.amazon_software_100 import AmazonSoftware100
from evals.amazon_software_1000 import AmazonSoftware1000
from evals.amazon_software_500 import AmazonSoftware500
from evals.amazon_sports_and_outdoors_100 import AmazonSportsAndOutdoors100
from evals.amazon_sports_and_outdoors_1000 import AmazonSportsAndOutdoors1000
from evals.amazon_sports_and_outdoors_500 import AmazonSportsAndOutdoors500
from evals.amazon_tools_and_home_improvement_100 import AmazonToolsAndHomeImprovement100
from evals.amazon_tools_and_home_improvement_1000 import AmazonToolsAndHomeImprovement1000
from evals.amazon_tools_and_home_improvement_500 import AmazonToolsAndHomeImprovement500
from evals.amazon_toys_and_games_100 import AmazonToysAndGames100
from evals.amazon_toys_and_games_1000 import AmazonToysAndGames1000
from evals.amazon_toys_and_games_500 import AmazonToysAndGames500
from evals.amazon_video_games_100 import AmazonVideoGames100
from evals.amazon_video_games_1000 import AmazonVideoGames1000
from evals.amazon_video_games_500 import AmazonVideoGames500
from evals.bitrecs_basic_eval import BitrecsBasicEval
from evals.bitrecs_prompt_eval import BitrecsPromptEval
from evals.bitrecs_reason_eval import BitrecsReasonEval
from evals.bitrecs_safe_prompt import BitrecsSafeEval
from evals.bitrecs_sku_eval import BitrecsSkuEval
from evals.eval_result import EvalResult
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

class EvalFactory:
    """
    Factory for creating and running evals.
    Supports dynamic registration of new evals.
    """
    
    _registry: Dict[BitrecsEvaluationType, type] = {
        BitrecsEvaluationType.BITRECS_BASIC_DAILY: BitrecsBasicEval,
        BitrecsEvaluationType.BITRECS_SAFE_DAILY: BitrecsSafeEval,
        BitrecsEvaluationType.BITRECS_PROMPT_DAILY: BitrecsPromptEval,
        BitrecsEvaluationType.BITRECS_REASON_DAILY: BitrecsReasonEval,
        BitrecsEvaluationType.BITRECS_SKU_DAILY: BitrecsSkuEval,        
        
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_100: AmazonAllBeauty100,
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_500: AmazonAllBeauty500,  
        BitrecsEvaluationType.AMAZON_ALL_BEAUTY_1000: AmazonAllBeauty1000,

        BitrecsEvaluationType.AMAZON_FASHION_100: AmazonFashion100,
        BitrecsEvaluationType.AMAZON_FASHION_500: AmazonFashion500,
        BitrecsEvaluationType.AMAZON_FASHION_1000: AmazonFashion1000,

        BitrecsEvaluationType.AMAZON_APPLIANCES_100: AmazonAppliances100,
        BitrecsEvaluationType.AMAZON_APPLIANCES_500: AmazonAppliances500,
        BitrecsEvaluationType.AMAZON_APPLIANCES_1000: AmazonAppliances1000,

        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_100: AmazonArtsCraftsAndSewing100,
        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_500: AmazonArtsCraftsAndSewing500,
        BitrecsEvaluationType.AMAZON_ARTS_CRAFTS_AND_SEWING_1000: AmazonArtsCraftsAndSewing1000,

        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_100: AmazonAutomotive100,
        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_500: AmazonAutomotive500,
        BitrecsEvaluationType.AMAZON_AUTOMOTIVE_1000: AmazonAutomotive1000,

        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_100: AmazonBabyProducts100,
        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_500: AmazonBabyProducts500,
        BitrecsEvaluationType.AMAZON_BABY_PRODUCTS_1000: AmazonBabyProducts1000,

        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_100: AmazonBeautyAndPersonalCare100,
        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_500: AmazonBeautyAndPersonalCare500,
        BitrecsEvaluationType.AMAZON_BEAUTY_AND_PERSONAL_CARE_1000: AmazonBeautyAndPersonalCare1000,

        BitrecsEvaluationType.AMAZON_BOOKS_100: AmazonBooks100,
        BitrecsEvaluationType.AMAZON_BOOKS_500: AmazonBooks500,
        BitrecsEvaluationType.AMAZON_BOOKS_1000: AmazonBooks1000,

        BitrecsEvaluationType.AMAZON_CDS_AND_VINYL_100: AmazonCdsAndVinyl100,
        BitrecsEvaluationType.AMAZON_CDS_AND_VINYL_500: AmazonCdsAndVinyl500,
        BitrecsEvaluationType.AMAZON_CDS_AND_VINYL_1000: AmazonCdsAndVinyl1000,

        BitrecsEvaluationType.AMAZON_CELL_PHONES_AND_ACCESSORIES_100: AmazonCellPhonesAndAccessories100,
        BitrecsEvaluationType.AMAZON_CELL_PHONES_AND_ACCESSORIES_500: AmazonCellPhonesAndAccessories500,
        BitrecsEvaluationType.AMAZON_CELL_PHONES_AND_ACCESSORIES_1000: AmazonCellPhonesAndAccessories1000,

        BitrecsEvaluationType.AMAZON_CLOTHING_SHOES_AND_JEWELRY_100: AmazonClothingShoesAndJewelry100,
        BitrecsEvaluationType.AMAZON_CLOTHING_SHOES_AND_JEWELRY_500: AmazonClothingShoesAndJewelry500,
        BitrecsEvaluationType.AMAZON_CLOTHING_SHOES_AND_JEWELRY_1000: AmazonClothingShoesAndJewelry1000,      
        BitrecsEvaluationType.AMAZON_DIGITAL_MUSIC_100: AmazonDigitalMusic100,
        BitrecsEvaluationType.AMAZON_DIGITAL_MUSIC_500: AmazonDigitalMusic500,
        BitrecsEvaluationType.AMAZON_DIGITAL_MUSIC_1000: AmazonDigitalMusic1000,

        BitrecsEvaluationType.AMAZON_ELECTRONICS_100: AmazonElectronics100,
        BitrecsEvaluationType.AMAZON_ELECTRONICS_500: AmazonElectronics500,
        BitrecsEvaluationType.AMAZON_ELECTRONICS_1000: AmazonElectronics1000,

        BitrecsEvaluationType.AMAZON_GROCERY_AND_GOURMET_FOOD_100: AmazonGroceryAndGourmetFood100,
        BitrecsEvaluationType.AMAZON_GROCERY_AND_GOURMET_FOOD_500: AmazonGroceryAndGourmetFood500,
        BitrecsEvaluationType.AMAZON_GROCERY_AND_GOURMET_FOOD_1000: AmazonGroceryAndGourmetFood1000,

        BitrecsEvaluationType.AMAZON_HANDMADE_PRODUCTS_100: AmazonHandmadeProducts100,
        BitrecsEvaluationType.AMAZON_HANDMADE_PRODUCTS_500: AmazonHandmadeProducts500,
        BitrecsEvaluationType.AMAZON_HANDMADE_PRODUCTS_1000: AmazonHandmadeProducts1000,

        BitrecsEvaluationType.AMAZON_HEALTH_AND_HOUSEHOLD_100: AmazonHealthAndHousehold100,
        BitrecsEvaluationType.AMAZON_HEALTH_AND_HOUSEHOLD_500: AmazonHealthAndHousehold500,
        BitrecsEvaluationType.AMAZON_HEALTH_AND_HOUSEHOLD_1000: AmazonHealthAndHousehold1000,

        BitrecsEvaluationType.AMAZON_HEALTH_AND_PERSONAL_CARE_100: AmazonHealthAndPersonalCare100,
        BitrecsEvaluationType.AMAZON_HEALTH_AND_PERSONAL_CARE_500: AmazonHealthAndPersonalCare500,
        BitrecsEvaluationType.AMAZON_HEALTH_AND_PERSONAL_CARE_1000: AmazonHealthAndPersonalCare1000,

        BitrecsEvaluationType.AMAZON_HOME_AND_KITCHEN_100: AmazonHomeAndKitchen100,
        BitrecsEvaluationType.AMAZON_HOME_AND_KITCHEN_500: AmazonHomeAndKitchen500,
        BitrecsEvaluationType.AMAZON_HOME_AND_KITCHEN_1000: AmazonHomeAndKitchen1000,

        BitrecsEvaluationType.AMAZON_INDUSTRIAL_AND_SCIENTIFIC_100: AmazonIndustrialAndScientific100,
        BitrecsEvaluationType.AMAZON_INDUSTRIAL_AND_SCIENTIFIC_500: AmazonIndustrialAndScientific500,
        BitrecsEvaluationType.AMAZON_INDUSTRIAL_AND_SCIENTIFIC_1000: AmazonIndustrialAndScientific1000,

        BitrecsEvaluationType.AMAZON_MOVIES_AND_TV_100: AmazonMoviesAndTV100,
        BitrecsEvaluationType.AMAZON_MOVIES_AND_TV_500: AmazonMoviesAndTV500,
        BitrecsEvaluationType.AMAZON_MOVIES_AND_TV_1000: AmazonMoviesAndTV1000,

        BitrecsEvaluationType.AMAZON_MUSICAL_INSTRUMENTS_100: AmazonMusicalInstruments100,
        BitrecsEvaluationType.AMAZON_MUSICAL_INSTRUMENTS_500: AmazonMusicalInstruments500,
        BitrecsEvaluationType.AMAZON_MUSICAL_INSTRUMENTS_1000: AmazonMusicalInstruments1000,

        BitrecsEvaluationType.AMAZON_OFFICE_PRODUCTS_100: AmazonOfficeProducts100,
        BitrecsEvaluationType.AMAZON_OFFICE_PRODUCTS_500: AmazonOfficeProducts500,
        BitrecsEvaluationType.AMAZON_OFFICE_PRODUCTS_1000: AmazonOfficeProducts1000,

        BitrecsEvaluationType.AMAZON_PATIO_LAWN_AND_GARDEN_100: AmazonPatioLawnAndGarden100,
        BitrecsEvaluationType.AMAZON_PATIO_LAWN_AND_GARDEN_500: AmazonPatioLawnAndGarden500,
        BitrecsEvaluationType.AMAZON_PATIO_LAWN_AND_GARDEN_1000: AmazonPatioLawnAndGarden1000,

        BitrecsEvaluationType.AMAZON_PET_SUPPLIES_100: AmazonPetSupplies100,
        BitrecsEvaluationType.AMAZON_PET_SUPPLIES_500: AmazonPetSupplies500,
        BitrecsEvaluationType.AMAZON_PET_SUPPLIES_1000: AmazonPetSupplies1000,

        BitrecsEvaluationType.AMAZON_SOFTWARE_100: AmazonSoftware100,
        BitrecsEvaluationType.AMAZON_SOFTWARE_500: AmazonSoftware500,
        BitrecsEvaluationType.AMAZON_SOFTWARE_1000: AmazonSoftware1000,

        BitrecsEvaluationType.AMAZON_SPORTS_AND_OUTDOORS_100: AmazonSportsAndOutdoors100,        
        BitrecsEvaluationType.AMAZON_SPORTS_AND_OUTDOORS_500: AmazonSportsAndOutdoors500,
        BitrecsEvaluationType.AMAZON_SPORTS_AND_OUTDOORS_1000: AmazonSportsAndOutdoors1000,

        BitrecsEvaluationType.AMAZON_TOOLS_AND_HOME_IMPROVEMENT_100: AmazonToolsAndHomeImprovement100,        
        BitrecsEvaluationType.AMAZON_TOYS_AND_GAMES_100: AmazonToysAndGames100,        
        BitrecsEvaluationType.AMAZON_TOYS_AND_GAMES_500: AmazonToysAndGames500,
        BitrecsEvaluationType.AMAZON_TOYS_AND_GAMES_1000: AmazonToysAndGames1000,
        
        BitrecsEvaluationType.AMAZON_TOOLS_AND_HOME_IMPROVEMENT_500: AmazonToolsAndHomeImprovement500,
        BitrecsEvaluationType.AMAZON_TOOLS_AND_HOME_IMPROVEMENT_1000: AmazonToolsAndHomeImprovement1000,
        
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_100: AmazonVideoGames100,        
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_500: AmazonVideoGames500,
        BitrecsEvaluationType.AMAZON_VIDEO_GAMES_1000: AmazonVideoGames1000,

    }
    
    @classmethod
    def register_eval(cls, name: BitrecsEvaluationType, eval_class: type):
        """Register a new eval class."""
        cls._registry[name] = eval_class
    
    @classmethod
    def run_eval(cls, eval_type: BitrecsEvaluationType, miner_artifact: Artifact, run_id: str, max_iterations: int = 10) -> EvalResult:
        """Create and run a specific eval."""
        if eval_type not in cls._registry:
            raise ValueError(f"Unknown eval type: {eval_type}")
        
        eval_instance = cls._registry[eval_type](run_id, miner_artifact)
        return eval_instance.run(max_iterations)
    
    @classmethod
    def run_all_evals(cls, run_id: str, miner_artifact: Artifact, eval_types: List[BitrecsEvaluationType] = None, max_iterations: int = 10) -> List[EvalResult]:
        """Run multiple evals and return aggregated results."""
        if eval_types is None:
            raise ValueError("eval_types must be provided")
            #eval_types = list(cls._registry.keys())
        
        results = []        
        for eval_type in eval_types:
            try:
                logger.debug(f"\033[34mRunning eval type: {eval_type}\033[0m")
                result = cls.run_eval(eval_type, miner_artifact, run_id, max_iterations)
                results.append(result)                
            except Exception as e:
                # Log error and continue (don't fail all evals)
                logger.error(f"Failed to run {eval_type} eval: {e}")
                results.append(EvalResult(
                    eval_name=f"{eval_type} Eval",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    hot_key=miner_artifact.miner_hotkey,
                    score=0.0,
                    passed=False,
                    rows_evaluated=0,
                    details=f"FAIL - Error: {e}",
                    duration_seconds=0.0,
                    run_id=run_id                    
                ))
        return results