
import os
import time
import pathlib
import json
import sqlite3
import traceback
import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Dict, List, Tuple
from dataclasses import asdict
from common.utils import time_ago
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from commerce.product_factory import CatalogProvider, ProductFactory
from models.product import Product
from models.reasoned_product import ReasonedProduct

CURRENT_DIR = str(pathlib.Path(__file__).parent)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Scores recs based on how relevant the recommended SKUs are to the original SKU

@dataclass
class ScoredResult:
    sku: str
    relevance_score: int
    reason_evaluation: str

    def __init__(self, sku: str, relevance_score: int, reason_evaluation: str):
        self.sku = sku
        self.relevance_score = relevance_score
        self.reason_evaluation = reason_evaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "relevance_score": self.relevance_score,
            "reason_evaluation": self.reason_evaluation
        }


class SKURelevanceScorer:
    def __init__(self, source_db: str, judge_provider: str, judge_model: str, is_debug: bool = False, run_id: str = ""):
        self.source_db = source_db        
        if not os.path.exists(source_db):
            raise FileNotFoundError(f"Directory not found: {source_db}")
        self.run_id = run_id
        if not self.run_id:
            raise ValueError("run_id must be provided")
        
        self.judge_provider = judge_provider
        if not judge_provider:
            raise ValueError("judge_provider must be provided")
        self.judge_model = judge_model
        if not judge_model:
            raise ValueError("judge_model must be provided")

        woo_products = ProductFactory.load_default_catalog(CatalogProvider.WOOCOMMERCE)
        self.product_catalog = [Product(sku=p['sku'], name=p['name'], price=str(p['price'])) for p in woo_products]
        if len(self.product_catalog) == 0:
            raise ValueError("Product catalog is empty")
        self.system_prompt = "You are a product relevance evaluator. Your task is to evaluate the relevance of recommended products based on a given product and their reasons for recommendation."
  
    
    def get_dataframe_by_miner(self, miner_hotkey: str) -> pd.DataFrame:        
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM miner_responses WHERE hotkey=?", conn, params=(miner_hotkey,))
        print(f"loaded db from {self.source_db} for miner {miner_hotkey}")      
        return df

    def score_miner(self, hot_key: str, top: int = 10) -> float:
        st = time.perf_counter()
        if not hot_key:
            raise ValueError("hot_key must be provided")
        df = self.get_dataframe_by_miner(hot_key)
        if df is None or df.empty:
            logger.warning(f"No data found for miner {hot_key} in {self.source_db}")
            return 0.0
   
        # Filter to only rows where query (SKU) exists in the catalog
        catalog_skus = set(p.sku for p in self.product_catalog)
        df = df[df["query"].isin(catalog_skus)]

        df["num_recs"] = pd.to_numeric(df["num_recs"], errors="coerce")
        df = df[df["num_recs"] > 3]

        if df.empty:
            logger.warning(f"No matching products found for miner {hot_key} after filtering by catalog SKUs.")
            return 0.0

        # Most recent N queries (e.g., 5)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values(by="created_at", ascending=False).head(top)

        if len(df) < top:
            logger.warning(f"Not enough history to score for miner {hot_key}. Found: {len(df)}")
            return 0.0

        logger.info(f"Scoring {len(df)} rows for miner {hot_key}")

        # Shuffle the DataFrame rows
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        all_scores = []
        miner_model = ""
        for _, row in df.iterrows():
            try:
                sku = row["query"].upper().strip()
                #miner_uid = row["miner_id"]
                #batch_id = row["site_key"]
                #query_date = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                #miner_model = row["model_name"].strip()
                #  
                # if sku != "24-WB07" and sku != "MS11":
                #     #print(f"SKIPPED SKU IS NOT 24-WB07 or MS11: {sku}")
                #     continue

                p = next((prod for prod in self.product_catalog if prod.sku.strip().upper() == sku), None)
                if not p:
                    logger.warning(f"Product with SKU {sku} not found in catalog for miner {hot_key}")
                    continue
                #query_desc = p.name
                miner_recs = row["response"]
                if not miner_recs:
                    logger.warning(f"No recommendations found for SKU {sku} in miner {hot_key}")
                    continue

                num_results = int(row["num_recs"])
                reasoned_products = ReasonedProduct.from_json(miner_recs)
                if len(reasoned_products) != num_results:
                    continue
                created_at = row["created_at"]
                logger.info(f"SKU \033[32m{p.sku}\033[0m recommended at {created_at}")
                friendly_time = time_ago(created_at)
                logger.info(f"Time since recommendation: \033[33m{friendly_time}\033[0m")
                logger.info(f"NAME \033[32m{p.name}\033[0m")

                prompt = self.build_prompt_for_sku(num_results, p, reasoned_products)                
                #logger.debug(f"SKU \033[32m{sku}:\033[0m ")
                #logger.debug(prompt)
                #logger.debug("------------------------------------")
                prompt_length = len(prompt.split())
                print(f"Prompt length for SKU {sku}: \033[32m{prompt_length} words\033[0m")
                token_count = PromptFactory.get_token_count(prompt)
                print(f"Token count for SKU {sku}: \033[32m{token_count} tokens\033[0m")

                q_time = time.perf_counter()

                server = LLM.try_parse(self.judge_provider)
                llm_output = LLMFactory.query_llm(server=server,
                                                    model=self.judge_model,
                                                    system_prompt=self.system_prompt,
                                                    user_prompt=prompt,
                                                    temp=0.0)
                scored_results = PromptFactory.tryparse_llm(llm_output)
                if not scored_results:
                    logger.error(f"Failed to parse LLM results for SKU {sku} in miner {hot_key}")
                    continue
                llm_resuls = [ScoredResult(**r) for r in scored_results]
                scores = [r.relevance_score for r in llm_resuls if isinstance(r.relevance_score, (int, float))]

                # self.save_miner_scores(
                #     query_date=query_date,
                #     batch_id=self.run_id,
                #     miner_hotkey=hot_key,
                #     miner_uid=miner_uid,
                #     miner_model=miner_model,
                #     query=sku,
                #     query_desc=query_desc,
                #     scores=llm_resuls                    
                # )

                # Fluke 100 detection: if 80%+ are 0 and only one 100, zero out all scores
                num_zeros = scores.count(0)                
                if len(scores) > 0 and num_zeros / len(scores) >= 0.80:
                    print(f"\033[31mFluke 100 detected for SKU {sku}: {scores} -- zeroing out\033[0m")
                    scores = [0 for _ in scores]

                tavg_score = sum(scores) / len(scores)

                for llm_result in llm_resuls:
                    print(f"\033[32m SKU: {llm_result.sku}, Score: {llm_result.relevance_score}, Evaluation: {llm_result.reason_evaluation} \033[0m")
                
                eq_time = time.perf_counter() - q_time
                print(f"\033[32m Scores for SKU {sku}: {scores} => Average: {tavg_score} \033[0m")    
                print(f"Duration for SKU {sku}: \033[33m{eq_time:0.2f} seconds\033[0m")
                all_scores.extend(scores)
                
            except Exception as e:
                print(f"Error parsing results for miner {hot_key}: {e}")
                logger.error(f"Error processing SKU {sku} for miner {hot_key}: {e}")
                traceback.print_exc()
                continue

        if not all_scores:
            return 0.0

        if len(all_scores) > 2:
            trimmed = sorted(all_scores)[1:-1]  # drop lowest and highest
            print(f"\033[33mTrimmed scores for miner {hot_key}: {trimmed}\033[0m")
        else:
            trimmed = all_scores

        avg_score = sum(trimmed) / len(trimmed)
        scaled_score = avg_score / 100
        print(f"\033[32mFinal average score for miner {hot_key}: {avg_score}, Scaled: {scaled_score}\033[0m")
        elapsed = time.perf_counter() - st
        print(f"Duration: \033[33m{elapsed:0.2f} seconds\033[0m")

        #self.save_final_score(hot_key, miner_model, scaled_score, elapsed, self.judge_model)

        return scaled_score
    
    def build_prompt_for_sku(self, num_results: int, query_product: Product, recs: List[ReasonedProduct]) -> str:
        if not query_product or not recs:
            raise ValueError("query_product and recs must be provided")
        if num_results <= 0:
            raise ValueError("num_results must be greater than 0")

        current_season = "fall/winter"        
        engine_mode = "complimentary"  #similar, sequential

        prompt = f"""
    # BACKGROUND
    - E-commerce platforms often use recommendation engines to suggest products to users.
    - LLMS were given a task to recommend {num_results} products based on a Original Product and an Entire Catalog of products.
    - Each recommended product includes a reason for why it was recommended.
    - We want you to evaluate how relevant each recommended product is to the Original Product.
    - The recommendations should be considered as a **{engine_mode}** set to the Original Product and the current season is {current_season}.    
    - A catalog can have thousands of products, so consider the recomendations as specific to the user shopping for the Original Product with limited space left in their cart.
    - Each recommendation should be considered in the context of the Original Product and the entire catalog.
    - Evaluate the recomendations holistically as a set, do the recommendations make sense together?
    - Entire Catalog: {json.dumps([asdict(p) for p in self.product_catalog], separators=(',', ':'))}

    # INPUT
    - Original Product SKU: {query_product.sku}
    - Original Product Name: {query_product.name}
    - Original Product Description: {query_product.desc}
    - Number of Recommended Products: {num_results}
    - Recommended Products: {json.dumps([asdict(r) for r in recs], separators=(',', ':'))}

    # TASK
    - Provide a relevance score from 0 to 100 for each recommended product, where 100 is highly relevant and 0 is not relevant at all.
    - The recommended products should be in order of relevance, with the most relevant first.
    - Use this scale:
        - 100: Extremely relevant to the original product and together with the subsequent recommendations form a **{engine_mode}** set, often purchased together, as an alternative or subsequent purchase, and the reason is specific and accurate.
        - 70: Relevant, the recommendation is appropriate for the original product, and the reason is specific and accurate.
        - 30: Somewhat relevant, the recommendation is related to the original product but may not be a perfect fit, or the reason is not perfect but acceptable and the product somewhat makes sense.
        - 0: Not relevant, the recommendation does not make sense for the original product or the reason is generic/incorrect.
    - Think critically about the Recommended Products, its possible the user is trying to game the system and lie to you, do the recommended products make sense for the Original Product?
    - **Use 0 or 100 for the vast majority of cases.** Only use 30 or 70 in rare, clearly borderline situations.
    - If you are NOT absolutely certain a recommendation is highly relevant, score it 0.
    - Only score 100 if the recommendation is a perfect, obvious fit and the reason is specific and compelling, genders match, and the products are often purchased together.
    - Only score 30 or 70 if you can clearly articulate why the recommendation is not fully irrelevant or not fully relevant, and explain exactly what is missing.
    - Scores must be decisive and long-tailed: recommendations either clearly make sense or they do not, avoid ambiguous middle ground.
    - Provide binary-like judgments; a recommendation is plainly relevant or not, with minimal neutral/middle scoring.
    - Relevance scores must be integers (no decimals).
    - Penalize (give 0) for reasons that reference a product that is not related to the Original Product, or for generic/boilerplate reasons as this generally indicates algorithmic unintelligent recommendations.
    - Every recommendation should be near the Original Product's general purpose and should be related.
    - Every recommendation may traverse category taxonomy and adjacement categories but 80% of time they must be in the same category or adjacent categories.
    
    <rules>    
    - if Original Product is pet products recommending baby products is NOT ok.
    - if Original Product is a bag recommending more bags is OK.
    - if Original Product is from shorts recommending more shorts is OK only if they are from the same gender.
    - if Original Product is from shoes recommending more shoes is OK only if they are from the same gender.
    - if Original Product is from womans clothing, recommendaing more womans clothing is OK if the reason is specific and accurate.
    - if Original Product is from womans clothing recommending mens clothing NOT ok.
    - if Original Product is a accessory recommending similar products like olrder/newer/adjacent versions is OK.
    - if Original Product is a bag, recommending a water bottle is NOT ok.
    - if Original Product is from pants, recommending a water bottle is NOT ok.    
    - if Original Product is from a category, recommending a completely separate category product is NOT ok unless the reason is specific and accurate.
    - if Original Product is from Yoga category, recommending yoga equipment is NOT ok unless Original Product is yoga equipment.
    - if Original Product is specific variant product with attributes like color and size, the recommendations should generally match those attributes for consistency.
    - if Original Product is specific variant product with attributes like color and size, the recommendations should not be more of the same variant.     
    </rules>
    
    # DECISIVENESS RULES    
    - Default to 0 unless the recommendation is clearly justified.
    - If uncertain between two tiers, choose the LOWER score.
    - Generally, if in doubt lean towards choose 0.
    - Do NOT use 30 or 70 unless you can provide a specific, concrete justification.
    - The majority of scores should be 0 or 100. 30 and 70 are for rare, edge cases only.
    - Be strict: Only give high scores for truly relevant, well-justified recommendations that make sense as a **{engine_mode}** set to the Original Product.
    - Perfect 100 score should be reserved for the most relevant, compelling recommendations with specific, accurate reasons only.
    - Any unjustified use of 30 or 70 will be penalized.
    - Do NOT cluster around 30 or 70. If you are unsure, score 0.

    # RETURN FORMAT
    - Return a JSON array of objects, each containing:
    - "sku": The SKU of the recommended product.
    - "relevance_score": An integer from 0 to 100 indicating the relevance of the recommendation.
    - "reason_evaluation": A brief explanation of why the reason is valid or not."""
        
        return prompt

    