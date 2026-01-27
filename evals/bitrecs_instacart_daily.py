import os
import random
import time
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from common import constants as CONST
from common.r2 import download_text_file_from_r2
from evals.eval_result import EvalResult
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from evals.base_eval import BaseEval
from models.product import Product


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Evaluates instacart dataset

check: benchmark artifact against instacart data
data: instacart dataset files

"""

class BitrecsInstacartEval(BaseEval):

    @property
    def sample_size(self) -> int:
        return 3   
    
    @property
    def num_recs(self) -> int:
        return 20
    
    @property
    def pass_threshold(self) -> float:
        return 0.08 #Recall@20


    def __init__(self, run_id: str, miner_artifact: Artifact = None):      
        super().__init__(run_id, miner_artifact)

        data_success = self.data_init()
        if not data_success:
            raise RuntimeError("Data initialization failed. Required files may be missing.")    

        app_root = Path(__file__).parent.parent
        data_dir = app_root / 'data' / 'instacart'
        
        PRODUCTS_PATH = data_dir / 'products.csv'
        ORDERS_PATH = data_dir / 'orders.csv'
        ORDER_PRODUCTS_PRIOR_PATH = data_dir / 'order_products__prior.csv'
        ORDER_PRODUCTS_TRAIN_PATH = data_dir / 'order_products__train.csv'

        self.products = pd.read_csv(PRODUCTS_PATH)[['product_id', 'product_name']]
        if self.products.empty:
            raise ValueError("Products dataframe is empty after loading products.csv")
        self.product_map = dict(zip(self.products['product_id'], self.products['product_name']))
        if not self.product_map:
            raise ValueError("Product map is empty after loading products.csv")        
        self.orders = pd.read_csv(ORDERS_PATH)
        if self.orders.empty:
            raise ValueError("Orders dataframe is empty after loading orders.csv")
        
        bproducts = self.convert_products()
        print(f"Loaded {len(bproducts):,} products from Instacart dataset.")
        
        # Train: all prior purchases
        order_products_prior = pd.read_csv(ORDER_PRODUCTS_PRIOR_PATH)
        train = order_products_prior.merge(self.orders[['order_id', 'user_id']], on='order_id')
        train = train[['user_id', 'product_id', 'order_id']]
        if train.empty:
            raise ValueError("Train dataframe is empty after merging order_products__prior.csv with orders.csv")
        self.train = train
        
        # Test: last purchases (from train.csv, which has the last order per user)
        order_products_train = pd.read_csv(ORDER_PRODUCTS_TRAIN_PATH)
        test = order_products_train.merge(self.orders[['order_id', 'user_id']], on='order_id')
        test = test[['user_id', 'product_id', 'order_id']]
        if test.empty:
            raise ValueError("Test dataframe is empty after merging order_products__train.csv with orders.csv")
        self.test = test
        
        logger.info(f"Train: {len(self.train):,} rows | Test: {len(self.test):,} rows")
        logger.info(f"Users in test: {self.test['user_id'].nunique():,}")
      
        TOP_POPULAR = 20        # Top products for popularity baseline
        LLM_TOP_POPULAR = 5000  # Top products to show LLM
        
        self.top_popular = self.train['product_id'].value_counts().index[:TOP_POPULAR].tolist()
        self.top_products_llm = self.train['product_id'].value_counts().head(LLM_TOP_POPULAR).index.tolist()

    
    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_INSTACART_DAILY
            
        
    def data_init(self) -> bool:
        """
        Checks if all 6 required CSV files are present in data/instacart/.
        If any are missing, downloads them from the R2 bucket using the helper.
        """
        required_files = [
            'aisles.csv',
            'departments.csv',
            'order_products__prior.csv',
            'order_products__train.csv',
            'orders.csv',
            'products.csv'
        ]
        try:
            app_root = Path(__file__).parent.parent
            data_dir = app_root / 'data' / 'instacart'
            data_dir.mkdir(parents=True, exist_ok=True)            
            
            bucket = os.getenv('R2_BUCKET_NAME')
            access_key_id = os.getenv('R2_ACCESS_KEY_ID')
            secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
            endpoint_url = os.getenv('R2_ENDPOINT_URL')            
            if not all([bucket, access_key_id, secret_access_key, endpoint_url]):
                raise ValueError("R2 environment variables not set. Check .env file.")
            
            all_present = True
            for filename in required_files:
                file_path = data_dir / filename
                if not file_path.exists():
                    logger.info(f"Missing {filename}, downloading from R2...")
                    path = f'instacart/{filename}'
                    try:
                        content = asyncio.run(download_text_file_from_r2(
                            bucket=bucket,
                            access_key_id=access_key_id,
                            secret_access_key=secret_access_key,
                            endpoint_url=endpoint_url,
                            path=path
                        ))
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Downloaded {filename}")
                    except Exception as e:
                        logger.error(f"Failed to download {filename}: {e}")
                        all_present = False
                else:
                    logger.info(f"{filename} already exists")
            
            if all_present:
                logger.info("All required files are present.")
            else:
                logger.warning("Some files could not be downloaded. Check R2 config or network.")
            
            return all_present
        except Exception as e:
            logger.error(f"Error in data initialization: {e}")
            return False


    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the Bitrecs prompt evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0
        eval_success = False
        
        popularity_sample_size = 100
        pop_metrics = self.evaluate_recommender(
            test_data=self.test,
            recommender_func=self.popularity_recommender,
            train_data=self.top_popular,
            k=self.num_recs,
            sample_size=popularity_sample_size,
            top_products=None
        )
        print(f"\nPopularity Baseline from {popularity_sample_size} samples:")
        logger.info("Popularity Baseline Metrics:")
        
        self.baseline_precision = pop_metrics.get(f'Precision@{self.num_recs}', 0.0)
        self.baseline_recall = pop_metrics.get(f'Recall@{self.num_recs}', 0.0)
        self.baseline_ndcg = pop_metrics.get(f'NDCG@{self.num_recs}', 0.0)
        self.baseline_hitrate = pop_metrics.get(f'HitRate@{self.num_recs}', 0.0)

        print("\nPopularity Recommender:")
        for k, v in pop_metrics.items():
            print(f"  {k:20} {v}")
            #logger.info(f"  {k:20} {v}")
        
        logger.info("Done evaluating Popularity baseline.")        
        logger.info("Starting LLM recommender evaluation...")

        llm_metrics = self.evaluate_recommender(
            test_data=self.test,
            recommender_func=self.llm_recommender,
            train_data=self.train,
            k=self.num_recs,
            sample_size=self.sample_size,
            top_products=self.top_products_llm  # Pass top products for LLM
        )
        print("\nLLM Recommender:")
        for k, v in llm_metrics.items():
            print(f"  {k:20} {v}")

        self.llm_precision = llm_metrics.get(f'Precision@{self.num_recs}', 0.0)
        self.llm_recall = llm_metrics.get(f'Recall@{self.num_recs}', 0.0)
        self.llm_ndcg = llm_metrics.get(f'NDCG@{self.num_recs}', 0.0)
        self.llm_hitrate = llm_metrics.get(f'HitRate@{self.num_recs}', 0.0)

        print(f"Done evaluating LLM recommender.\n")

        comparison = self.render_compare_table(markdown=False)
        print("Comparison Table:\n")
        print(comparison)

        duration = time.monotonic() - start_time
        final_score = self.llm_recall
        if final_score >= self.pass_threshold:
            print(f"\033[32mEval Passed! Final Score: {final_score:.4f} >= Threshold: {self.pass_threshold}\033[0m")
            eval_success = True
    
        result = EvalResult(
            eval_name=self.get_eval_name(),
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} of {self.sample_size} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )
        return result
        

    def render_compare_table(self, markdown: bool = True) -> str:
        data = {
            "Metric": [f"Precision@{self.num_recs}", f"Recall@{self.num_recs}", f"NDCG@{self.num_recs}", f"HitRate@{self.num_recs}"],
            "Popularity": [f"{self.baseline_precision:.4f}", f"{self.baseline_recall:.4f}", f"{self.baseline_ndcg:.4f}", f"{self.baseline_hitrate:.4f}"],
            "LLM Recommender": [f"{self.llm_precision:.4f}", f"{self.llm_recall:.4f}", f"{self.llm_ndcg:.4f}", f"{self.llm_hitrate:.4f}"],
        }
        df = pd.DataFrame(data)
        return df.to_markdown(index=False) if markdown else df.to_string(index=False)

    
    def precision_at_k(self, recommended, relevant, k):
        rec_k = recommended[:k]
        hits = len(set(rec_k) & set(relevant))
        return hits / k if k > 0 else 0.0


    def recall_at_k(self, recommended, relevant, k):
        rec_k = recommended[:k]
        hits = len(set(rec_k) & set(relevant))
        return hits / len(relevant) if relevant else 0.0
    
    
    def ndcg_at_k(self, recommended, relevant, k):        
        rec_k = recommended[:k]
        if not rec_k:
            return 0.0        
        # Binary relevance: 1 if in relevant, 0 otherwise
        relevance_scores = [1.0 if item in relevant else 0.0 for item in rec_k]        
        # Compute DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))        
        # Compute IDCG (ideal DCG: sorted descending)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))        
        return dcg / idcg if idcg > 0 else 0.0

    
    def hit_rate_at_k(self, recommended, relevant, k):
        return 1.0 if any(r in relevant for r in recommended[:k]) else 0.0


    def llm_recommender(self, user_history, num_recs=10, top_products=None) -> List[str]:
        if not user_history:
            return self.popularity_recommender(self.top_popular, num_recs)  # fallback
        
        if top_products is None:
            raise ValueError("top_products must be provided for llm_recommender")
        
        top_names = [self.product_map.get(pid, str(pid)) for pid in top_products]
        sorted_names = sorted(top_names)
        
        # Convert ids → names for better prompting
        history_names = [self.product_map.get(pid, str(pid)) for pid in user_history[:10]]

        prompt = f"""User previously bought: {', '.join(history_names)}.
        \n   
        Here is a subset of popular products: {', '.join(sorted_names)}.
        \n
        Suggest exactly {self.num_recs} next products from this subset list. Return ONLY a comma-separated list of product names, nothing else.
        """

        size = PromptFactory.get_token_count(prompt)
        logger.info(f"Prompt size (tokens): {size}")

        #server = LLM.OPEN_ROUTER
        #server = LLM.CHUTES
        #model = "google/gemini-2.5-flash-lite"
        #model = "moonshotai/Kimi-K2-Instruct-0905"

        model = self.miner_artifact.model
        server = LLM.try_parse(self.miner_artifact.provider)

        llm_output = LLMFactory.query_llm(server=server,
                                            model=model,
                                            system_prompt="You're a helpful shopping assistant.",
                                            user_prompt=prompt,
                                            temp=0.0)
        
        predictions = [name.strip() for name in llm_output.split(',') if name.strip()]
        recommended_ids = []
        name_to_id = {v.lower(): k for k, v in self.product_map.items()}
        for name in predictions:
            pid = name_to_id.get(name.lower())
            if pid and pid not in recommended_ids:
                recommended_ids.append(pid)
            if len(recommended_ids) >= num_recs:
                break
        return recommended_ids


    def popularity_recommender(self, top_items, num_recs=10):
        """Global most frequent items (precomputed)"""
        return top_items[:num_recs]
    
   
    def evaluate_recommender(self, test_data, recommender_func, train_data=None, k=10, sample_size=100, top_products=None):
        user_test = defaultdict(set)
        for _, row in test_data.iterrows():
            user_test[row['user_id']].add(row['product_id'])
        
        all_users = list(user_test.keys())
        if not all_users:
            return {"error": "No test users"}
        
        sampled_users = random.sample(all_users, min(sample_size, len(all_users))) if sample_size else all_users
        
        precs, recs, ndcgs, hits = [], [], [], []
        valid_count = 0
        
        for user_id in sampled_users:
            relevant = user_test[user_id]
            if not relevant:
                continue
            
            # Only compute history for LLM
            if recommender_func == self.llm_recommender:
                history_df = train_data[train_data['user_id'] == user_id] if train_data is not None else pd.DataFrame()
                user_history = history_df['product_id'].unique().tolist()
                if not user_history or len(user_history) < 1:                    
                    #print(f"Skipping user {user_id} with no history")
                    logger.info(f"Skipping user {user_id} with no history")
                    continue
            else:
                user_history = []  # Not needed for popularity        
            
            if recommender_func == self.llm_recommender:
                recommended = recommender_func(user_history, k, top_products)
            else:
                recommended = recommender_func(train_data, k)  # train_data is top_popular list
        
            precs.append(self.precision_at_k(recommended, relevant, k))
            recs.append(self.recall_at_k(recommended, relevant, k))
            ndcgs.append(self.ndcg_at_k(recommended, relevant, k))
            hits.append(self.hit_rate_at_k(recommended, relevant, k))
            
            valid_count += 1
        
        if valid_count == 0:
            return {"note": "No valid test cases in sample"}
        
        result = {
            f'Precision@{k}':  round(np.mean(precs), 4),
            f'Recall@{k}':     round(np.mean(recs),  4),
            f'NDCG@{k}':        round(np.mean(ndcgs), 4),
            f'HitRate@{k}':     round(np.mean(hits),  4),
            'Evaluated users':  valid_count,
            'Sampled from':     len(all_users),
        }
        #logger.info(f"Recommender evaluation: {result}")
        return result

    def convert_products(self) -> List[Product]:
        result = []
        for p in self.products.itertuples():
            product = Product(
                sku=str(p.product_id),  
                name=p.product_name,
                price="0"  # Price not available in dataset
            )
            result.append(product)
        sorted_result = sorted(result, key=lambda x: (x.name.lower(), x.sku.lower()))
        return sorted_result
        
