import os
import secrets
import time
import traceback
import sqlite3
import logging
import pandas as pd
from typing import List
from datetime import datetime, timezone
from commerce.product_factory import ProductFactory
from commerce.user_profile import UserProfile
from common import constants as CONST
from evals.eval_result import EvalResult
from evals.scoring.order_predict import OrderForecasting
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.eval_type import BitrecsEvaluationType
from models.miner_artifact import Artifact
from common.utils import rec_list_to_set
from evals.base_eval import BaseEval
from models.product import Product
from tests.order_predict import get_simple_sku_stats, load_products_from_db


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

"""
Evaluates prompt and model ability to predict product recommendations.

check: if recommended skus are in order history 
data: sample test db

"""

DB_PATH = os.path.join(CONST.ROOT_DIR, "data", "testdb", "store.sqlite")

MIN_ORDER_CLIP = 25.00

class BitrecsPredictEval(BaseEval):
   
    @property
    def sample_size(self) -> int:
        return 1
    
    def pass_threshold(self) -> float:
        return 1.0

    def __init__(self, run_id: str, miner_artifact: Artifact = None):      
        super().__init__(run_id, miner_artifact)

        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        predictor = OrderForecasting(db)
        self.predictor = predictor
     

    def eval_type(self) -> BitrecsEvaluationType:
        return BitrecsEvaluationType.BITRECS_PREDICT_DAILY

    def run(self, max_iterations=10) -> EvalResult:
        """
        Run the Bitrecs prompt evaluation.
        """
        start_time = time.monotonic()        
        count = 0
        success_count = 0
        exception_count = 0
        
        
        for idx in range(self.sample_size):
            try:               
                logger.info(f"\033[34mATTEMPT {idx+1}...\033[0m")
                st = time.monotonic()
                eval_result = self.evaluate_row()                
                et = time.monotonic()
                duration = et - st
                logger.info(f"\033[32m Row {idx} evaluation took {duration:.2f}s \033[0m")
                if eval_result:
                    logger.info(f"\033[32m Row {idx} SUCCESS \033[0m")
                    success_count += 1
                else:
                    logger.info(f"\033[31m Row {idx} FAILURE \033[0m")
                    
                logger.info(f"Current score: {success_count}/{idx + 1} = {success_count / (idx + 1):.2f}")
            except Exception as e:
                error_message = traceback.format_exc()                
                logger.error(f"Error evaluating row {idx}: {e} \n{error_message}")
                exception_count += 1
                continue
            finally:
                 count += 1

        
        end_time = time.monotonic()
        total_duration = end_time - start_time        
        final_score = success_count / count if count > 0 else 0.0                
        eval_success = False

        if final_score >= self.pass_threshold:
            eval_success = True
    
        result = EvalResult(           
            eval_name=self.get_eval_name(),  # Use base method
            created_at=datetime.now(timezone.utc).isoformat(),
            hot_key=self.miner_artifact.miner_hotkey,
            score=final_score,
            passed=eval_success,
            rows_evaluated=count,
            details=f"Evaluated {count} rows with {exception_count} exceptions (max_iterations {max_iterations}).",
            duration_seconds=total_duration,
            temperature=self.miner_artifact.sampling_params.temperature,
            model_name=self.miner_artifact.model,
            provider_name=self.miner_artifact.provider,
            run_id=self.run_id
        )        

        return result
    
    
    def load_products_from_db(self, sql_lite_path: str, truncate_names: bool = True) -> List[Product]:  
        products = []
        try:
            conn = sqlite3.connect(sql_lite_path)
            cursor = conn.cursor()
            sql = """SELECT 
                sku,
                CASE 
                    WHEN INSTR(name, '-') > 0 
                    THEN SUBSTR(name, 1, INSTR(name, '-') - 1)
                    ELSE name
                END AS name,
                price
            FROM music_products;"""
            if not truncate_names:
                sql = """SELECT sku, name, price FROM music_products"""
                print(" \033[1;31m  Warning Loading full product names, check token limits! \033[0m")
            else:
                print(" \033[1;33m  Product Names Truncated \033[0m")
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                product = Product(
                    sku=str(row[0]), 
                    name=str(row[1]),
                    price=str(row[2])
                )
                products.append(product)            
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
        finally:
            if conn:
                conn.close()    
        return ProductFactory.dedupe(products)
    
    
    def get_sample_user_profile(self, min_orders: int = 5) -> UserProfile:
        sql = f"""
            select group_id, count(1) as count from music_orders
            where status == 'complete' and total_paid > {MIN_ORDER_CLIP}
            group by group_id
            having count(1) > {min_orders}"""
        profiles = []
        profile_orders = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                raise ValueError("No user profiles found in the database.")        
            profiles = [{"group_id": row[0], "count": row[1]} for row in rows]
            print(f"Found {len(profiles)} distinct user profiles with more than {min_orders} orders.")
            if min_orders== 5:
                assert 245 == len(profiles), "Expected 245 distinct user profiles with 5 orders or more."
            r = secrets.choice(profiles)        
            sql = f"""select o.*, i.qty, i.sku, i.name, i.price from music_orders o 
                    left join music_order_items i on o.order_id = i.order_id
                    where group_id = '{r['group_id']}' and o.status = 'complete' and o.total_paid > 0"""
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                raise ValueError("No orders found for the selected user profile.")
            orders = []        
            for row in rows:
                order = {
                    "order_id": row[0],
                    "grand_total": str(row[1]),
                    "status": str(row[2]),
                    "subtotal": str(row[3]),
                    "subtotal_inc_tax": str(row[4]),
                    "subtotal_invoiced": str(row[5]),
                    "total_item_count": str(row[6]),
                    "total_paid": str(row[7]),
                    "total_qty_ordered": str(row[8]),
                    "updated_at": str(row[9]),
                    "group_id": str(row[10]),
                    "qty": str(row[11]),
                    "sku": str(row[12]),
                    "name": str(row[13]),
                    "price": str(row[14])
                }
                orders.append(order)
            print(f"Found {len(orders)} orders for user profile {r['group_id']}.")
            for order in orders:
                order_id = order["order_id"]
                if order_id not in profile_orders:
                    profile_orders[order_id] = {
                        "order_id": order_id,
                        "total": str(order["grand_total"]),
                        "status": str(order["status"]),
                        "created_at": str(order["updated_at"]),
                        "items": []
                    }
                profile_orders[order_id]["items"].append({
                    "sku": str(order["sku"]),
                    "name": str(order["name"]),
                    "price": str(order["price"]),
                    "quantity": str(order["qty"])
                })
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
        finally:
            if conn:
                conn.close()
        if not profiles:    
            raise ValueError("No user profiles found in the database.")
        
        print(f"Found {len(profiles)} distinct user profiles in the database.")
        user_profile : UserProfile = UserProfile(
            id=str(r['group_id']),
            created_at="2025-05-31T18:45:13Z",  
            site_config={"profile": "ecommerce_retail_store_manager"},
            cart=[],  
            orders=list(profile_orders.values())
        )
        print(f"Selected random profile: \033[1;32m  {user_profile} \033[0m")
        return user_profile 

    def load_catalog(self, truncate_names: bool = True) -> List[Product]: 
        catalog = self.load_products_from_db(DB_PATH, truncate_names=truncate_names)   
        return catalog  
    
    
    def get_simple_sku_stats(self, sku: str) -> dict:  
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        query = """
        SELECT 
            sku, 
            COUNT(DISTINCT order_id) as total_orders, 
            SUM(row_total) as total_revenue, 
            SUM(qty) as total_items_sold
        FROM music_order_items
        WHERE sku = ?
        GROUP BY sku
        """    
        cursor.execute(query, (sku,))
        row = cursor.fetchone()
        if not row:
            return {'sku': sku, 'total_orders': 0, 'total_revenue': 0.0, 'total_items_sold': 0}
            
        stats = {
            'sku': row['sku'],
            'total_orders': row['total_orders'],
            'total_revenue': round(row['total_revenue'], 2),
            'total_items_sold': row['total_items_sold']
        }        
        db.close()
        return stats
    
    def evaluate_row(self) -> bool:
        """
        Fetches a sample profile and order history and evaluates the LLM's ability to recommend products

        """
       
        duration = 0.0
        result = False
        num_recs = 5
        st = time.perf_counter()
        profile = self.get_sample_user_profile()    
        #$products = products_music()[:5000]
        products = self.load_catalog()
        #print(f"Loaded {len(products)} products from the database.")
        #print(f"User profile: {profile.id} with {len(profile.orders)} orders and {len(profile.cart)} items in cart.")
    
        #first_order = profile.orders[0]
        #first_sku = first_order['items'][0]['sku']
        #first_sku = "FASDSLUSGSBLK"
        profile.orders.sort(key=lambda o: o['created_at'])
        first_order = profile.orders[0]
        last_order = profile.orders[-1]
        #print(f"First order ID: {first_order['order_id']} at {first_order['created_at']}")
        #print(f"Last order ID: {last_order['order_id']} at {last_order['created_at']}")

        first_item = secrets.choice(first_order['items'])
        first_sku = first_item['sku']
        #first_sku = first_order['items'][0]['sku']    

        viewing_product = next((p for p in products if p.sku == first_sku), None)
        assert viewing_product is not None, f"Product with SKU {first_sku} not found in products list."
        
        query = viewing_product.sku
        logger.info(f"Viewing product: {viewing_product.name} \033[1;32m (SKU: {viewing_product.sku}) \033[0m")
        stats = self.get_simple_sku_stats(viewing_product.sku)
        print(f"SKU Stats: {stats}")
        
        prompt_factory = PromptFactory(miner_artifact=self.miner_artifact,                                       
                                        sku=query,
                                        products=products,
                                        num_recs=num_recs)
        system_prompt, user_prompt = prompt_factory.generate_prompt()
        # print("System Prompt:")
        # print(system_prompt)
        # print("User Prompt:")
        # print(user_prompt)    
        tc = PromptFactory.get_token_count(user_prompt)
        logger.info(f"Token count: {tc}")
        
        #model = "google/gemini-2.5-flash-lite-preview-09-2025"
        model = "google/gemini-3-flash-preview"

        #server = LLM.OLLAMA_LOCAL
        server = LLM.OPEN_ROUTER

        logger.info(f"Using model:  \033[1;32m {model} \033[0m")
        st = time.perf_counter()
        llm_response = LLMFactory.query_llm(server=server,
                                    model=model, 
                                    system_prompt=system_prompt, 
                                    temp=0.0, 
                                    user_prompt=user_prompt)
        et = time.perf_counter()
        logger.info(f"LLM response time: {et - st:0.2f} seconds")    
        #print(llm_response)
        recommended_skus = PromptFactory.tryparse_llm(llm_response)   
        logger.info(f"parsed {len(recommended_skus)} records")
        #print(recommended_skus)    
        #assert len(recommended_skus) == num_recs    
        for rec in recommended_skus:
            sku = rec['sku']
            assert sku not in [product['sku'] for product in profile.cart], f"SKU {sku} should not be in cart"
            this_product = next((p for p in products if p.sku == sku), None)
            if this_product:
                #assert this_product is not None, f"Product with SKU {first_sku} not found in products list."
                #assert sku != viewing_product.sku, f"Recommended SKU {sku} should not be the same as viewing SKU {viewing_product.sku}"
                logger.info(f"Recommended: {this_product.name} \033[1;32m (SKU: {rec['sku']}) \033[0m - Reason: {rec['reason']}")        

        # Check if any recommended SKU is in the user's order history
        ordered_skus = set()
        for order in profile.orders:
            for item in order['items']:
                ordered_skus.add(item['sku'])
        recommended_sku_set = rec_list_to_set(recommended_skus)
        logger.info(f"User has ordered {len(ordered_skus)} distinct SKUs in their order history.")
        logger.info(f"Recommended SKUs: {recommended_sku_set}")
        intersection = ordered_skus.intersection(recommended_sku_set)
        logger.info(f"Intersection of recommended SKUs and ordered SKUs: {intersection}")
        if intersection:
            logger.info(f"\033[32m SUCCESS: At least one recommended SKU is in the user's order history. \033[0m")
            result = True
        else:
            logger.info(f"\033[31m FAILURE: No recommended SKUs are in the user's order history. \033[0m")
            result = False

        duration = et - st
        self.log_miner_response(
            run_id=self.run_id,
            query=query,
            num_recs=num_recs,
            recommended_skus=recommended_skus,
            duration=duration
        )

        return result    
    
