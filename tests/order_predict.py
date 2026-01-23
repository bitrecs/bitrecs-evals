import os
import sys
project_root = os.getcwd() 
sys.path.insert(0, project_root)
import sqlite3
import time
import secrets
from typing import List
from commerce.product_factory import ProductFactory
from commerce.user_profile import UserProfile
from llm.factory import LLMFactory
from llm.llm_provider import LLM
from llm.prompt_factory import PromptFactory
from models.product import Product
from models.miner_artifact import Artifact
from common import constants as CONST
from evals.scoring.order_predict import OrderForecasting
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.path.join(CONST.ROOT_DIR, "data", "testdb", "store.sqlite")

MIN_ORDER_CLIP = 25.00



def products_music(truncate_names: bool = True) -> List[Product]: 
    catalog = load_products_from_db(DB_PATH, truncate_names=truncate_names)   
    return catalog


def random_product() -> Product:
    return secrets.choice(products_music())


def load_products_from_db(sql_lite_path: str, truncate_names: bool = True) -> List[Product]:  
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


def get_sample_user_profile(min_orders: int = 5) -> UserProfile:
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


def get_simple_sku_stats(sku: str) -> dict:  
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




def analyze_recommendations_with_sequential(sku: str, rec_skus: List[str] = None):
    """Analyze recommendations including sequential purchase patterns"""
    if not sku:
        raise ValueError("SKU must be provided for analysis")
    if rec_skus is None:
        raise ValueError("rec_skus must be provided for analysis")    
    
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    predictor = OrderForecasting(db)
    print(f"Analyzing SKU: {sku}")
    print(f"Recommendation SKUs: {rec_skus}")
    
    # Get sequential patterns
    sequential = predictor.find_sequential_orders(sku, rec_skus)
    print("\nSequential Patterns Found:")
    print(f"  Customers: {sequential['summary_stats']['total_customers']}")
    print(f"  Sequential orders: {sequential['summary_stats']['total_sequential_orders']}")
    
    if sequential['summary_stats']['total_customers'] > 0:
        print(f"  Average days between purchases: {sequential['summary_stats']['avg_days_between_purchases']}")
        print(f"  Total revenue: ${sequential['summary_stats']['total_rec_revenue']}")
        
        print("\nRecommendation Performance:")
        for rec_sku in sequential['summary_stats']['rec_sku_frequency'].keys():
            frequency = sequential['summary_stats']['rec_sku_frequency'][rec_sku]
            unique_customers = sequential['summary_stats']['rec_sku_unique_customers'][rec_sku]
            conversion = sequential['summary_stats']['conversion_rate_by_sku'][rec_sku]
            avg_purchases = sequential['summary_stats']['avg_purchases_per_customer_by_sku'][rec_sku]
            
            print(f"  {rec_sku}: {frequency} total purchases by {unique_customers} customers "
                  f"({conversion}% conversion, {avg_purchases} purchases/customer)")
    
    # Get same-order patterns (fix the variable names)
    same_order_patterns = predictor.find_similar_orders(sku, rec_skus)
    same_order_count = same_order_patterns['total_orders']
    sequential_count = sequential['summary_stats']['total_sequential_orders']
    
    print("\nComprehensive Analysis:")
    print(f"  Same-order patterns: {same_order_count}")
    print(f"  Sequential patterns: {sequential_count}")
    
    # Show co-occurrence stats if any
    if same_order_count > 0:
        print("\nSame-Order Co-occurrence Stats:")
        for rec_sku, count in same_order_patterns['co_occurrence_stats'].items():
            print(f"  {rec_sku}: appears with {sku} in {count} orders")
    
    db.close()

def analyze_recommendations(sku: str, rec_skus: List[str] = None):
    if not sku:
        raise ValueError("SKU must be provided for analysis")
    if rec_skus is None:
        raise ValueError("rec_skus must be provided for analysis")    
   
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    predictor = OrderForecasting(db) 
    similar_orders = predictor.find_similar_orders(sku, rec_skus)
    print(f"Found {similar_orders['total_orders']} orders with co-occurrences")
    for order in similar_orders['orders']:
        print(f"Order ID: {order['order_id']}, Total: {order['grand_total']}, "
              f"Status: {order['status']}, Rec SKUs: {order['matching_rec_skus']}")
    print("Co-occurrence stats:", similar_orders['co_occurrence_stats'])    
    
    strength_analysis = predictor.get_recommendation_strength(sku, rec_skus)
    print(f"\nBase SKU appears in {strength_analysis['base_sku_orders']} orders")
    print("Recommendation Strength Analysis:")
    for rec in strength_analysis['recommendations']:
        print(f"  {rec['sku']}: {rec['strength_percentage']}% strength, "
              f"${rec['total_revenue']} revenue, {rec['total_items_sold']} items sold")
    
    # Get customer patterns
    patterns = predictor.get_customer_purchase_patterns(sku, rec_skus)
    print(f"\nFound {patterns['total_customers']} customers with purchase patterns")
    
    db.close()



def dump_schema_simple(db_path, output_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        statements = [row[0] + ';\n' for row in cursor.fetchall()]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(statements)


def test_generate_schema_file():
    schema_file = os.path.join(CONST.ROOT_DIR, "data", "testdb", "music_store_schema.sql")
    dump_schema_simple(DB_PATH, schema_file)
    assert os.path.exists(schema_file), "Schema file was not created."


def test_load_products_from_db():
    products = load_products_from_db(DB_PATH)
    print(f"Loaded {len(products)} products from the database.")
    assert len(products) == 22664, "Expected 22680 products to be loaded from the database."
    for product in products:
        assert isinstance(product, Product), "Loaded item is not a Product instance."
        assert product.sku is not None, "Product SKU should not be None."
        assert product.name is not None, "Product name should not be None."
        assert product.price is not None, "Product price should not be None."


def test_sample_user_profile():
    profile = get_sample_user_profile()
    assert isinstance(profile, UserProfile), "Expected UserProfile instance."
    assert profile.id is not None, "UserProfile ID should not be None."
    assert profile.created_at is not None, "UserProfile created_at should not be None."  
    assert profile.site_config is not None, "UserProfile site_config should not be None."    
    assert len(profile.orders) > 0, "UserProfile should have at least one order."    
    

def test_sample_profile_get_similar_orders():
    num_recs = 5
    profile = get_sample_user_profile()    
    #$products = products_music()[:5000]
    products = products_music()
    print(f"Loaded {len(products)} products from the database.")
    print(f"User profile: {profile.id} with {len(profile.orders)} orders and {len(profile.cart)} items in cart.")
  
    #first_order = profile.orders[0]
    #first_sku = first_order['items'][0]['sku']
    #first_sku = "FASDSLUSGSBLK"
    profile.orders.sort(key=lambda o: o['created_at'])
    first_order = profile.orders[0]
    last_order = profile.orders[-1]
    print(f"First order ID: {first_order['order_id']} at {first_order['created_at']}")
    print(f"Last order ID: {last_order['order_id']} at {last_order['created_at']}")

    first_item = secrets.choice(first_order['items'])
    first_sku = first_item['sku']
    #first_sku = first_order['items'][0]['sku']    

    viewing_product = next((p for p in products if p.sku == first_sku), None)
    assert viewing_product is not None, f"Product with SKU {first_sku} not found in products list."
       
    query = viewing_product.sku
    print(f"Viewing product: {viewing_product.name} \033[1;32m (SKU: {viewing_product.sku}) \033[0m")
    stats = get_simple_sku_stats(viewing_product.sku)
    print(f"SKU Stats: {stats}")

    miner_yaml = os.path.join(CONST.ROOT_DIR, "input", "miner_input.yaml")    
    artifact = Artifact.from_yaml(miner_yaml)
    prompt_factory = PromptFactory(miner_artifact=artifact,
                                   sku=query,
                                    products=products,
                                    num_recs=num_recs)
    system_prompt, user_prompt = prompt_factory.generate_prompt()
    # print("System Prompt:")
    # print(system_prompt)
    # print("User Prompt:")
    # print(user_prompt)
 
    tc = PromptFactory.get_token_count(user_prompt)
    print(f"Token count: {tc}")
    
    #model = "google/gemini-2.5-flash-lite-preview-09-2025"
    model = "google/gemini-3-flash-preview"

    #server = LLM.OLLAMA_LOCAL
    server = LLM.OPEN_ROUTER

    print(f"Using model:  \033[1;32m {model} \033[0m")
    st = time.perf_counter()
    llm_response = LLMFactory.query_llm(server=server,
                                 model=model, 
                                 system_prompt=system_prompt, 
                                 temp=0.0, 
                                 user_prompt=user_prompt)
    et = time.perf_counter()
    print(f"LLM response time: {et - st:0.2f} seconds")    
    #print(llm_response)
    parsed_recs = PromptFactory.tryparse_llm(llm_response)   
    print(f"parsed {len(parsed_recs)} records")
    print(parsed_recs)    
    assert len(parsed_recs) == num_recs    
    for rec in parsed_recs:
        sku = rec['sku']
        assert sku not in [product['sku'] for product in profile.cart], f"SKU {sku} should not be in cart"
        this_product = next((p for p in products if p.sku == sku), None)
        if this_product:
            #assert this_product is not None, f"Product with SKU {first_sku} not found in products list."
            #assert sku != viewing_product.sku, f"Recommended SKU {sku} should not be the same as viewing SKU {viewing_product.sku}"
            print(f"Recommended: {this_product.name} \033[1;32m (SKU: {rec['sku']}) \033[0m - Reason: {rec['reason']}")
    
    analyze_recommendations_with_sequential(viewing_product.sku, [rec['sku'] for rec in parsed_recs])
    analyze_recommendations(viewing_product.sku, [rec['sku'] for rec in parsed_recs])


if __name__ == "__main__":
    #test_load_products_from_db()
    #test_sample_user_profile()
    
    test_sample_profile_get_similar_orders()
    

   