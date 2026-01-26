from dotenv import load_dotenv
load_dotenv()
import random
import pandas as pd
import numpy as np
from llm.prompt_factory import PromptFactory
from sklearn.metrics import ndcg_score
from collections import defaultdict
from llm.factory import LLMFactory
from llm.llm_provider import LLM

# ────────────────────────────────────────────────
#   Instacart Recommendation Evaluation
# ────────────────────────────────────────────────

products = pd.read_csv('data/instacart/products.csv')[['product_id', 'product_name']]
product_map = dict(zip(products['product_id'], products['product_name']))


def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    """Safe NDCG@K using binary relevance"""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    
    # Relevance scores for recommended items
    y_true = np.array([1 if item in relevant else 0 for item in rec_k])
    y_score = y_true  # since we don't have ranked scores, use binary
    
    # sklearn ndcg expects 2D arrays
    y_true = y_true.reshape(1, -1)
    y_score = y_score.reshape(1, -1)
    
    try:
        return ndcg_score(y_true, y_score, k=k)
    except:
        # Fallback pure-python version if sklearn complains
        dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(y_true[0]))
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted(y_true[0], reverse=True)))
        return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended, relevant, k):
    return 1.0 if any(r in relevant for r in recommended[:k]) else 0.0


# ────────────────────────────────────────────────
#   RECOMMENDERS
# ────────────────────────────────────────────────

def llm_recommender(user_history, num_recs=10, top_products=None):
    if not user_history:
        return popularity_recommender(top_popular, num_recs)  # fallback
    
    if top_products is None:
        # Fallback if not passed (but should be)
        top_products = []  # Or compute here, but inefficient
    
    top_names = [product_map.get(pid, str(pid)) for pid in top_products]
    sorted_names = sorted(top_names)
    
    # Convert ids → names for better prompting
    history_names = [product_map.get(pid, str(pid)) for pid in user_history[:30]]
    prompt = f"""User previously bought: {', '.join(history_names)}.
    \n   
    Here is a subset of popular products: {', '.join(sorted_names)}.
    \n
    Suggest exactly 10 next products from this list. Return ONLY a comma-separated list of product names, nothing else.
    """

    size = PromptFactory.get_token_count(prompt)
    print(f"Prompt size (tokens): {size}")

    server = LLM.OPEN_ROUTER
    model = "google/gemini-2.5-flash-lite"
    llm_output = LLMFactory.query_llm(server=server,
                                        model=model,
                                        system_prompt="You're a helpful grocery shopping assistant.",
                                        user_prompt=prompt,
                                        temp=0.0)
    
    predictions = [name.strip() for name in llm_output.split(',') if name.strip()]
    recommended_ids = []
    name_to_id = {v.lower(): k for k, v in product_map.items()}
    for name in predictions:
        pid = name_to_id.get(name.lower())
        if pid and pid not in recommended_ids:
            recommended_ids.append(pid)
        if len(recommended_ids) >= num_recs:
            break
    return recommended_ids


def popularity_recommender(top_items, num_recs=10):
    """Global most frequent items (precomputed)"""
    return top_items[:num_recs]


# ────────────────────────────────────────────────
#   EVALUATION
# ────────────────────────────────────────────────

def evaluate_implicit(test_data, recommender_func, train_data=None, k=10, sample_size=200, top_products=None):
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
        if recommender_func == llm_recommender:
            history_df = train_data[train_data['user_id'] == user_id] if train_data is not None else pd.DataFrame()
            user_history = history_df['product_id'].unique().tolist()
            if not user_history or len(user_history) < 1:
                print(f"Skipping user {user_id} with no history")
                continue
        else:
            user_history = []  # Not needed for popularity
    
        # Update call: remove user_id for llm_recommender
        if recommender_func == llm_recommender:
            recommended = recommender_func(user_history, k, top_products)
        else:
            recommended = recommender_func(train_data, k)  # train_data is top_popular list
    
        precs.append(precision_at_k(recommended, relevant, k))
        recs.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        hits.append(hit_rate_at_k(recommended, relevant, k))
        
        valid_count += 1
    
    if valid_count == 0:
        return {"note": "No valid test cases in sample"}
    
    return {
        f'Precision@{k}':  round(np.mean(precs), 4),
        f'Recall@{k}':     round(np.mean(recs),  4),
        f'NDCG@{k}':        round(np.mean(ndcgs), 4),
        f'HitRate@{k}':     round(np.mean(hits),  4),
        'Evaluated users':  valid_count,
        'Sampled from':     len(all_users),
    }


# ────────────────────────────────────────────────
#   MAIN
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust paths to where you actually extracted the Instacart data
    ORDERS_PATH = 'data/instacart/orders.csv'
    ORDER_PRODUCTS_PRIOR_PATH = 'data/instacart/order_products__prior.csv'
    ORDER_PRODUCTS_TRAIN_PATH = 'data/instacart/order_products__train.csv'  # For test (last orders)

    print("Loading data...")
    orders = pd.read_csv(ORDERS_PATH)

    # Train: all prior purchases
    order_products_prior = pd.read_csv(ORDER_PRODUCTS_PRIOR_PATH)
    train = order_products_prior.merge(orders[['order_id', 'user_id']], on='order_id')
    train = train[['user_id', 'product_id', 'order_id']]

    # Test: last purchases (from train.csv, which has the last order per user)
    order_products_train = pd.read_csv(ORDER_PRODUCTS_TRAIN_PATH)
    test = order_products_train.merge(orders[['order_id', 'user_id']], on='order_id')
    test = test[['user_id', 'product_id', 'order_id']]

    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"Users in test: {test['user_id'].nunique():,}")


    # ── Evaluation ─────────────────────────────────────────────────────────────
    SAMPLE_SIZE_LLM = 10        # Small for LLM due to cost
    SAMPLE_SIZE_POP = 100        # Larger for Popularity (fast, no cost)

    print(f"\nEvaluating LLM on {SAMPLE_SIZE_LLM} users, Popularity on {SAMPLE_SIZE_POP} users ...")

    # Compute after loading train
    top_popular = train['product_id'].value_counts().index[:10].tolist()
    top_products_llm = train['product_id'].value_counts().head(5000).index.tolist()

    llm_metrics = evaluate_implicit(
        test_data=test,
        recommender_func=llm_recommender,
        train_data=train,
        k=10,
        sample_size=SAMPLE_SIZE_LLM,
        top_products=top_products_llm  # Pass top products for LLM
    )
    print("\nLLM Recommender:")
    for k, v in llm_metrics.items():
        print(f"  {k:20} {v}")

    print(f"Done evaluating LLM recommender.\n")


    pop_metrics = evaluate_implicit(
        test_data=test,
        recommender_func=popularity_recommender,
        train_data=top_popular,  # Pass the list instead of full train
        k=10,
        sample_size=SAMPLE_SIZE_POP,
        top_products=None  # Not needed for pop
    )
    print("\nPopularity Baseline:")
    for k, v in pop_metrics.items():
        print(f"  {k:20} {v}")

    print(f"Done evaluating Popularity baseline.\n")    




