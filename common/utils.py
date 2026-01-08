import json
from typing import Set


def rec_list_to_set(recs: list) -> Set[str]:   
    sku_set = set()
    for item in recs:
        if isinstance(item, dict) and 'sku' in item:
            sku_set.add(item['sku'])
        elif isinstance(item, str):       
            product = json.loads(item)
            if isinstance(product, dict) and 'sku' in product:
                sku_set.add(product['sku'])
        else:
            print(f"Invalid item type in results: {item}")
    return sku_set


def normalize_model_name(model_name: str) -> str:
    normalized_model = model_name.split('/')[-1] if '/' in model_name else model_name
    normalized_model = normalized_model.split(':')[0] if ':' in normalized_model else normalized_model
    return normalized_model
