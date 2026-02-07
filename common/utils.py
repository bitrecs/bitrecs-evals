from datetime import datetime, timezone
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


def time_ago(dt: datetime) -> str:
    """
    Convert a datetime object to a human-friendly 'time ago' string.
    If dt has no timezone, assume UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - dt

    seconds = diff.total_seconds()
    minutes = int(seconds // 60)
    hours = int(seconds // 3600)
    days = int(seconds // 86400)
    weeks = int(seconds // 604800)

    if seconds < 60:
        return "just now"
    elif minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif weeks < 5:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    else:
        return dt.strftime("%Y-%m-%d")
