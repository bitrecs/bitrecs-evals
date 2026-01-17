
import ast
import json
from typing import List
from dataclasses import dataclass


@dataclass
class ReasonedProduct:
    sku : str
    name : str
    price : str
    reason: str 
    
    @staticmethod
    def from_json(json_str: str) -> List['ReasonedProduct']:
        reasoned_products = []
        try:            
            items = json.loads(json_str)
        except json.JSONDecodeError:
            try:                
                items = ast.literal_eval(json_str)
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse data: {e}")
                return reasoned_products    
        
        for item in items:
            if isinstance(item, dict):                
                data = item
            elif isinstance(item, str):                
                try:
                    data = json.loads(item)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(item)
                    except:
                        print(f"Failed to parse item: {item}")
                        continue
            else:
                print(f"Unexpected item type: {type(item)}")
                continue            
            
            if isinstance(data, dict):
                rp = ReasonedProduct(
                    sku=data.get("sku", ""),
                    name=data.get("name", ""),
                    price=data.get("price", ""),
                    reason=data.get("reason", "")
                )
                reasoned_products.append(rp)
    
        return reasoned_products
    