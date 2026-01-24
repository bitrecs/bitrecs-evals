import re
import json
import tiktoken
import logging
from jinja2 import Template
from common import constants as CONST
from functools import lru_cache
from typing import List, Tuple
from datetime import datetime, timezone
from models.miner_artifact import Artifact
from models.product import Product
from pydantic import TypeAdapter

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

class PromptFactory:    

    def __init__(self, 
                 miner_artifact: Artifact,
                 sku: str, 
                 products: List["Product"], 
                 num_recs: int = 5,
                 persona: str = "Expert Ecommerce Product Recommender",
                 debug: bool = False) -> None:
     
        self.miner_artifact = miner_artifact
        self.sku = sku
        self.context = TypeAdapter(List[Product]).dump_json(products, exclude_none=True).decode('utf-8')
        self.num_recs = num_recs
        self.debug = debug
        self.catalog = []
        self.cart = []
        self.cart_json = "[]"
        self.orders = []
        self.order_json = "[]"
        self.persona = persona
        self.sku_info = "N/A"

        query_product = next((p for p in products if p.sku == sku), None)
        if query_product:
            self.sku_info = f"{query_product.name}"
        else:
            self.sku_info = "N/A"      
   
    
    def generate_prompt(self) -> Tuple[str, str]:
        """
        Generate full prompt using Jinja2 templates from miner artifact.
        """
        # Prepare template variables
        template_vars = {            
            "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sku": self.sku,
            "product_catalog": self.context,
            "num_recs": self.num_recs,
            "sku_info": self.sku_info,
            "cart_json": self.cart_json,
            "order_json": self.order_json,
            "persona": self.persona
        }

        # Render system prompt
        system_template = Template(self.miner_artifact.system_prompt_template)
        system_prompt = system_template.render(**template_vars)

        # Render user prompt
        user_template = Template(self.miner_artifact.user_prompt_template)
        user_prompt = user_template.render(**template_vars)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        if self.debug:
            logger.debug(f"Generated prompt:\n{full_prompt}")
        
        return system_prompt, user_prompt 
    

    def generate_prompt_compressed(self) -> Tuple[str, str]:
        """
        Generate full prompt using Jinja2 templates from miner artifact.
        Add compression for the context
        
        """
        # Prepare template variables
        template_vars = {            
            "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sku": self.sku,
            "product_catalog": self.context,
            "num_recs": self.num_recs,
            "sku_info": self.sku_info,
            "cart_json": self.cart_json,
            "order_json": self.order_json,
            "persona": self.persona
        }

        # Render system prompt
        system_template = Template(self.miner_artifact.system_prompt_template)
        system_prompt = system_template.render(**template_vars)

        # Render user prompt
        user_template = Template(self.miner_artifact.user_prompt_template)
        user_prompt = user_template.render(**template_vars)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        if self.debug:
            logger.debug(f"Generated prompt:\n{full_prompt}")
        
        return system_prompt, user_prompt 
    
    
    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = PromptFactory._get_cached_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_encoding(encoding_name: str):
        return tiktoken.get_encoding(encoding_name)
    
    
    @staticmethod
    def get_word_count(prompt: str) -> int:
        return len(prompt.split())
    

    @staticmethod
    def tryparse_llm(input_str: str) -> list:
        """
        Take raw LLM output and parse to an array 

        """
        try:
            if not input_str:
                logging.error("Empty input string tryparse_llm")   
                return []
            input_str = input_str.replace("```json", "").replace("```", "").strip()
            pattern = r'\[.*?\]'
            regex = re.compile(pattern, re.DOTALL)
            match = regex.findall(input_str)        
            for array in match:
                try:
                    llm_result = array.strip()
                    return json.loads(llm_result)
                except json.JSONDecodeError:                    
                    logging.error(f"Invalid JSON in prompt factory: {array}")
            return []
        except Exception as e:
            logging.error(str(e))
            return []


    @staticmethod
    def extract_skus_from_response(response_json: dict) -> List[str]:
        """
        Extracts a list of SKUs from an OpenAI-compatible chat completion response.
        Handles JSON arrays wrapped in ```json ... ```, plain text, or malformed responses.
        
        Args:
            response_json (dict): The full OpenAI response dict.
        
        Returns:
            List[str]: A list of SKUs extracted from the response. 
        """
        try:            
            if not response_json.get('choices') or len(response_json['choices']) == 0:
                return []
            
            content = response_json['choices'][0].get('message', {}).get('content', '')
            if not content:
                return []
            
            items = PromptFactory.tryparse_llm(content)
            if not isinstance(items, list):
                return []
                        
            skus = []
            for item in items:
                if isinstance(item, dict):
                    sku = item.get('sku')
                    if sku and isinstance(sku, str) and sku.strip():
                        skus.append(sku.strip())
                    else:
                        logging.warning(f"Invalid or missing 'sku' in item: {item}")
                        print(f"Invalid or missing 'sku' in item: {item}")
                        pass
                else:
                    logging.warning(f"Item is not a dict: {item}")
                    print(f"Item is not a dict: {item}")
            
            return skus
        except Exception as e:
            logging.error(f"Error extracting SKUs from response: {str(e)}")
            return []