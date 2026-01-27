import time
import json
import httpx
import logging
from llm.llm_provider import LLM
from common import constants as CONST
from llm.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)

class OpenRouter:    
    def __init__(self, 
                 key,
                 model="google/gemini-flash-1.5-8b", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0
        ):

        self.OPENROUTER_API_KEY = key
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model       
        self.system_prompt = system_prompt
        self.temp = temp        
        self.provider = LLM.OPEN_ROUTER.name
        
        # Add pricing map (example rates in USD per 1K tokens; update with real OpenRouter pricing)
        self.pricing = {
            "google/gemini-2.5-flash-lite-preview-09-2025": {"input": 0.10, "output": 0.40},
            "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
            "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.30},
            "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.30},

            "x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
            "x-ai/x-ai/grok-4-fast": {"input": 0.20, "output": 0.50},

            "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
            "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
            "openai/gpt-4.1-nano": {"input": 0.10, "output": 0.40},

            "qwen/qwen/qwen3-embedding-8b": {"input": 0.01, "output": 0.00},
            "qwen/qwen3-next-80b-a3b-instruct": {"input": 0.09, "output": 1.10},
            "qwen/qwen3-235b-a22b-2507": {"input": 0.071, "output": 0.463},

            "amazon/nova-2-lite-v1": {"input": 0.30, "output": 2.50},           
            
        }

    def call_open_router(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError("Prompt too short")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bitrecs.ai",
            "X-Title": "bitrecs"
        }
        reasoning = {
            "enabled": False,
            "exclude": True,
            "effort": "minimal"
        }
        # Handle specific models that require different reasoning settings
        if "gpt-5" in self.model.lower():
            reasoning = {
                "exclude": True,
                "effort": "minimal"
            }

        data_payload = {
            "model": self.model,
            "messages": [
                #{"role": "system", "content": "/no_think"},
                {
                    "role": "user", 
                    "content": prompt
                }],
            "reasoning": reasoning,
            "stream": False,
            "temperature": self.temp,
            "max_tokens": 2048
        }
        
        timeout = (3, 30)  # Reduced connect/read timeouts
        max_retries = 1  # Reduced to 1 retry (2 total attempts)
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    if CONST.LOG_LEVEL <= logging.DEBUG:
                        start_time = time.perf_counter()
                        content = data_payload["messages"][0]["content"]
                        token_count = PromptFactory.get_token_count(content)
                        logger.debug(f"OPEN_ROUTER request token count: {token_count} tokens")
                    response = client.post(
                        url,
                        headers=headers,
                        json=data_payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    if CONST.LOG_LEVEL <= logging.DEBUG:
                        end_time = time.perf_counter()
                        duration = end_time - start_time
                        logger.debug(f"OPEN_ROUTER request completed in {duration:.2f}s")
                    
                    # Extract and log additional info
                    usage = data.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    finish_reason = data.get('choices', [{}])[0].get('finish_reason', 'unknown')
                    actual_model = data.get('model', self.model)
                    
                    # Calculate cost
                    model_pricing = self.pricing.get(actual_model, {"input": 0, "output": 0})
                    cost = (prompt_tokens / 1000000 * model_pricing["input"]) + (completion_tokens / 1000000 * model_pricing["output"])
                    
                    logger.info(f"Request ID: {data.get('id')}, Model: {actual_model}, Tokens: {total_tokens} (Prompt: {prompt_tokens}, Completion: {completion_tokens}), Cost: ${cost:.6f}, Finish Reason: {finish_reason}")
                    
                    # Return full data for caller to use
                    #return data
                    
                    return data['choices'][0]['message']['content']
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    wait_time = 1 * (attempt + 1)  # Linear backoff: 1s, 2s
                    logger.warning(f"429 received, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenRouter request failed: {e}") from e
            except httpx.ConnectTimeout:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Connect timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"OpenRouter connect timed out after {timeout[0]}s")
            except httpx.ReadTimeout:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Read timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"OpenRouter read timed out after {timeout[1]}s")
            except httpx.RequestError as e:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Request error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenRouter request failed: {e}") from e