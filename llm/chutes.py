import time
import httpx
import logging
from llm.prompt_factory import PromptFactory
from .llm_provider import LLM
from common import constants as CONST


logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

class Chutes:
    def __init__(self, 
                 key, 
                 model="deepseek-ai/DeepSeek-V3", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0):
        
        self.CHUTES_API_KEY = key
        if not self.CHUTES_API_KEY:
            raise ValueError("CHUTES_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp      
        self.provider = LLM.CHUTES.name

        self.pricing = {
            
            "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE": {"input": 0.08, "output": 0.55},
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE": {"input": 0.22, "output": 0.95},
            "Qwen/Qwen3-Next-80B-A3B-Instruct": {"input": 0.10, "output": 0.80},

            "moonshotai/Kimi-K2-Instruct-0905": {"input": 0.39, "output": 1.90},
            "moonshotai/Kimi-K2-Instruct-0905": {"input": 0.39, "output": 1.90},

            "zai-org/GLM-4.7-FP8": {"input": 0.30, "output": 1.20},
            "zai-org/GLM-4.7-Flash": {"input": 0.06, "output": 0.35},         
            
        }
                


    def call_chutes(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()
        url = "https://llm.chutes.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.CHUTES_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": 2048,
            "temperature": self.temp
        }      
        timeout = (3, 60)
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    if CONST.LOG_LEVEL <= logging.DEBUG:
                        start_time = time.perf_counter()
                        content = data["messages"][0]["content"]
                        token_count = PromptFactory.get_token_count(content)
                        logger.debug(f"CHUTES request token count: {token_count} tokens")
                    response = client.post(
                        url,
                        headers=headers,
                        json=data
                    )
                    response.raise_for_status()
                    data = response.json()
                    if CONST.LOG_LEVEL <= logging.DEBUG:
                        end_time = time.perf_counter()
                        duration = end_time - start_time
                        logger.debug(f"CHUTES request completed in {duration:.2f}s")
                    #print(data)

                     # Extract and log additional info
                    usage = data.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    finish_reason = data.get('choices', [{}])[0].get('finish_reason', 'unknown')
                    actual_model = data.get('model', self.model)
                    
                    # Calculate cost
                    model_pricing = self.pricing.get(actual_model, {"input": 0, "output": 0})
                    #cost = (prompt_tokens / 1000 * model_pricing["input"]) + (completion_tokens / 1000 * model_pricing["output"])
                    cost = (prompt_tokens / 1000000 * model_pricing["input"]) + (completion_tokens / 1000000 * model_pricing["output"])
                    
                    logger.info(f"Request ID: {data.get('id')}, Model: {actual_model}, Tokens: {total_tokens} (Prompt: {prompt_tokens}, Completion: {completion_tokens}), Cost: ${cost:.6f}, Finish Reason: {finish_reason}")
                    
                    return data['choices'][0]['message']['content']
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    wait_time = 1 * (attempt + 1)  # Linear backoff: 1s, 2s
                    logger.warning(f"429 received, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"CHUTES request failed: {e}") from e
            except httpx.ConnectTimeout:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Connect timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"CHUTES connect timed out after {timeout[0]}s")
            except httpx.ReadTimeout:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Read timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"CHUTES read timed out after {timeout[1]}s")
            except httpx.RequestError as e:
                if attempt < max_retries:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Request error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"CHUTES request failed: {e}") from e

