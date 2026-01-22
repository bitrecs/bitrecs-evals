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
        timeout = (3, 30)  # Reduced connect/read timeouts
        max_retries = 1  # Reduced to 1 retry (2 total attempts)
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

