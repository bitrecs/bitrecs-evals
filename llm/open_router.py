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
        

    def call_open_router(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()

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

        data = {
            "model": self.model,
            "messages": [
                #{"role": "system", "content": "/no_think"},
                {
                    "role": "user", 
                    "content": prompt
                }],
            "reasoning": reasoning,
            "stream": False,
            "temperature": self.temp
        }
        
        timeout = (5, 60) #connect, read timeout
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    if CONST.LOG_LEVEL <= logging.DEBUG:
                        start_time = time.perf_counter()
                        content = data["messages"][0]["content"]
                        token_count = PromptFactory.get_token_count(content)
                        logger.debug(f"OPEN_ROUTER request token count: {token_count} tokens")
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
                        logger.debug(f"OPEN_ROUTER request completed in {duration:.2f}s")
                    #print(data)
                    return data['choices'][0]['message']['content']
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"429 received, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenRouter request failed: {e}") from e
            except httpx.ConnectTimeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Connect timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"OpenRouter connect timed out after {timeout[0]}s")
            except httpx.ReadTimeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Read timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(f"OpenRouter read timed out after {timeout[1]}s")
            except httpx.RequestError as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenRouter request failed: {e}") from e