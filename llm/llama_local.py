import os
import time
import base64
import httpx
import logging
from typing import Tuple
from common import constants as CONST
from llm.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)

class OllamaLocal():
    def __init__(self, 
                 ollama_url: str, 
                 model: str, 
                 system_prompt: str, 
                 temp=0.0):
        
        if not ollama_url:
            raise Exception
        self.ollama_url = ollama_url
        self.model = model
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        self.system_prompt = system_prompt        
        if temp < 0 or temp > 1:
            raise Exception
        self.temp = temp
        self.keep_alive = 3600


    def file_to_base64(self, file_path) -> str:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
        
        
    def test_warmup(self) -> Tuple[bool, str]:
        test_prompt = "What is 7 + 5? Respond with only the number."
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": test_prompt,
                    "stream": False,
                    "system": "You are a helpful assistant. Answer questions accurately and concisely."
                }
            ],
            "stream": False,
            "keep_alive": 300,
            "options": {
                "temperature": 0.0,
                "seed": 42,
            }
        }
        
        try:
            with httpx.Client() as client:
                response = client.post(self.ollama_url, json=data, timeout=30)
            if response.status_code != 200:
                return False, f"HTTP error: {response.status_code}"
            response_json = response.json()
            if "message" not in response_json:
                return False, f"Invalid response structure: {response_json}"
            
            message = response_json.get("message", {})
            content = message.get("content", "").strip()
            if not content:
                return False, "Empty response from LLM"
            
            error_indicators = ["error", "not loaded", "failed", "unable", "cannot", "sorry"]
            if any(indicator in content.lower() for indicator in error_indicators):
                return False, f"LLM error response: {content}"
            # Check for the correct mathematical answer
            if "12" in content:
                return True, f"LLM responding correctly: {content}"
            else:
                return False, f"Incorrect or unexpected response: {content}"                
        except httpx.TimeoutException:
            return False, "Request timeout - Ollama server may be overloaded"
        except httpx.ConnectError:
            return False, "Connection error - Ollama server may be down"
        except Exception as e:
            return False, f"Exception during warmup test: {str(e)}"
        
    def ask_ollama(self, prompt) -> str:
        #return self.ask_ollama_long_ctx(prompt, 8000)
        data = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temp                
            }
        }
        # print(data)
        return self.call_ollama(data)
    
        
    def ask_ollama_long_ctx(self, prompt, num_ctx: int = None) -> str:
        """Send a prompt to Ollama with optional longer context window.
        
        Args:
            prompt (str): The prompt to send to the model
            num_ctx (int, optional): Context window size. If None, uses environment variable
                
        Returns:
            str: The model's response
        """
        options = {
            "temperature": self.temp,
        }        
     
        if num_ctx is not None:
            options["num_ctx"] = max(int(num_ctx), 2048)        
        elif os.environ.get("num_ctx") is not None:
            env_ctx = os.environ.get("num_ctx")
            try:
                ctx_value = max(int(env_ctx), 2048)
                options["num_ctx"] = ctx_value
                print(f"Using context length from environment: {ctx_value}")
            except ValueError:
                print(f"Invalid context length in environment: {env_ctx}, using default 2048")
                options["num_ctx"] = 2048
        else:
            options["num_ctx"] = 2048

        data = {
            "model": self.model,       
            "system": self.system_prompt,    
            "messages": [              
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options
        }
        return self.call_ollama(data)
    

    def get_ollama_caption(self, file_path) -> str:        
        base64_image = self.file_to_base64(file_path)
        data = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": "What is this?",
                    "stream": False,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
             "options": {
                "temperature": self.temp
            }
        }
        # print(data)
        return self.call_ollama(data)   


    def call_ollama(self, data) -> str:
        timeout = (5, 60) #connect, read timeout
        with httpx.Client() as client:
            if CONST.LOG_LEVEL <= logging.DEBUG:
                start_time = time.perf_counter()
                content = data["messages"][0]["content"]
                token_count = PromptFactory.get_token_count(content)
                logger.debug(f"OLLAMA_LOCAL request token count: {token_count} tokens")

            response = client.post(self.ollama_url, json=data, timeout=timeout)
            if response.status_code == 200:
                response_json = response.json()
                message = response_json["message"]
                content = message["content"]
                if CONST.LOG_LEVEL <= logging.DEBUG:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    logger.debug(f"OLLAMA_LOCAL request completed in {duration:.2f}s")
                return content
            else:
                print(response.text)
                return "Error: Unable to get caption from LLama server status {}".format(response.status_code)
        
  