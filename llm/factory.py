import os
import asyncio
from typing import List, Dict, Optional
from llm.chutes import Chutes
from llm.llama_local import OllamaLocal
from llm.llm_provider import LLM
from llm.open_router import OpenRouter


class LLMFactory:

    @staticmethod
    async def async_llm_call(server: LLM, 
                             model: str, 
                             messages: List[Dict[str, str]], 
                             temperature: float, 
                             max_tokens: int, 
                             top_p: float, 
                             stop: Optional[List[str]] = None) -> str:
        """
        Async wrapper for LLM calls. Parses messages to extract system and user prompts,
        then calls the synchronous query_llm in a thread pool.
        
        Note: max_tokens, top_p, and stop parameters are not yet supported by all interfaces.
        They will be passed where possible (e.g., OpenRouter).
        """
        # Parse messages to extract system and user prompts
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg.get('role') == 'system':
                system_prompt = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_prompt = msg.get('content', '')
        
        # Call sync method in thread pool
        return await asyncio.to_thread(
            LLMFactory.query_llm,
            server=server,
            model=model,
            system_prompt=system_prompt,
            temp=temperature,
            user_prompt=user_prompt
        )   
  

    @staticmethod
    def query_llm(server: LLM, model: str, 
                  system_prompt="You are a helpful assistant", 
                  temp=0.0, user_prompt="") -> str:
        match server:
            case LLM.OLLAMA_LOCAL:
                return OllamaLocalInterface(model, system_prompt, temp).query(user_prompt)
            case LLM.OPEN_ROUTER:
                return OpenRouterInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.CHAT_GPT:
            #     return ChatGPTInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.VLLM:
            #     return VllmInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.GEMINI:
            #     return GeminiInterface(model, system_prompt, temp).query(user_prompt)         
            case LLM.CHUTES:
                return ChutesInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.GROK:
            #     return GrokInterface(model, system_prompt, temp).query(user_prompt)                
            # case LLM.CLAUDE:
            #     return ClaudeInterface(model, system_prompt, temp).query(user_prompt)                
            # case LLM.CEREBRAS:
            #     return CerebrasInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.GROQ:
            #     return GroqInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.NVIDIA:
            #     return NvidiaInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.PERPLEXITY:
            #     return PerplexityInterface(model, system_prompt, temp).query(user_prompt)
            case _:
                raise ValueError("Unknown LLM server")
            

class OpenRouterInterface:
    def __init__(self, model, system_prompt, temp, miner_wallet = None, use_verified_inference = False):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
        self.miner_wallet = miner_wallet
        self.use_verified_inference = use_verified_inference
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
    
    def query(self, user_prompt) -> str:
        router = OpenRouter(self.OPENROUTER_API_KEY, model=self.model, 
                            system_prompt=self.system_prompt, temp=self.temp)
        return router.call_open_router(user_prompt)


class OllamaLocalInterface:
    def __init__(self, model, system_prompt, temp):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.OLLAMA_LOCAL_URL = os.environ.get("OLLAMA_LOCAL_URL").removesuffix("/")
        if not self.OLLAMA_LOCAL_URL:
            raise ValueError("OLLAMA_LOCAL_URL is not set")
    
    def query(self, user_prompt) -> str:
        llm = OllamaLocal(ollama_url=self.OLLAMA_LOCAL_URL, model=self.model, 
                          system_prompt=self.system_prompt, temp=self.temp)
        return llm.ask_ollama(user_prompt)
    
class ChutesInterface:
    def __init__(self, model, system_prompt, temp):
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp
        self.CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY")
        if not self.CHUTES_API_KEY:
            raise ValueError("CHUTES_API_KEY is not set")
    
    def query(self, user_prompt) -> str:
        chutes = Chutes(key=self.CHUTES_API_KEY, model=self.model, 
                        system_prompt=self.system_prompt, temp=self.temp)
        return chutes.call_chutes(user_prompt)