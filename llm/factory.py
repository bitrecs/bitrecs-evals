import os
import asyncio
from typing import List, Dict, Optional
from pydantic.dataclasses import dataclass
from llm.chutes import Chutes
from llm.llama_local import OllamaLocal
from llm.llm_provider import LLM
from llm.open_router import OpenRouter

@dataclass
class QueryWithData:
    response: str
    data: Dict

class LLMFactory:

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
            

    @staticmethod
    def query_llm_with_usage(server: LLM, model: str, 
                  system_prompt="You are a helpful assistant", 
                  temp=0.0, user_prompt="") -> QueryWithData:
        match server:
            case LLM.OLLAMA_LOCAL:
                raise NotImplementedError("Ollama local usage data not implemented yet")
            case LLM.OPEN_ROUTER:
                return OpenRouterInterface(model, system_prompt, temp).query_with_data(user_prompt)
            # case LLM.CHAT_GPT:
            #     return ChatGPTInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.VLLM:
            #     return VllmInterface(model, system_prompt, temp).query(user_prompt)
            # case LLM.GEMINI:
            #     return GeminiInterface(model, system_prompt, temp).query(user_prompt)         
            case LLM.CHUTES:
                return ChutesInterface(model, system_prompt, temp).query_with_data(user_prompt)
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
        response, usage_data = router.call_open_router(user_prompt)
        return response

    def query_with_data(self, user_prompt) -> QueryWithData:
        router = OpenRouter(self.OPENROUTER_API_KEY, model=self.model, 
                            system_prompt=self.system_prompt, temp=self.temp)
        response, usage_data = router.call_open_router(user_prompt)
        return QueryWithData(response=response, data=usage_data)


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
        response, usage_data = chutes.call_chutes(user_prompt)
        return response
    
    
    def query_with_data(self, user_prompt) -> QueryWithData:
        chutes = Chutes(key=self.CHUTES_API_KEY, model=self.model, 
                        system_prompt=self.system_prompt, temp=self.temp)
        response, usage_data = chutes.call_chutes(user_prompt)
        return QueryWithData(response=response, data=usage_data)
      