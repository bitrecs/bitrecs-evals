
import os
import sys
import pytest
project_root = os.getcwd()
sys.path.insert(0, project_root)
from llm.factory import LLMFactory
from llm.llm_provider import LLM

@pytest.mark.asyncio
async def test_open_router_query_with_data():        

    provider = LLM.OPEN_ROUTER
    model = "qwen/qwen3.5-flash-02-23"
    result = LLMFactory.query_llm_with_usage(server=provider, 
                                  model=model,
                                  system_prompt="You are a helpful assistant.",
                                  temp=0.0,
                                  user_prompt="What is the capital of France?")
    
    response = result.response
    data = result.data
    print(f"Response: {response}")
    print(f"Usage Data: {data}")

    assert isinstance(response, str)
    assert isinstance(data, dict)
  

@pytest.mark.asyncio
async def test_chutes_query_with_data():
    provider = LLM.CHUTES
    model = "unsloth/Mistral-Nemo-Instruct-2407"
    result = LLMFactory.query_llm_with_usage(server=provider, 
                                  model=model,
                                  system_prompt="You are a helpful assistant.",
                                  temp=0.0,
                                  user_prompt="What is the capital of France?")
    
    response = result.response
    data = result.data
    print(f"Response: {response}")
    print(f"Usage Data: {data}")

    assert isinstance(response, str)
    assert isinstance(data, dict)
 