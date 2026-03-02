
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from llm.inference_coster import InferenceCoster

project_root = os.getcwd()
sys.path.insert(0, project_root)

from llm.factory import LLMFactory, QueryWithData
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

    coster = InferenceCoster(provider=provider.name, model_name=model)
    input = data.get("prompt_tokens", 0) + data.get("system_tokens", 0)
    output = data.get("completion_tokens", 0)
    print(f"Input tokens: {input}, Output tokens: {output}")
    cost_result = coster.calculate_cost(input_tokens=input, output_tokens=output)
    print(f"Cost Result: {cost_result}")
    assert cost_result is not None
    assert cost_result >= 0


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

    coster = InferenceCoster(provider=provider.name, model_name=model)
    input = data.get("prompt_tokens", 0) + data.get("system_tokens", 0)
    output = data.get("completion_tokens", 0)
    print(f"Input tokens: {input}, Output tokens: {output}")
    cost_result = coster.calculate_cost(input_tokens=input, output_tokens=output)
    print(f"Cost Result: {cost_result}")
    assert cost_result is not None
    assert cost_result >= 0