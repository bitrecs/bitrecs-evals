import unittest
from llm.inference_coster import InferenceCoster


class TestInferenceCosterIntegration(unittest.TestCase):
    def test_chutes_fetch(self):
        coster = InferenceCoster(provider="CHUTES", model_name="openai/gpt-oss-120b-TEE")
        pricing = coster.fetch_cost()
        print(f"Input per million tokens: {pricing.input}, Output per million tokens: {pricing.output}")
        assert pricing.input >= 0.0
        assert pricing.output >= 0.0

    def test_openrouter_fetch(self):
        coster = InferenceCoster(provider="OPEN_ROUTER", model_name="anthropic/claude-opus-4.6")
        pricing = coster.fetch_cost()
        print(f"Input per million tokens: {pricing.input}, Output per million tokens: {pricing.output}")
        assert pricing.input >= 0.0
        assert pricing.output >= 0.0        

    def test_unknown_model(self):
        coster = InferenceCoster(provider="OPEN_ROUTER", model_name="unknown-model")
        pricing = coster.fetch_cost()
        assert pricing is None

    def test_calculate_cost(self):
        coster = InferenceCoster(provider="OPEN_ROUTER", model_name="anthropic/claude-opus-4.6")
        total_cost = coster.calculate_cost(input_tokens=50_000, output_tokens=2048)
        print(f"Total cost for 50k input tokens and 2k output tokens: ${total_cost:.6f}")
        assert total_cost is not None
        assert total_cost >= 0.30
        

if __name__ == "__main__":
    unittest.main()