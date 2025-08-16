import unittest
from rag.openai_utils import estimate_cost

class TestOpenAIUtils(unittest.TestCase):
    def test_estimate_cost(self):
        prompt = "Hello, how are you?"
        input_tokens, output_tokens, cost = estimate_cost(prompt)
        self.assertTrue(input_tokens > 0)
        self.assertEqual(output_tokens, 75)
        self.assertTrue(cost > 0)

if __name__ == "__main__":
    unittest.main()