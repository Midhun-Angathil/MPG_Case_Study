# For token counting
import tiktoken

# Helper: Estimate token count and cost for gpt-3.5-turbo-16k
def estimate_cost(prompt, model="gpt-3.5-turbo-16k", output_tokens=75):
    input_token_price = 0.005 / 1000
    output_token_price = 0.015 / 1000
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(prompt))
    est_cost = input_tokens * input_token_price + output_tokens * output_token_price
    return input_tokens, output_tokens, round(est_cost, 4)