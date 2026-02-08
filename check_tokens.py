import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

tokens = model.to_str_tokens("washing machine")
print(f"Tokens for 'washing machine': {tokens}")

ids = model.to_tokens("washing machine")
print(f"Token IDs: {ids}")

tokens_washing = model.to_str_tokens(" washing")
print(f"Tokens for ' washing': {tokens_washing}")

tokens_machine = model.to_str_tokens(" machine")
print(f"Tokens for ' machine': {tokens_machine}")
