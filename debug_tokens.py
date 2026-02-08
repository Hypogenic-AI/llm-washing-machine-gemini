import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

print(f"ID for ' washing': {model.to_single_token(' washing')}")
print(f"ID for 'washing': {model.to_single_token('washing')}")

text1 = "The washing machine is broken."
tokens1 = model.to_tokens(text1)[0]
print(f"'{text1}' -> {tokens1.tolist()} | {model.to_str_tokens(text1)}")

text2 = "washing machine"
tokens2 = model.to_tokens(text2)[0]
print(f"'{text2}' -> {tokens2.tolist()} | {model.to_str_tokens(text2)}")

text3 = "I am washing the car."
tokens3 = model.to_tokens(text3)[0]
print(f"'{text3}' -> {tokens3.tolist()} | {model.to_str_tokens(text3)}")
