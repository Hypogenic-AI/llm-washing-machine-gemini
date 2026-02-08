from datasets import load_from_disk
import os

dataset = load_from_disk("datasets/washing_machine_corpus")

count_wm = 0
count_w = 0
count_m = 0

for example in dataset['train']:
    text = example['text'].lower()
    if "washing machine" in text:
        count_wm += 1
    if "washing" in text and "washing machine" not in text:
        count_w += 1
    if "machine" in text and "washing machine" not in text:
        count_m += 1

print(f"Total examples: {len(dataset['train'])}")
print(f"Examples with 'washing machine': {count_wm}")
print(f"Examples with 'washing' (only): {count_w}")
print(f"Examples with 'machine' (only): {count_m}")
