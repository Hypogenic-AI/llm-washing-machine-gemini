from datasets import load_dataset
import os
import json

# Create directory
output_dir = "datasets/washing_machine_corpus"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

keywords = ["washing machine", "washing", "machine"]

def filter_function(example):
    text = example['text'].lower()
    return any(k in text for k in keywords)

print("Filtering dataset...")
filtered_dataset = dataset.filter(filter_function)

print(f"Original train size: {len(dataset['train'])}")
print(f"Filtered train size: {len(filtered_dataset['train'])}")

print("Saving to disk...")
filtered_dataset.save_to_disk(output_dir)

# Save a sample
samples = []
for i in range(min(10, len(filtered_dataset['train']))):
    samples.append(filtered_dataset['train'][i])

with open(os.path.join(output_dir, "samples.json"), "w") as f:
    json.dump(samples, f, indent=2)

print("Done.")
