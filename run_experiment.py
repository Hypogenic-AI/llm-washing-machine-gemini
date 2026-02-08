import torch
from transformer_lens import HookedTransformer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import json
import os

# Configuration
DEVICE = "cpu" # "cuda" if available, but cpu is fine for inference on small model
MODEL_NAME = "gpt2-small"
DATASET_PATH = "datasets/washing_machine_corpus"
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print(f"Loading model {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)

# Token IDs
# We need to be careful with spacing. 
# " washing machine" -> " washing" (space) + " machine" (space)
# "washing machine" -> "washing" (no space) + " machine" (space)
# Most usually in text: " washing machine" (middle of sentence)

# Let's find the IDs for " machine" and " washing"
# Note: GPT-2 tokens usually include the leading space.
token_machine = model.to_single_token(" machine")
token_washing = model.to_single_token(" washing")

print(f"Target Token ' machine': {token_machine}")
print(f"Target Token ' washing': {token_washing}")

activations_washing_machine = [] # ' machine' when preceded by ' washing'
activations_other_machine = []   # ' machine' when NOT preceded by ' washing'
activations_washing = []         # ' washing' tokens

# Iterate and extract
print("Processing examples...")
for example in tqdm(dataset['train']):
    text = example['text']
    try:
        # Prepend space to ensure consistent tokenization if start of string?
        # Actually, let's just use the text as is, but handle tokenization carefully.
        tokens = model.to_tokens(text, prepend_bos=True)[0] # Shape [seq_len]
        
        # Convert to list for easier indexing
        tokens_list = tokens.tolist()
        
        # Find occurrences
        for i in range(len(tokens_list)):
            token = tokens_list[i]
            
            # Case 1: Found " washing"
            if token == token_washing:
                # Get residual stream at this position
                # We need to run the model. Efficient way: run batch? 
                # For simplicity/speed in this script, let's run one by one or minimal caching.
                # To get activations efficiently, we can run model on the whole sequence 
                # and pick indices.
                pass 
                
            # We'll do a second pass or structure this better to avoid re-running model multiple times per line.
            
    except Exception as e:
        print(f"Error processing text: {e}")
        continue

# Better approach: 
# 1. Collect all valid indices for each category.
# 2. Run model once per example, extract relevant vectors.

count_wm = 0
count_m = 0
count_w = 0

with torch.no_grad():
    for example in tqdm(dataset['train']):
        text = example['text']
        tokens = model.to_tokens(text, prepend_bos=True)[0]
        tokens_list = tokens.tolist()
        
        # Check if we have anything interesting
        has_machine = token_machine in tokens_list
        has_washing = token_washing in tokens_list
        
        if not (has_machine or has_washing):
            continue
            
        # Run model to get cache
        # We want the residual stream at the end of the model (before LN_f) or output of last layer?
        # Usually "resid_post" of the last layer.
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("resid_post"))
        
        # Get the tensor for the last layer's residual stream
        # Shape: [batch=1, seq_len, d_model]
        resid = cache["blocks.11.hook_resid_post"][0] 
        
        for i, token in enumerate(tokens_list):
            
            # Collect ' washing' vectors
            if token == token_washing:
                vec = resid[i].cpu().numpy()
                activations_washing.append(vec)
                count_w += 1
                
            # Collect ' machine' vectors
            if token == token_machine:
                # Check previous token
                if i > 0 and tokens_list[i-1] == token_washing:
                    # It is " washing machine"
                    vec = resid[i].cpu().numpy()
                    activations_washing_machine.append(vec)
                    count_wm += 1
                else:
                    # It is " machine" but NOT " washing machine"
                    # Note: could be "sewing machine", "time machine", etc.
                    vec = resid[i].cpu().numpy()
                    activations_other_machine.append(vec)
                    count_m += 1

print(f"Collected counts:")
print(f"  ' washing': {count_w}")
print(f"  ' washing machine': {count_wm}")
print(f"  ' machine' (other): {count_m}")

# Analysis
if count_wm < 5 or count_m < 5 or count_w < 5:
    print("WARNING: Not enough data for robust analysis.")

# Compute means
mean_w = np.mean(activations_washing, axis=0) if activations_washing else np.zeros(768)
mean_wm = np.mean(activations_washing_machine, axis=0) if activations_washing_machine else np.zeros(768)
mean_m_other = np.mean(activations_other_machine, axis=0) if activations_other_machine else np.zeros(768)

# 1. Cosine Similarity between "washing machine" (whole) and "machine" (other)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_wm_m = cosine_sim(mean_wm, mean_m_other)
print(f"Cosine Sim (Washing Machine vs Other Machine): {sim_wm_m:.4f}")

# 2. Vector Difference: What does "washing" add to "machine"?
diff_vector = mean_wm - mean_m_other

# 3. Does this difference align with the "washing" vector?
sim_diff_w = cosine_sim(diff_vector, mean_w)
print(f"Cosine Sim ((WM - M) vs Washing): {sim_diff_w:.4f}")

# 4. Check linearity: wm approx m + w?
# composition = mean_m_other + mean_w
# sim_composition = cosine_sim(mean_wm, composition)
# print(f"Cosine Sim (WM vs (M + W)): {sim_composition:.4f}")

# Save results
results = {
    "counts": {
        "washing": count_w,
        "washing_machine": count_wm,
        "other_machine": count_m
    },
    "metrics": {
        "sim_wm_vs_m_other": float(sim_wm_m),
        "sim_diff_vs_washing": float(sim_diff_w)
    }
}

with open(os.path.join(RESULTS_DIR, "experiment_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Experiment complete. Results saved.")
