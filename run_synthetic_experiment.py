import torch
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm
import json
import os

# Configuration
DEVICE = "cpu"
MODEL_NAME = "gpt2-small"
DATASET_PATH = "datasets/synthetic/dataset.json"
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print(f"Loading model {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, "r") as f:
    raw_data = json.load(f)

# Token IDs
# ' washing' (20518), ' machine' (4572)
token_machine = model.to_single_token(" machine")
token_washing = model.to_single_token(" washing")

print(f"Target Token ' machine': {token_machine}")
print(f"Target Token ' washing': {token_washing}")

activations_washing_machine = [] # ' machine' in ' washing machine'
activations_other_machine = []   # ' machine' in ' [other] machine' or ' machine'
activations_washing_verb = []    # ' washing' in ' washing [obj]' (NOT machine)
activations_washing_in_wm = []   # ' washing' in ' washing machine'

count_wm = 0
count_m_other = 0
count_w_verb = 0

print("Processing examples...")
with torch.no_grad():
    for item in tqdm(raw_data):
        text = item['text']
        dataset_type = item['type'] # washing_machine, other_machine, generic_machine, washing_verb
        
        tokens = model.to_tokens(text, prepend_bos=True)[0]
        tokens_list = tokens.tolist()
        
        # Run model
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("resid_post"))
        resid = cache["blocks.11.hook_resid_post"][0] # Last layer
        
        for i, token in enumerate(tokens_list):
            
            # Check for ' washing'
            if token == token_washing:
                # Is the NEXT token ' machine'?
                if i + 1 < len(tokens_list) and tokens_list[i+1] == token_machine:
                    # It is " washing" in " washing machine"
                    vec = resid[i].cpu().numpy()
                    activations_washing_in_wm.append(vec)
                else:
                    # It is " washing" NOT followed by machine (e.g. washing the car)
                    vec = resid[i].cpu().numpy()
                    activations_washing_verb.append(vec)
                    count_w_verb += 1
            
            # Check for ' machine'
            if token == token_machine:
                # Is the PREVIOUS token ' washing'?
                if i > 0 and tokens_list[i-1] == token_washing:
                    # It is " machine" in " washing machine"
                    vec = resid[i].cpu().numpy()
                    activations_washing_machine.append(vec)
                    count_wm += 1
                else:
                    # It is " machine" in other contexts
                    vec = resid[i].cpu().numpy()
                    activations_other_machine.append(vec)
                    count_m_other += 1

print(f"Collected counts:")
print(f"  ' washing' (verb): {count_w_verb}")
print(f"  ' washing' (in WM): {len(activations_washing_in_wm)}")
print(f"  ' washing machine' (machine token): {count_wm}")
print(f"  ' machine' (other): {count_m_other}")

# Analysis

# Compute means
mean_w_verb = np.mean(activations_washing_verb, axis=0)
mean_w_in_wm = np.mean(activations_washing_in_wm, axis=0)
mean_wm_machine = np.mean(activations_washing_machine, axis=0)
mean_m_other = np.mean(activations_other_machine, axis=0)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# 1. washing(verb) vs washing(in WM)
sim_w_contexts = cosine_sim(mean_w_verb, mean_w_in_wm)
print(f"Cosine Sim (Washing(verb) vs Washing(in WM)): {sim_w_contexts:.4f}")
# Expect high similarity if 'washing' is stable

# 2. machine(WM) vs machine(other)
sim_m_contexts = cosine_sim(mean_wm_machine, mean_m_other)
print(f"Cosine Sim (Machine(WM) vs Machine(Other)): {sim_m_contexts:.4f}")

# 3. Vector Addition Hypothesis
# Does v(Machine_WM) = v(Machine_Other) + v(Washing_Verb)?
# Or rather, does the *difference* align with Washing?
diff_wm_m = mean_wm_machine - mean_m_other
sim_diff_w = cosine_sim(diff_wm_m, mean_w_verb)
print(f"Cosine Sim ((Machine_WM - Machine_Other) vs Washing_Verb): {sim_diff_w:.4f}")

# 4. Orthogonality Check
# Is the "Washing Machine" concept (the difference) orthogonal to "Washing"?
# If sim_diff_w is low, it implies orthogonality (new concept).
# If high, it implies composition.

# Save detailed results
results = {
    "counts": {
        "washing_verb": count_w_verb,
        "washing_in_wm": len(activations_washing_in_wm),
        "washing_machine_token": count_wm,
        "other_machine": count_m_other
    },
    "metrics": {
        "sim_washing_contexts": sim_w_contexts,
        "sim_machine_contexts": sim_m_contexts,
        "sim_diff_vs_washing": sim_diff_w
    }
}

with open(os.path.join(RESULTS_DIR, "synthetic_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Synthetic experiment complete.")
