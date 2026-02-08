
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import json
import os

def load_model():
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    return model

def get_layer_vectors(model, prompts, token_of_interest):
    """
    Get residual stream vectors for ALL layers at specific token position.
    Returns: [layers, d_model] (averaged over prompts)
    """
    layer_vectors = []
    
    # We'll batch this properly or just loop
    # For simplicity and small data, loop is fine
    
    all_resids = [] # [prompt, layer, d_model]
    
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        
        idx = -1
        for i, t in enumerate(str_tokens):
            if token_of_interest.strip() in t:
                idx = i
        
        if idx == -1: continue # Skip if not found
        
        _, cache = model.run_with_cache(tokens)
        
        prompt_resids = []
        for L in range(model.cfg.n_layers):
            # hook_resid_post: output of the block
            resid = cache[f"blocks.{L}.hook_resid_post"][0, idx, :].detach().cpu()
            prompt_resids.append(resid)
        
        all_resids.append(torch.stack(prompt_resids))
        
    # Stack: [n_prompts, n_layers, d_model]
    stack = torch.stack(all_resids)
    # Mean over prompts: [n_layers, d_model]
    return stack.mean(dim=0)

def main():
    model = load_model()
    
    wm_prompts = [
        "The clothes are in the washing machine",
        "I need to buy a new washing machine",
        "The washing machine is broken",
        "She loaded the washing machine",
        "Repair the washing machine"
    ]
    w_prompts = [
        "I am washing the dishes",
        "She is washing her hands",
        "He is washing the car",
        "Stop washing the floor",
        "Keep washing the vegetables"
    ]
    m_prompts = [
        "The time machine travel",
        "A slot machine game",
        "The vending machine stuck",
        "A sewing machine needle",
        "The fax machine beeped"
    ]
    
    print("Extracting vectors...")
    # [layers, d_model]
    vecs_wm = get_layer_vectors(model, wm_prompts, " machine")
    vecs_w = get_layer_vectors(model, w_prompts, " washing") # Comparing to 'washing' concept
    vecs_m = get_layer_vectors(model, m_prompts, " machine") # Comparing to 'machine' concept
    
    n_layers = vecs_wm.shape[0]
    
    # Centering
    # We calculate one global mean vector per layer to center the space
    # or mean of these 3 concepts. 
    # Let's use mean of these 3 to see relative geometry.
    
    sims_wm_m = []
    sims_wm_w = []
    sims_wm_sum = []
    
    for L in range(n_layers):
        v_wm = vecs_wm[L]
        v_w = vecs_w[L]
        v_m = vecs_m[L]
        
        # Center at this layer
        mean_vec = (v_wm + v_w + v_m) / 3
        
        c_wm = v_wm - mean_vec
        c_w = v_w - mean_vec
        c_m = v_m - mean_vec
        
        # Sim(WM, M)
        sims_wm_m.append(torch.cosine_similarity(c_wm, c_m, dim=0).item())
        
        # Sim(WM, W)
        sims_wm_w.append(torch.cosine_similarity(c_wm, c_w, dim=0).item())
        
        # Sim(WM, W+M)
        # For centered arithmetic, we check if c_wm is similar to (c_w + c_m)
        # Note: c_w + c_m = (v_w - mu) + (v_m - mu). 
        # Ideally we want to check if raw v_wm ≈ v_w + v_m, but we need to handle the shift.
        # Let's stick to checking if the *direction* of WM aligns with the sum of directions.
        
        c_sum = c_w + c_m
        sims_wm_sum.append(torch.cosine_similarity(c_wm, c_sum, dim=0).item())

    # Save Results
    results = {
        "layers": list(range(n_layers)),
        "sim_wm_m": sims_wm_m,
        "sim_wm_w": sims_wm_w,
        "sim_wm_sum": sims_wm_sum
    }
    
    with open("results/refined_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["layers"], sims_wm_m, label="Sim(WM, Machine)", marker='o')
    plt.plot(results["layers"], sims_wm_w, label="Sim(WM, Washing)", marker='s')
    plt.plot(results["layers"], sims_wm_sum, label="Sim(WM, Washing+Machine)", marker='^', linestyle='--')
    
    plt.title("Layer-wise Centered Cosine Similarity")
    plt.xlabel("Layer")
    plt.ylabel("Centered Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/figures/layerwise_similarity.png")
    print("Plot saved to results/figures/layerwise_similarity.png")

if __name__ == "__main__":
    main()
