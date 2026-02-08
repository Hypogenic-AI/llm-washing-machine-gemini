
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from transformer_lens import HookedTransformer
import os
import json

# Ensure directories exist
os.makedirs("results/figures", exist_ok=True)

def load_model():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    return model

def get_vectors(model, prompts, token_of_interest):
    """
    Get the residual stream vector at the specific token position.
    """
    vectors = []
    for prompt in prompts:
        # We need to find the index of the token of interest
        # This is a bit tricky with tokenization.
        # We'll assume the token of interest is at the end or specific place.
        # For this script, we'll design prompts so the token is last or we know the index.
        
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        
        # Find index. This is a heuristic.
        idx = -1
        found = False
        for i, t in enumerate(str_tokens):
            if token_of_interest.strip() in t: # partial match
                idx = i
                found = True
        
        if not found:
            # Fallback for " machine" vs " machine" (space)
            idx = -1 
        
        _, cache = model.run_with_cache(tokens)
        # Get final residual stream (blocks.11.hook_resid_post)
        # Shape: [batch, pos, d_model]
        resid = cache["blocks.11.hook_resid_post"][0, idx, :].detach().cpu()
        vectors.append(resid)
        
    return torch.stack(vectors)

def main():
    model = load_model()
    
    # 1. Synthetic Data
    # Note: Spaces are important for GPT-2 tokenization
    
    # "Washing Machine" prompts
    wm_prompts = [
        "The clothes are in the washing machine",
        "I need to buy a new washing machine",
        "The washing machine is broken",
        "She loaded the washing machine",
        "Repair the washing machine"
    ]
    
    # "Washing" (alone) prompts
    w_prompts = [
        "I am washing the dishes",
        "She is washing her hands",
        "He is washing the car",
        "Stop washing the floor",
        "Keep washing the vegetables"
    ]
    
    # "Machine" (alone) prompts
    m_prompts = [
        "The time machine travel",
        "A slot machine game",
        "The vending machine stuck",
        "A sewing machine needle",
        "The fax machine beeped"
    ]
    
    # Check tokenization first to ensure alignment
    print("Checking tokenization...")
    print(f"WM tokens: {model.to_str_tokens(wm_prompts[0])}")
    # We expect " washing" and " machine" to be distinct tokens.
    
    # 2. Extract Vectors
    print("Extracting vectors...")
    
    # Vector at " machine" in "washing machine" context
    # Ideally, this represents the full "washing machine" concept
    v_wm_full = get_vectors(model, wm_prompts, " machine")
    
    # Vector at " washing" in "washing machine" context
    # This represents "washing" before "machine" is integrated
    v_wm_part1 = get_vectors(model, wm_prompts, " washing")
    
    # Vector at " washing" in "washing only" context
    v_w_only = get_vectors(model, w_prompts, " washing")
    
    # Vector at " machine" in "machine only" context
    v_m_only = get_vectors(model, m_prompts, " machine")
    
    # Mean vectors
    mean_wm_full = v_wm_full.mean(dim=0)
    mean_wm_part1 = v_wm_part1.mean(dim=0)
    mean_w_only = v_w_only.mean(dim=0)
    mean_m_only = v_m_only.mean(dim=0)
    
    # 3. Analysis: Cosine Similarities
    print("Calculating similarities...")
    
    sims = {
        "WM_Full vs W_Only": torch.cosine_similarity(mean_wm_full, mean_w_only, dim=0).item(),
        "WM_Full vs M_Only": torch.cosine_similarity(mean_wm_full, mean_m_only, dim=0).item(),
        "WM_Part1 vs W_Only": torch.cosine_similarity(mean_wm_part1, mean_w_only, dim=0).item(),
        "W_Only vs M_Only": torch.cosine_similarity(mean_w_only, mean_m_only, dim=0).item(),
    }
    
    # 4. Analysis: Vector Arithmetic
    # Does WM_Full ≈ W_Only + M_Only?
    sum_vec = mean_w_only + mean_m_only
    sim_sum = torch.cosine_similarity(mean_wm_full, sum_vec, dim=0).item()
    sims["WM_Full vs (W+M)"] = sim_sum
    
    print("Similarities:", json.dumps(sims, indent=2))
    
    # 5. Logit Lens Analysis
    # What does the model predict after " washing" in "washing machine" context?
    print("\nLogit Lens Analysis...")
    logit_results = []
    
    for prompt in wm_prompts:
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        
        # Find index of " washing"
        idx = -1
        for i, t in enumerate(str_tokens):
            if " washing" in t:
                idx = i
                break
        
        if idx == -1: continue
            
        # Get logits at this position
        logits = model(tokens)[0, idx, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Top predictions
        top_k = torch.topk(probs, k=5)
        top_tokens = [model.to_string(t) for t in top_k.indices]
        top_probs = top_k.values.tolist()
        
        # Rank of " machine"
        machine_token_id = model.to_single_token(" machine")
        machine_rank = (logits > logits[machine_token_id]).sum().item()
        machine_prob = probs[machine_token_id].item()
        
        logit_results.append({
            "prompt": prompt,
            "top_tokens": top_tokens,
            "top_probs": top_probs,
            "machine_rank": machine_rank,
            "machine_prob": machine_prob
        })
        
    # Save results
    results = {
        "similarities": sims,
        "logit_lens": logit_results
    }
    
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Results saved to results/experiment_results.json")
    
    # Plotting
    labels = list(sims.keys())
    values = list(sims.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Cosine Similarity of 'Washing Machine' Representations")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/similarity_plot.png")
    print("Plot saved.")

if __name__ == "__main__":
    main()
