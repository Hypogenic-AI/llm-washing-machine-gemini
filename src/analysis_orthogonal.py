
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sklearn.linear_model import LinearRegression
import json
import os

def load_model():
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    return model

def get_layer_vectors(model, prompts, token_of_interest):
    all_resids = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        idx = -1
        for i, t in enumerate(str_tokens):
            if token_of_interest.strip() in t: idx = i
        if idx == -1: continue
        _, cache = model.run_with_cache(tokens)
        prompt_resids = []
        for L in range(model.cfg.n_layers):
            prompt_resids.append(cache[f"blocks.{L}.hook_resid_post"][0, idx, :].detach().cpu())
        all_resids.append(torch.stack(prompt_resids))
    return torch.stack(all_resids).mean(dim=0)

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
    vecs_wm = get_layer_vectors(model, wm_prompts, " machine")
    vecs_w = get_layer_vectors(model, w_prompts, " washing")
    vecs_m = get_layer_vectors(model, m_prompts, " machine")
    
    n_layers = vecs_wm.shape[0]
    
    results = {
        "layers": list(range(n_layers)),
        "alpha_w": [], # Coeff for Washing
        "beta_m": [],  # Coeff for Machine
        "r2": [],      # Explained Variance
        "resid_ratio": []
    }
    
    for L in range(n_layers):
        y = vecs_wm[L].numpy().reshape(1, -1) # Target: WM
        X_w = vecs_w[L].numpy().reshape(1, -1)
        X_m = vecs_m[L].numpy().reshape(1, -1)
        
        # We want to solve y = a*X_w + b*X_m
        # But X is shape [1, d_model]. We have d_model equations? 
        # No, we treat dimensions as samples? 
        # Yes, we are asking: can the vector y be formed by linear combo of vectors X_w and X_m?
        # So we transpose.
        
        target = y.T
        features = np.concatenate([X_w.T, X_m.T], axis=1)
        
        reg = LinearRegression(fit_intercept=False).fit(features, target)
        
        alpha = reg.coef_[0][0]
        beta = reg.coef_[0][1]
        score = reg.score(features, target)
        
        # Residual
        pred = reg.predict(features)
        resid = target - pred
        resid_norm = np.linalg.norm(resid)
        target_norm = np.linalg.norm(target)
        
        results["alpha_w"].append(float(alpha))
        results["beta_m"].append(float(beta))
        results["r2"].append(float(score))
        results["resid_ratio"].append(float(resid_norm / target_norm))
        
    with open("results/orthogonal_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Explained Variance (R2)', color='tab:blue')
    ax1.plot(results["layers"], results["r2"], color='tab:blue', label='R2 Score', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Coefficients', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(results["layers"], results["alpha_w"], color='tab:orange', label='Coeff Washing (alpha)', linestyle='--')
    ax2.plot(results["layers"], results["beta_m"], color='tab:red', label='Coeff Machine (beta)', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("Decomposition of 'Washing Machine' Vector: WM ≈ α*W + β*M")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.savefig("results/figures/orthogonal_analysis.png", bbox_inches='tight')
    print("Plot saved to results/figures/orthogonal_analysis.png")

if __name__ == "__main__":
    main()
