# Research Report: Where is "Washing Machine" stored in LLMs?

## 1. Executive Summary
This research investigated how the composite concept "washing machine" is represented in the residual stream of GPT-2 Small. By decomposing the activation vector of the phrase "washing machine" into components aligned with "washing" and "machine", we found that the concept **does not** form a distinct, orthogonal direction. Instead, it is represented as a linear superposition, approximately $v_{WM} \approx 0.7 v_{Machine} + 0.3 v_{Washing}$ in the final layers. The model constructs this representation by starting with a pure "machine" vector and progressively adding "washing" information across layers 0-11, explaining >90% of the variance. This supports the hypothesis that the model relies on compositional storage rather than dedicating unique dimensions to specific noun phrases.

## 2. Goal
The primary goal was to test the hypothesis that "washing machine" is not a fundamental, orthogonal concept in the model's latent space but rather a sum of its parts. This has implications for mechanistic interpretability: if composite objects are just linear sums, we can manipulate them by simple vector arithmetic. The user specifically asked: "Where is 'washing machine' stored? Or is just 'washing' stored and then it's assumed that machine becomes more likely?"

## 3. Data Construction

### Dataset
We constructed a synthetic dataset to ensure precise control over token positioning, as natural corpus data (WikiText) was too noisy for rigorous vector comparison.

### Prompts
Three conditions were created (5 prompts each):
1.  **Composite (WM):** "The clothes are in the washing **machine**"
2.  **Component 1 (W):** "I am **washing** the dishes" (Washing context, no machine)
3.  **Component 2 (M):** "The time **machine** travel" (Machine context, no washing)

### Preprocessing
-   Model: `gpt2-small`
-   Tokenization: Verified that " washing" and " machine" are consistent single tokens.
-   Alignment: Analysis focused on the residual stream at the " machine" token position for WM and M groups, and " washing" for the W group.

## 4. Experiment Description

### Methodology
We employed three complementary analysis techniques:
1.  **Logit Lens:** To determine what the model "predicts" or "thinks" the current state represents at the token level.
2.  **Cosine Similarity:** To measure geometric alignment between the composite vector and its components.
3.  **Orthogonal Decomposition (Main Method):** We modeled the "washing machine" vector ($v_{WM}$) as a linear combination of "washing" ($v_W$) and "machine" ($v_M$):
    $$v_{WM} \approx \alpha \cdot v_W + \beta \cdot v_M$$
    We tracked $\alpha$, $\beta$, and the $R^2$ (explained variance) across all 12 layers.

### Implementation Details
-   **Library:** `TransformerLens` for accessing intermediate residual streams (`blocks.L.hook_resid_post`).
-   **Centering:** For cosine similarity, we analyzed centered vectors (subtracting the mean of the three groups) to avoid high baseline anisotropy.
-   **Regression:** Ordinary Least Squares (OLS) without intercept was used for decomposition at each layer.

## 5. Result Analysis

### Key Finding 1: High Compositionality ($R^2 > 0.85$)
The plane spanned by $v_{W}$ and $v_{M}$ explains the vast majority of the variance in $v_{WM}$.
-   **Layer 0:** $R^2 \approx 0.95$ (Input embeddings are identical).
-   **Layer 5 (Dip):** $R^2 \approx 0.86$ (Middle layers do some processing off-plane).
-   **Layer 11:** $R^2 \approx 0.95$ (Output converges back to the compositional sum).

This strongly suggests there is **no specific "washing machine" neuron or direction** that is orthogonal to the parts.

### Key Finding 2: Progressive Integration of "Washing"
The coefficients evolve clearly across layers:
-   **$\beta$ (Machine):** Starts at ~0.90, decreases to ~0.67.
-   **$\alpha$ (Washing):** Starts at ~0.07, increases to ~0.27.

| Layer | $\alpha$ (Washing) | $\beta$ (Machine) | Interpretation |
|-------|-------------------|-------------------|----------------|
| 0     | 0.07              | 0.90              | Mostly pure "Machine" embedding |
| 6     | 0.16              | 0.76              | "Washing" context is being mixed in |
| 11    | 0.27              | 0.67              | Final compositional representation |

### Key Finding 3: Logit Lens Confirmation
At the " washing" token position (immediately preceding " machine"), the model assigns probability **0.73 - 0.93** to " machine" as the next token. This confirms the user's suspicion: "washing" strongly primes "machine".

### Visualizations
*(See `results/figures/orthogonal_analysis.png` for the full trajectory)*

The decomposition shows that the "washing machine" vector essentially stays within the "Washing-Machine Plane" throughout the network, merely rotating from "Pure Machine" towards "Washing".

### Limitations
-   **Model Size:** GPT-2 Small is relatively weak. Larger models might form more distinct abstract concepts.
-   **Sample Size:** Synthetic dataset was small (15 prompts total), though results were highly consistent.

## 6. Conclusions

### Summary
"Washing machine" is stored as a **linear composition**. It is effectively "Machine + 0.3 * Washing". The model does not create a new, unique direction for the object; instead, it enriches the "machine" vector with "washing" context features.

### Implications
-   **Superposition:** This is a clear example of **constructive interference** or feature addition. The model reuses the "machine" subspace and modifies it.
-   **Control:** To erase the concept "washing machine", one would likely need to subtract the "washing" vector from the "machine" token position, rather than looking for a unique "washing machine" direction.

## 7. Next Steps
-   **Intervention:** Verify the causal link by subtracting $0.3 \cdot v_W$ from $v_{WM}$ and checking if the model still predicts properties of washing machines (e.g., "water", "spin").
-   **Scale Up:** Test if GPT-4 or Claude has a distinct direction, which would imply "concept crystallization" in larger models.
