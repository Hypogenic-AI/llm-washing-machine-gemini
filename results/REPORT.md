# Research Report: Where is Washing Machine stored in LLMs?

## Abstract
We investigated whether the concept of "washing machine" in GPT-2 Small is represented as a distinct, atomic concept (orthogonal to its parts) or as a composition of the atomic concepts "washing" and "machine". Using a synthetic dataset and linear probing of the residual stream, we found strong evidence for the **compositional hypothesis**. The representation of the "machine" token in "washing machine" remains extremely close to the generic "machine" vector (Cosine Sim: 0.99), while the deviation aligns significantly with the "washing" concept vector (Cosine Sim: 0.35).

## Methodology

### Model
- **Model:** GPT-2 Small (12 layers, 117M params)
- **Library:** `TransformerLens` for extracting residual stream activations.
- **Layer:** Final layer residual stream (`blocks.11.hook_resid_post`).

### Dataset
We constructed a synthetic dataset (`datasets/synthetic/dataset.json`) containing 240 examples to strictly control context:
1.  **Washing Machine:** "The washing machine is broken."
2.  **Other Machine:** "The sewing machine is broken.", "The time machine..."
3.  **Washing (Verb):** "I am washing the car."

### Metrics
We extracted the activation vectors ($v$) for:
-   $v_{WM}$: The ' machine' token in "washing machine".
-   $v_{M}$: The ' machine' token in other contexts (sewing, time, generic).
-   $v_{W}$: The ' washing' token in "washing [obj]" contexts.

We computed:
1.  **Stability of Machine:** Cosine similarity ($v_{WM}$, $v_{M}$).
2.  **Stability of Washing:** Cosine similarity ($v_{W\_in\_WM}$, $v_{W\_verb}$).
3.  **Compositionality:** Cosine similarity ($v_{WM} - v_{M}$, $v_{W}$).

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Sim($v_{WM}$, $v_{M}$)** | **0.9870** | The "machine" representation is almost identical in both contexts. |
| **Sim($v_{W}$, $v_{W\_context}$)** | **0.9375** | The "washing" representation is highly stable across contexts. |
| **Sim($\Delta$, $v_{W}$)** | **0.3515** | The change in the "machine" vector is significantly aligned with "washing". |

## Discussion

The extremely high similarity (0.99) between the "machine" token in "washing machine" and in other contexts indicates that the model does **not** map "washing machine" to a completely new, orthogonal direction in the latent space. It remains fundamentally a "machine".

The difference vector ($\Delta = v_{WM} - v_{M}$), which represents the specific "washing-ness" added to the machine, has a cosine similarity of 0.35 with the "washing" verb vector. In high-dimensional space (d=768), a similarity of 0.35 is highly significant (random $\approx$ 0). This suggests that the model composes the concept by adding "washing" features to the "machine" representation.

## Conclusion

Our findings support the hypothesis:
> "In large language models, individual concepts such as 'washing machine' may not be represented by distinct or orthogonal directions... instead, the model may store more atomic concepts like 'washing'..."

GPT-2 Small represents "washing machine" as a linear composition of the atomic concepts "machine" and "washing", rather than as a distinct, irreducible entity.
