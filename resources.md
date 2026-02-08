# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Washing Machine in LLMs" research project.

## Papers
Total papers downloaded: 4

| Title | Filename | Key Relevance |
|-------|----------|---------------|
| Toy Models of Superposition | `2209.10652v1_Toy_Models_of_Superposition.pdf` | Superposition & Polysemanticity |
| Funny or Persuasive... | `2601.18483v1_Funny_or_Persuasive...pdf` | Compositionality Limits |
| Exploring Multilingual Concepts... | `2402.18120v3_Exploring_Multilingual...pdf` | Linear Representation Validation |
| Tackling Polysemanticity... | `2411.08166v1_Polysemanticity.pdf` | Disentangling Neurons |

See `papers/README.md` for details.

## Datasets
Total datasets: 1

| Name | Source | Size | Location |
|------|--------|------|----------|
| washing_machine_corpus | WikiText-2 | ~350 samples | `datasets/washing_machine_corpus` |

See `datasets/README.md` for details.

## Code Repositories
Total repositories: 2

| Name | Purpose | Location |
|------|---------|----------|
| TransformerLens | Mechanistic Interpretability Library | `code/TransformerLens` |
| concept-erasure | Concept Erasure / Linear Probing | `code/concept-erasure` |

See `code/README.md` for details.

## Resource Gathering Notes
- **Search Strategy:** Focused on "linear representation", "superposition", and "compositionality".
- **Challenges:** arXiv API fuzziness required robust filtering.
- **Selection:** Prioritized foundational work (Anthropic) and recent relevant empirical studies (2024-2026).

## Recommendations for Experiment Design
1.  **Primary Dataset:** `datasets/washing_machine_corpus` for probing.
2.  **Baseline Methods:** Linear Probing (using `TransformerLens` or `concept-erasure`).
3.  **Metrics:** Cosine similarity between $v_{composite}$ and $v_{parts}$. Accuracy of probe on held-out set.
4.  **Code to Reuse:** `TransformerLens` for hook points and caching activations.

## Generated Resources

### Code
-   `src/analysis_main.py`: Basic cosine similarity analysis and Logit Lens investigation.
-   `src/analysis_refined.py`: Layer-wise centered cosine similarity analysis.
-   `src/analysis_orthogonal.py`: Orthogonal decomposition (OLS regression) of the residual stream vectors.
-   `src/check_data.py`: Utility to inspect the provided corpus.

### Results
-   **Report:** `REPORT.md` (Full analysis)
-   **Data:**
    -   `results/experiment_results.json`: Raw similarities and logit probabilities.
    -   `results/refined_results.json`: Layer-wise centered similarities.
    -   `results/orthogonal_results.json`: Regression coefficients ($\alpha, \beta$) and $R^2$ scores per layer.
-   **Figures:**
    -   `results/figures/similarity_plot.png`
    -   `results/figures/layerwise_similarity.png`
    -   `results/figures/orthogonal_analysis.png`
