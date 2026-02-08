# Paper Outline: Compositional Representation of "Washing Machine" in LLMs

## 1. Title
- **Proposed Title:** "Washing Machine" in the Residual Stream: Evidence for Compositional Concept Representation in Large Language Models
- **Alternative:** Atomic or Compositional? Dissecting the Representation of Noun Phrases in GPT-2

## 2. Abstract
- **Context:** LLMs represent concepts as vectors, but it's unclear if composite concepts (Noun Phrases) are atomic or compositional.
- **Gap:** Hypothesis that models store "atomic" concepts (washing, machine) vs. distinct vectors for every combination.
- **Approach:** Linear probing of GPT-2 Small residual streams using synthetic datasets with controlled contexts.
- **Results:** "Machine" vector is stable (sim 0.99) across contexts. The "washing" modifier adds a specific direction (sim 0.35 with "washing" verb).
- **Conclusion:** "Washing Machine" is represented compositionally ($v_{WM} \approx v_{M} + \Delta_{washing}$), not as a unique orthogonal concept.

## 3. Introduction
- **Hook:** How do LLMs represent the world? Linearity hypothesis (Elhage, Xu).
- **Background:** Superposition, polysemanticity. The question of "binding" attributes to objects.
- **Problem:** Does "washing machine" get its own neuron/direction, or is it just "machine" + "washing"?
- **Contribution:**
    - Empirical analysis of noun phrase representation in GPT-2.
    - Evidence against "atomic" unique representation for common compounds.
    - Quantification of the "compositional delta".

## 4. Related Work
- **Mechanistic Interpretability:** Anthropic's Toy Models (superposition), TransformerLens.
- **Concept Representation:** Xu et al. (2024) on linear value concepts.
- **Attribute Binding:** Labroo et al. (2026) on multi-concept control.
- **Polysemanticity:** Foote (2024).

## 5. Methodology
- **Model:** GPT-2 Small (12 layers).
- **Dataset:** Synthetic dataset generation (controlled templates) to isolate the "washing" and "machine" tokens.
    - Contexts: "washing machine", "sewing machine", "generic machine", "washing (verb)".
- **Metric:** Cosine Similarity of residual stream activations at Layer 11 (pre-unembedding).
    - Stability metrics: $Sim(v_{ctx1}, v_{ctx2})$.
    - Compositionality metric: $Sim(v_{compound} - v_{head}, v_{modifier})$.

## 6. Experiments & Results
- **Setup:** 240 examples, rigorous control.
- **Result 1: Stability of the Head Noun.**
    - $Sim(v_{WM\_machine}, v_{Other\_machine}) = 0.99$.
    - Finding: The "machine" token stays in the "machine" subspace.
- **Result 2: Stability of the Modifier.**
    - $Sim(v_{washing\_verb}, v_{washing\_adj}) = 0.94$.
    - Finding: "Washing" concept is stable across POS tags.
- **Result 3: The Compositional Delta.**
    - $\Delta = v_{WM} - v_{Other}$.
    - $Sim(\Delta, v_{washing}) = 0.35$.
    - Finding: The modification is aligned with the modifier, but not perfectly (0.35 vs 1.0). Suggests non-linear or feature-specific interactions, but clearly not orthogonal (random).

## 7. Discussion
- **Interpretation:** The model uses a "base + modifier" strategy. It doesn't waste capacity on a unique "washing machine" vector.
- **Implications for Safety/Control:** Modifying the "washing" vector could broadly affect all "washing" related concepts.
- **Limitations:** Only GPT-2 Small. Only one phrase type.

## 8. Conclusion
- Summary: Strong evidence for compositional representation.
- Future Work: Test on larger models, more abstract compounds (e.g., "red herring" - idiomatic).
