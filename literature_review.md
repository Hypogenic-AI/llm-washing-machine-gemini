# Literature Review: Concept Representation and Compositionality in LLMs

## Research Area Overview
The research focuses on how Large Language Models (LLMs) represent concepts, specifically investigating whether composite concepts like "washing machine" are represented as distinct orthogonal directions (atomic) or as a composition of their constituent parts ("washing" + "machine"). This touches upon **mechanistic interpretability**, **linear representation hypothesis**, and **superposition**.

## Key Papers

### 1. Toy Models of Superposition (Elhage et al., 2022)
- **Source:** arXiv:2209.10652
- **Key Contribution:** Demonstrates that neural networks can store more features than they have dimensions by placing features in "superposition" (non-orthogonal directions).
- **Relevance:** This is crucial for the "washing machine" hypothesis. If "washing machine" is a rare concept, it might be stored in superposition with other features, or it might not be a "feature" at all but a dynamic composition of "washing" and "machine" activations. The paper introduces "polysemantic neurons" which we should look for.

### 2. Funny or Persuasive, but Not Both (Labroo et al., 2026)
- **Source:** arXiv:2601.18483
- **Key Contribution:** Evaluates fine-grained control over multiple concepts. Finds that LLMs struggle with checking/controlling two distinct concepts simultaneously (compositionality gap).
- **Relevance:** Highlights limitations in how models handle multiple active concepts. If "washing machine" requires activating "washing" and "machine" simultaneously, this paper suggests there might be interference or a specific "binding" mechanism needed.

### 3. Exploring Multilingual Concepts of Human Values (Xu et al., 2024)
- **Source:** arXiv:2402.18120
- **Key Contribution:** Empirically confirms that abstract concepts (human values) are represented as linear directions in the residual stream, consistent across languages.
- **Relevance:** Validates the "Linear Representation Hypothesis". It provides a methodology (finding a vector direction for a concept) that we can apply to "washing machine". We can try to find a "washing machine vector" and check its cosine similarity to "washing" and "machine" vectors.

### 4. Tackling Polysemanticity with Neuron Embeddings (Foote, 2024)
- **Source:** arXiv:2411.08166
- **Key Contribution:** Proposes "neuron embeddings" to decompose the behavior of polysemantic neurons into distinct clusters.
- **Relevance:** If we find neurons that respond to "washing machine", they might also respond to other things (polysemanticity). This method offers a way to disentangle them.

## Common Methodologies
- **Linear Probing:** Training a linear classifier on the residual stream to detect the presence of a feature.
- **Activation Steering:** Adding a concept vector to the residual stream to induce a behavior.
- **Sparse Autoencoders (SAEs):** (From *Toy Models*) Used to disentangle superposition, though computationally expensive to train from scratch.
- **Ablation/Erasure:** Removing a direction to see if the concept disappears.

## Recommendations for Experiment
Based on the literature:
1.  **Method:** Use **Linear Probing** (via `TransformerLens`) to find the "washing machine" direction.
2.  **Analysis:** Compute the cosine similarity between the "washing machine" vector and the "washing" and "machine" vectors.
    -   *Hypothesis test:* If $v_{washing\_machine} \approx v_{washing} + v_{machine}$, it supports compositionality. If $v_{washing\_machine}$ is orthogonal to both, it supports atomic representation.
3.  **Dataset:** The constructed `washing_machine_corpus` is suitable for finding activations.
4.  **Model:** GPT-2 Small (supported by `TransformerLens` and `polysemanticity` paper) is a good starting point due to tractability.
