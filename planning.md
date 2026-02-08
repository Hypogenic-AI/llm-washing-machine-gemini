# Research Plan: Where is "Washing Machine" stored in LLMs?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding how Large Language Models (LLMs) represent composite concepts (like "washing machine") versus atomic concepts (like "washing" or "machine") is fundamental to the field of mechanistic interpretability. If we can determine whether concepts are stored as distinct, orthogonal vectors or as linear combinations of their parts, we can better control model generation, mitigate hallucinations, and understand "superposition." This specific case study of a concrete noun phrase ("washing machine") serves as a probe for the broader question of semantic compositionality in transformers.

### Gap in Existing Work
Most existing work focuses on abstract concepts (e.g., "truthfulness", "sentiment") or fully polysemantic neurons (e.g., *Toy Models of Superposition*). There is less work explicitly visualizing the *transition* from atomic representation to composite representation in the residual stream for everyday objects. The user's query highlights a specific ambiguity: is "washing machine" a "thing" in the model's high-dimensional space, or just a probabilistic consequence of "washing"?

### Our Novel Contribution
We will perform a targeted mechanistic analysis of the "washing machine" concept in GPT-2 Small. We will specifically test:
1.  **Vector Arithmetic:** Whether the residual stream vector for "washing machine" is well-approximated by the sum of "washing" and "machine" vectors.
2.  **Logit Lens:** What the model "thinks" the token is at intermediate layers.
3.  **Causal Intervention:** Whether "washing" acts as a necessary condition for "machine" via specific attention heads or residual updates.

### Experiment Justification
-   **Experiment 1 (Logit Lens):** Directly visualizes the model's intermediate state. If the model predicts "machine" immediately after processing "washing" (even in early layers), it suggests a strong bigram association.
-   **Experiment 2 (Residual Stream Similarity):** Quantifies the geometric relationship between the composite and its parts.
-   **Experiment 3 (Attention Pattern Analysis):** (Time permitting) Checks if the "machine" token attends back to "washing" to form the concept.

## Research Question
How is the concept "washing machine" represented in the residual stream of an LLM? Is it a distinct, semi-orthogonal direction, or is it represented as the superposition/composition of "washing" and "machine"?

## Hypothesis Decomposition
-   **H1 (Compositionality):** The representation of "washing machine" (at the second token) is largely explained by the linear combination of a "washing" vector and a generic "machine" vector.
-   **H2 (Atomic Concept):** There exists a specific "washing machine" direction that emerges in later layers that is significantly different from the sum of its parts.
-   **H3 (Bigram Probability):** The "washing" token simply upweights the "machine" token in the output logits without forming a distinct "washing machine" semantic vector in the residual stream.

## Proposed Methodology

### Approach
We will use `TransformerLens` to hook into the residual streams of `gpt2-small`. We will use the pre-gathered `washing_machine_corpus` to extract activations for:
1.  "washing machine" contexts.
2.  "washing" (not machine) contexts (e.g., "washing hands").
3.  "machine" (not washing) contexts (e.g., "time machine").

### Experimental Steps
1.  **Data Preparation:** Filter the `washing_machine_corpus` into the three categories above.
2.  **Activation Caching:** Run the model and cache residual stream vectors at the relevant token positions.
3.  **Analysis 1: Cosine Similarity:** Compute pairwise similarities between the mean vectors of the three groups.
4.  **Analysis 2: Vector Arithmetic:** Check if $||\vec{v}_{WM} - (\vec{v}_{W} + \vec{v}_{M})|| \ll ||\vec{v}_{WM}||$.
5.  **Analysis 3: Logit Lens:** Project the residual stream of "washing" (in WM context) to the vocabulary. Check the rank of " machine".

### Baselines
-   **Random Pairs:** Compare "washing machine" compositionality to random adjective-noun pairs (e.g., "blue machine", "green washing").
-   **Atomic Concepts:** Compare to single-token concepts (e.g., "computer") to see how "tight" a known atomic concept's cluster is.

### Evaluation Metrics
-   **Cosine Similarity**: Between composite and component vectors.
-   **Logit Rank**: Rank of "machine" in the logit lens output at the "washing" position.
-   **Explained Variance**: How much of the "washing machine" vector variance is explained by the "washing" and "machine" principal components.

## Timeline
-   **Phase 2 (Setup):** 10 min. Install `TransformerLens`, load data.
-   **Phase 3 (Impl):** 30 min. Write analysis scripts.
-   **Phase 4 (Exp):** 30 min. Run experiments on GPU.
-   **Phase 5 (Analysis):** 20 min. Generate plots and stats.
-   **Phase 6 (Docs):** 20 min. Write Report.

## Success Criteria
-   Clear visualization of the "washing machine" trajectory in the residual stream.
-   Quantitative evidence supporting either the compositionality or atomic hypothesis.
