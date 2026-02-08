# Washing Machine in LLMs: A Mechanistic Analysis

## Overview
This project investigates the representation of the composite concept "washing machine" in GPT-2 Small. We tested whether the model represents this concept as a unique, orthogonal vector or as a linear sum of "washing" and "machine".

## Key Findings
-   **Compositionality:** "Washing machine" is represented as a linear superposition of its parts, not a unique direction.
-   **Formula:** $v_{WM} \approx 0.7 \cdot v_{Machine} + 0.3 \cdot v_{Washing}$ (in final layers).
-   **Dynamics:** The representation starts as pure "machine" (Layer 0) and progressively integrates "washing" context (Layers 1-11).
-   **Variance:** >90% of the variance of the "washing machine" vector is explained by the plane spanned by "washing" and "machine".

## Reproducing Results

1.  **Environment Setup:**
    ```bash
    uv pip install -r requirements.txt
    ```

2.  **Run Analysis:**
    ```bash
    python src/analysis_main.py      # Basic Similarity & Logit Lens
    python src/analysis_refined.py   # Layer-wise Centered Similarity
    python src/analysis_orthogonal.py # Decompostion (Main Result)
    ```

3.  **View Results:**
    -   Report: `REPORT.md`
    -   Figures: `results/figures/`
    -   Raw Data: `results/*.json`

## File Structure
-   `src/`: Python analysis scripts.
-   `results/`: Generated plots and JSON data.
-   `datasets/`: Pre-downloaded corpus (used for validation).
-   `code/`: External libraries (TransformerLens).

## Research Question
"Where is 'washing machine' stored in LLMs? Or is just 'washing' stored and then it's assumed that machine becomes more likely?"

**Answer:** It is stored as a modification of the "machine" vector. "Washing" context is added to the "machine" vector, and "washing" strongly predicts "machine" (Logit Lens), supporting the user's intuition.