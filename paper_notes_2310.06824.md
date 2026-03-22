# Paper Notes: The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets
**arXiv: 2310.06824** | Marks & Tegmark (Northeastern/MIT) | COLM 2024

**NOTE**: Despite the arxiv ID suggesting this might be the RepE paper, this is actually "The Geometry of Truth" paper by Marks & Tegmark. The RepE paper (Zou et al. 2023, "Representation Engineering: A Top-Down Approach to AI Transparency") is cited but is a different paper (arXiv:2310.01405).

## Core Thesis
LLMs linearly represent the truth or falsehood of factual statements. At sufficient scale, this representation generalizes across structurally and topically diverse datasets. Simple difference-in-mean probes identify directions that are more causally implicated in model outputs than logistic regression probes.

## How They Define/Find Linear Directions

### Localization via Patching (Section 3)
- Use **activation patching** (causal mediation analysis) to find which hidden states encode truth.
- Method: Run model on true prompt p_T and false prompt p_F. Cache residual stream activations from p_T. Then run p_F but swap in individual activations from p_T. Measure change in P(TRUE) - P(FALSE).
- Found **three groups** of causally implicated hidden states:
  - **(a)** Early layers at subject token position (encodes subject identity)
  - **(b)** Middle layers at end-of-statement punctuation token (encodes truth value -- THIS IS THE KEY ONE)
  - **(c)** Late layers at prediction position (encodes final prediction)
- Group (b) is where they extract representations for all subsequent analysis.
- For LLaMA-2-13B: layer 15 residual stream over end-of-sentence punctuation token.

### PCA Visualization (Section 4)
- Extract activations at group (b) hidden states for true/false statements.
- Center activations by subtracting mean.
- Apply PCA -- true and false statements clearly separate along top PCs.
- Linear structure emerges in **early-middle layers** and later for more complex statements.
- Alignment between datasets: axes of separation for different datasets often (but not always) align.
- **Scale matters**: LLaMA-2-70B aligns larger_than and smaller_than along common direction; 13B shows antipodal separation.

### Mass-Mean Probing (Section 5 -- KEY METHODOLOGICAL CONTRIBUTION)

**Problem with Logistic Regression (LR)**:
- LR finds maximum margin separator, not the true concept direction.
- When another feature f is represented non-orthogonally to truth direction theta_t, LR adjusts to reduce "interference" from f.
- This means LR probe direction != truth direction.

**Mass-Mean (MM) Probe**:
- theta_mm = mu_+ - mu_- (mean of true statements minus mean of false statements)
- Basic probe: p_mm(x) = sigma(theta_mm^T * x)
- IID version with covariance correction: p_iid_mm(x) = sigma(theta_mm^T * Sigma^{-1} * x)
  - Where Sigma is covariance of class-centered data
  - Equivalent to Linear Discriminant Analysis (Fisher, 1936)
- **Key insight**: Mass-mean probes are similarly accurate to LR but identify directions MORE causally implicated in model outputs.

### Contrast-Consistent Search (CCS)
- Unsupervised method from Burns et al. (2023).
- Given contrast pairs (statement, negation), finds direction along which representations are far apart.
- Used as comparison baseline.

## How They Validate

### Classification Accuracy & Generalization (Section 5)
- Train probes on one dataset, test on all others.
- Training on **statements AND their opposites** improves generalization (e.g., cities + neg_cities better than cities alone).
- **Probes generalize better for larger models**: 13B and 70B show strong cross-dataset transfer.
- Probes trained on `likely` dataset (probable vs improbable text) perform poorly on anti-correlated datasets, confirming truth representation is distinct from probability.

### Causal Intervention (Section 6 -- CRUCIAL VALIDATION)
- Shift group (b) activations along probe direction theta.
- Normalize theta so adding it to mean false representation gives same probe score as mean true representation.
- **Normalized Indirect Effect (NIE)**: measures how effectively the intervention flips model output.
  - NIE=0: intervention ineffective; NIE=1: full flip.
- **Results**: MM probe directions are highly causal (NIE ~0.85-0.97 for cities+neg_cities training on 13B), substantially outperforming LR (0.33-0.52) and CCS (0.31-0.73).
- Evaluated on **OOD inputs** (sp_en_trans), not just in-distribution.

## Models Used
- **LLaMA-2 family**: 7B, 13B, 70B
- Main results on 13B and 70B

## Datasets
Curated datasets (simple, unambiguous factual statements):
- **cities**: "The city of [city] is in [country]." (1496 rows)
- **neg_cities**: Negated versions with "not" (1496 rows)
- **sp_en_trans**: "The Spanish word '[word]' means '[English word]'." (354 rows)
- **neg_sp_en_trans**: Negated versions (354 rows)
- **larger_than**: "x is larger than y." (1980 rows, numbers 51-99)
- **smaller_than**: "x is smaller than y." (1980 rows)
- **cities_cities_conj**: Conjunctions of two city statements with "and" (1500 rows)
- **cities_cities_disj**: Disjunctions with "or" (1500 rows)

Uncurated datasets:
- **companies_true_false**: From Azaria & Mitchell (2023) (1200 rows)
- **common_claim_true_false**: From Casper et al. (2023) (4450 rows)
- **counterfact_true_false**: From Meng et al. (2022) (31960 rows)

Control dataset:
- **likely**: Nonfactual text with likely/unlikely final tokens (10000 rows) -- tests whether direction is truth or merely probability

## Code Availability
- **github.com/saprmarks/geometry-of-truth** (code, datasets, interactive dataexplorer)

## Key Techniques Adaptable for "Is a Pokemon" Probing

1. **Difference-in-means (mass-mean) probing**: Compute mean activation for pokemon-mentioning contexts minus mean activation for non-pokemon contexts. This gives a candidate "pokemon direction." This is simpler and more causally meaningful than logistic regression.

2. **Localization via patching**: Use activation patching to find which layer and token position encodes "is a pokemon" information. Likely at the last token of the entity name, in middle layers.

3. **PCA visualization**: Extract activations for pokemon vs non-pokemon statements, center, and visualize top PCs. If there's a pokemon direction, it should separate cleanly.

4. **Cross-dataset generalization**: Train probe on one set of pokemon/non-pokemon pairs, test on completely different ones. If the direction is truly "pokemon-ness" and not surface features, it should transfer.

5. **Causal validation**: Add the pokemon direction to a non-pokemon entity's representation and check if the model starts generating pokemon-like completions (types, moves, evolutions, etc.).

6. **Covariance correction**: Use p_iid_mm = sigma(theta_mm^T * Sigma^{-1} * x) for IID evaluation to account for interference from non-orthogonal features.

7. **Likely/unlikely control**: Ensure the pokemon direction isn't just "fictional entity" or "popular culture" by testing on control datasets.

## Important Findings for Our Research

- **Linear truth representations emerge with scale** -- similarly, a "pokemon" concept direction might only emerge clearly in larger models.
- **Simple difference-in-means is better than LR** for finding causally meaningful directions.
- **Training on diverse data** (statements + negations) produces more generalizable probes.
- **Representations are localized** to specific layers and token positions -- need to find the right extraction point.
- **PCA on centered activations** is a simple but powerful visualization tool.
- The paper demonstrates that even "softer" concepts (not just hard factual associations) can be linearly represented and causally validated.
