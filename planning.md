# Research Plan: Do Any Non-Pokemon Have Non-Trivial Pokemon Quality?

## Motivation & Novelty Assessment

### Why This Research Matters
The Linear Representation Hypothesis (LRH) suggests high-level concepts are encoded as linear directions in LLM representation spaces. While this has been demonstrated for concepts like truth/falsehood, sentiment, and language — no one has investigated how *categorical membership in a specific fictional franchise* (Pokemon) is represented. Understanding this reveals how LLMs organize knowledge about fictional entities and whether such categories have sharp or fuzzy boundaries in representation space.

### Gap in Existing Work
Prior work (Marks & Tegmark 2024, Park et al. 2024) demonstrates linear probing for truth, language, and factual concepts. Representation Engineering (Zou et al. 2023) shows contrastive stimuli can extract concept directions for abstract traits. However: (1) No work probes "franchise membership" as a concept, (2) No work examines whether fictional entity categories have fuzzy boundaries in representation space, (3) The question "what non-Pokemon is most Pokemon-like?" is entirely novel.

### Our Novel Contribution
We extract a "Pokemon direction" from LLM activations using mass-mean probing, validate it via classification accuracy, and then project non-Pokemon entities onto this direction to discover which real animals, mythical creatures, game characters, and other entities the model considers "most Pokemon-like." This reveals the geometry of fictional franchise knowledge in LLMs.

### Experiment Justification
- **Experiment 1 (Direction Extraction)**: Find the Pokemon direction via mass-mean probe across layers — needed to identify WHERE and HOW pokemon-ness is encoded.
- **Experiment 2 (Classification Validation)**: Test probe accuracy on held-out entities — validates the direction actually captures pokemon-ness, not noise.
- **Experiment 3 (Non-Pokemon Ranking)**: Project non-Pokemon entities onto the direction — the core research question.
- **Experiment 4 (Control Directions)**: Compare against random and "fictional" directions — ensures we found pokemon-ness specifically, not just "fictional creature-ness."

## Research Question
How is the concept "X is a Pokemon" represented in an LLM, and which non-Pokemon entities score highest along this representation direction?

## Hypothesis Decomposition
1. "Is a Pokemon" is encoded as a linear direction in the residual stream (testable via probe accuracy).
2. This direction is concentrated in middle-to-late layers (testable via layer-wise accuracy).
3. Non-Pokemon entities vary in their projection onto this direction — some will have non-trivial "Pokemon quality" (testable via ranking).
4. Entities resembling Pokemon (Digimon, mythical creatures, Pokemon-like names) will score higher than common objects (testable via category-level comparison).

## Proposed Methodology

### Approach
Use TransformerLens to load a model with full access to internals, extract residual stream activations at each layer for Pokemon and non-Pokemon entities in a standardized prompt, compute the mass-mean direction, and project all entities onto it.

### Model Choice
GPT-2-XL (1.5B params) — large enough for good representations, small enough for fast iteration, and well-supported by TransformerLens. If time permits, validate on Pythia-6.9B.

### Experimental Steps
1. Load model via TransformerLens, extract activations for all entities using "{entity} is a" prompt
2. Split Pokemon into train (80%) and test (20%) sets; split non-Pokemon similarly
3. Compute mass-mean direction at each layer: θ_l = mean(act_pokemon_train) - mean(act_non_pokemon_train)
4. Evaluate classification accuracy per layer on test set
5. Select best layer, project ALL non-Pokemon entities onto θ
6. Rank non-Pokemon entities by Pokemon quality score
7. Visualize with PCA and bar charts
8. Control: compute random direction, compare accuracy

### Baselines
- Random direction (null baseline)
- Logistic regression probe (supervised baseline)

### Evaluation Metrics
- Classification accuracy (train/test) per layer
- AUC-ROC for the probe
- Top-k non-Pokemon entities by Pokemon quality
- Category-level mean Pokemon quality scores

### Statistical Analysis Plan
- Bootstrap confidence intervals for accuracy
- Compare category-level Pokemon quality with Kruskal-Wallis test
- Effect size via Cohen's d between Pokemon and non-Pokemon projections

## Expected Outcomes
- Probe accuracy > 90% at optimal layer (supporting linear encoding)
- Digimon and mythical creatures will have highest Pokemon quality among non-Pokemon
- Common objects will have lowest Pokemon quality
- Pokemon-like fake names may or may not score high (depends on morphological vs. semantic encoding)

## Timeline
- Phase 1 (Planning): Done
- Phase 2 (Setup): 10 min
- Phase 3 (Implementation): 45 min
- Phase 4 (Experiments): 30 min
- Phase 5 (Analysis): 20 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- Multi-token entity names may dilute signal → use last token position
- Some entities may be rare/unknown to model → check with tokenization
- Direction may capture "fictional" rather than "Pokemon" → control with fictional direction

## Success Criteria
- Probe accuracy > 85% on held-out test set
- Clear separation in PCA visualization
- Interpretable ranking of non-Pokemon entities
- Control direction performs significantly worse
