# Literature Review: Do Any Non-Pokemon Have Non-Trivial Pokemon Quality?

## Research Area Overview

This research sits at the intersection of **mechanistic interpretability** and **concept representation** in large language models. The central question is whether the concept "is a Pokemon" is encoded as a linear direction in an LLM's representation space, and if so, which non-Pokemon entities score highest along this direction.

The theoretical foundation rests on the **Linear Representation Hypothesis (LRH)**: high-level concepts are represented linearly as directions in LLM representation spaces. Recent work has formalized this hypothesis, provided tools for finding concept directions, and demonstrated that these directions are causally meaningful for model behavior.

## Key Papers

### Paper 1: The Linear Representation Hypothesis and the Geometry of LLMs
- **Authors**: Park, Choe, Veitch (U Chicago)
- **Year**: 2024 (ICML)
- **arXiv**: 2311.03658
- **Key Contribution**: Formal unification of three notions of linear representation (subspace, measurement, intervention) via a causal inner product.
- **Methodology**: Define concept directions using counterfactual word pairs in unembedding space. Estimate direction as normalized mean of pair differences. Validate via alignment tests, orthogonality heatmaps, probe accuracy, and intervention experiments.
- **Datasets Used**: 27 concepts from BATS 3.0 + custom language/frequency pairs. Requires single-token counterfactual pairs.
- **Results**: 26/27 concepts showed clear linear structure. Causal inner product (Cov(γ)^{-1}) correctly makes separable concepts orthogonal.
- **Code Available**: Yes - github.com/KihoPark/linear_rep_geometry
- **Relevance**: Provides the theoretical framework and practical methodology for constructing a "pokemon direction" from counterfactual pairs like (Pikachu, mouse).

### Paper 2: The Geometry of Truth (Marks & Tegmark)
- **Authors**: Marks, Tegmark (Northeastern/MIT)
- **Year**: 2024 (COLM)
- **arXiv**: 2310.06824
- **Key Contribution**: Shows LLMs linearly represent truth/falsehood; difference-in-means (mass-mean) probes are more causally meaningful than logistic regression probes.
- **Methodology**: (1) Localize truth representations via activation patching to middle layers at statement-ending tokens. (2) PCA visualization shows clear separation. (3) Mass-mean probe: θ_mm = μ_+ - μ_-. (4) Causal validation via activation intervention.
- **Datasets Used**: Curated true/false statements (cities, translations, comparisons, conjunctions). 1K-32K rows per dataset.
- **Results**: MM probe NIE ~0.85-0.97 (much higher causal implication than LR at 0.33-0.52). Linear truth emerges with scale.
- **Code Available**: Yes - github.com/saprmarks/geometry-of-truth
- **Relevance**: The MM probe methodology is directly applicable. The activation patching approach helps localize where "pokemon-ness" is encoded. The cross-dataset generalization paradigm validates that a direction captures the intended concept.

### Paper 3: Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Meng, Bau, Andonian, Belinkov
- **Year**: 2022 (NeurIPS)
- **arXiv**: 2202.05262
- **Key Contribution**: Factual associations are stored in middle-layer MLP modules and can be edited with rank-one updates.
- **Methodology**: Causal tracing identifies decisive neurons. ROME modifies MLP weights to update facts.
- **Datasets Used**: CounterFact dataset (factual recall).
- **Results**: Middle-layer MLPs mediate factual predictions; last subject token is critical.
- **Code Available**: Yes - github.com/kmeng01/rome
- **Relevance**: If "is a pokemon" is a factual association, it may be stored in MLPs at the entity's last token position. Causal tracing can locate it.

### Paper 4: Representation Engineering (Zou et al.)
- **Authors**: Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika, Dombrowski, Goel, Li, Byun, Wang, Mallen, Basart, Koyejo, Song, Fredrikson, Kolter, Hendrycks
- **Year**: 2023
- **arXiv**: 2310.01405
- **Key Contribution**: Top-down approach to AI transparency using population-level representations. Linear Artificial Tomography (LAT) uses contrastive stimuli to find concept directions.
- **Methodology**: Present model with contrasting prompts (e.g., "honest" vs "dishonest" personas), extract activations, compute difference to get concept direction. Can read (probe) and control (steer) model behavior.
- **Code Available**: Yes - github.com/andyzoujm/representation-engineering
- **Relevance**: The contrastive stimulus design is adaptable for "pokemon vs non-pokemon" prompts. RepE's reading vectors could detect pokemon-ness in arbitrary inputs.

### Paper 5: Steering LLMs with Activation Engineering (Turner et al.)
- **Authors**: Turner, Thacker, Krasheninnikov, Olsson, et al.
- **Year**: 2023
- **arXiv**: 2308.10248
- **Key Contribution**: Activation Addition (ActAdd) technique for steering model behavior by adding contrastive activation vectors.
- **Methodology**: Compute steering vector from contrasting prompt pairs. Add scaled vector to residual stream during inference.
- **Relevance**: After finding a "pokemon direction," ActAdd could validate it by adding the direction to non-pokemon entity representations and observing pokemon-like completions.

### Paper 6: Sparse Autoencoders Find Highly Interpretable Features in Language Models
- **Authors**: Cunningham, Ewart, Riggs, Huben, Sharkey
- **Year**: 2023
- **arXiv**: 2309.08600
- **Key Contribution**: SAEs decompose model activations into interpretable features, some corresponding to specific entity types or knowledge categories.
- **Relevance**: SAE features might reveal a specific "pokemon" feature or related features for fictional entities, game characters, etc. Could provide complementary evidence to linear probing.

### Paper 7: Not All Language Model Features Are Linear
- **Authors**: Engels, Liao, Michaud, Gurnee, Tegmark
- **Year**: 2024
- **arXiv**: 2405.14860
- **Key Contribution**: Some concepts are represented as multi-dimensional features (e.g., days of the week on a circle). Challenges pure 1D linear representation assumption.
- **Relevance**: Important caveat - "pokemon-ness" might not be purely 1-dimensional. Pokemon types, generations, or evolutionary stages might form a multi-dimensional subspace.

## Common Methodologies

### Finding Concept Directions
1. **Counterfactual pair differences** (Park et al.): Average γ(pokemon) - γ(analog) across pairs.
2. **Difference-in-means / Mass-mean** (Marks & Tegmark): μ_pokemon - μ_non_pokemon over activations.
3. **Contrastive stimuli** (Zou et al.): Contrasting prompts → activation differences → PCA/mean.
4. **Linear probes** (logistic regression): Train classifier on activations, extract weight vector.
5. **Sparse autoencoders**: Decompose activations, find pokemon-specific features.

### Validating Concept Directions
1. **Classification accuracy**: Does projecting onto direction separate pokemon from non-pokemon?
2. **Cross-dataset generalization**: Does probe trained on one set of entities transfer?
3. **Causal intervention**: Does adding direction to activations cause pokemon-like outputs?
4. **PCA visualization**: Do pokemon/non-pokemon cluster along direction?
5. **Normalized Indirect Effect (NIE)**: Quantifies causal impact of intervention.

## Standard Baselines
- **Logistic regression probe**: Standard supervised linear classifier on activations
- **Mass-mean probe**: Unsupervised difference-in-means direction
- **CCS probe**: Unsupervised contrast-consistent search (Burns et al.)
- **Random direction**: Control baseline for all probe metrics

## Evaluation Metrics
- **Probe accuracy**: Classification accuracy on held-out pokemon/non-pokemon examples
- **Cross-dataset transfer accuracy**: Accuracy on out-of-distribution entities
- **NIE (Normalized Indirect Effect)**: Causal impact of adding direction to activations
- **Cosine similarity**: Between concept direction and individual entity directions
- **Projection magnitude**: How far each entity projects onto the pokemon direction

## Gaps and Opportunities

1. **No prior work on "pokemon-ness" as a concept direction** - this is entirely novel
2. **Entity-type probing is understudied** compared to truth/sentiment/language probing
3. **Fictional entity representation** is an open question - how do LLMs distinguish fictional categories?
4. **Cross-franchise similarity** (pokemon vs digimon vs other game creatures) is unexplored
5. **Non-binary concept formulation**: Pokemon-ness might be better modeled as a continuous "quality" rather than binary

## Recommendations for Our Experiment

### Recommended Approach (Priority Order)
1. **Mass-mean probe** on residual stream activations (simplest, most causally meaningful)
2. **Counterfactual pair direction** in unembedding space (theoretically principled)
3. **Contrastive stimuli** with pokemon/non-pokemon prompt pairs (RepE-style)

### Recommended Models
- **GPT-2 Medium/Large** via TransformerLens (fast iteration, full access to internals)
- **LLaMA-2-7B** or **Pythia-6.9B** for larger-scale validation
- Larger models expected to have cleaner concept representations

### Recommended Datasets
- Our curated **pokemon_names.json** (1026 Pokemon) as positive class
- Our curated **non_pokemon_entities.json** (177 entities across 7 categories) as negative class
- Our **contrastive_pairs.json** (49 pokemon/analog pairs) for counterfactual direction estimation
- Our **prompt_templates.json** for activation extraction

### Recommended Metrics
- Probe accuracy on held-out pokemon/non-pokemon sets
- Top-k non-pokemon entities by projection onto pokemon direction (the main research question!)
- PCA visualization of pokemon vs non-pokemon activations
- Causal validation: does adding pokemon direction cause pokemon-like completions?

### Methodological Considerations
- Extract activations from **middle layers** at the **last token of the entity name** (following ROME and Geometry of Truth findings)
- Use **causal inner product** (Cov^{-1}) rather than Euclidean for proper geometric analysis
- Test multiple layers and token positions to find optimal extraction point
- Include **control experiments**: random directions, "fictional" direction, "animal" direction
- Consider that pokemon names are single tokens in some models but multi-token in others
