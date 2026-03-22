# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Do any non-pokemon have non-trivial pokemon quality?" — investigating whether the concept "is a Pokemon" is encoded as a linear direction in LLM representation spaces.

## Papers
Total papers downloaded: 15

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Linear Representation Hypothesis | Park, Choe, Veitch | 2024 | papers/2311.03658_*.pdf | Core theory: causal inner product, counterfactual pairs |
| Geometry of Truth | Marks, Tegmark | 2024 | papers/2310.06824_*.pdf | Mass-mean probes, activation patching, causal validation |
| Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | Contrastive stimuli, reading/control vectors |
| ROME | Meng et al. | 2022 | papers/2202.05262_*.pdf | Factual knowledge in MLPs, causal tracing |
| Activation Engineering | Turner et al. | 2023 | papers/2308.10248_*.pdf | ActAdd steering technique |
| SAE Features | Cunningham et al. | 2023 | papers/2309.08600_*.pdf | Sparse autoencoder interpretability |
| Not All Features Linear | Engels et al. | 2024 | papers/2405.14860_*.pdf | Multi-dimensional features caveat |
| Origins of Linear Reps | Various | 2024 | papers/2403.03867_*.pdf | Why linear representations emerge |
| Steering with Conceptors | Various | 2024 | papers/2410.16314_*.pdf | Advanced steering with ellipsoids |
| LLM Cognition Linear Reps | Various | 2024 | papers/2405.16964_*.pdf | Cognition to expression |
| Semantic Subspace Probing | Various | 2023 | papers/2310.11923_*.pdf | Structural probing methods |
| Embedded NER Probing | Various | 2024 | papers/2403.11747_*.pdf | Entity recognition via probes |
| RepE Survey | Various | 2025 | papers/2502.17601_*.pdf | Comprehensive RepE taxonomy |
| MechInterp Safety Review | Various | 2024 | papers/2404.14082_*.pdf | Safety applications survey |
| MechInterp ICLR 2026 | Various | 2025 | papers/2510.02917_*.pdf | Recent advances |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets created: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Pokemon Names | GitHub (lgreski) | 1026 names | Positive class | datasets/pokemon_names.json | All Gen 1-9 |
| Pokemon Full Stats | GitHub (lgreski) | 1026 rows | Reference | datasets/pokemon_full.csv | Full stats CSV |
| Non-Pokemon Entities | Curated | 177 entities, 7 categories | Negative class | datasets/non_pokemon_entities.json | Animals, mythical, digimon, etc. |
| Contrastive Pairs | Curated | 49 pairs, 3 categories | Direction estimation | datasets/contrastive_pairs.json | Pokemon/analog pairs |
| Prompt Templates | Designed | 4 template types | Activation extraction | datasets/prompt_templates.json | Classification, context, contrastive |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability library | code/TransformerLens/ | Core tool for activation extraction |
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Mass-mean probes, patching, interventions | code/geometry-of-truth/ | Most directly applicable codebase |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE official implementation | code/representation-engineering/ | Contrastive stimuli approach |
| repeng | github.com/vgel/repeng | Lightweight control vector training | code/repeng/ | Quick prototyping tool |
| linear_rep_geometry | github.com/KihoPark/linear_rep_geometry | Causal inner product, concept directions | code/linear_rep_geometry/ | Counterfactual pair methodology |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used web search for papers on linear representation hypothesis, concept directions, probing, steering vectors, and mechanistic interpretability
2. Focused on arxiv papers from 2022-2025 with code availability
3. Searched specifically for any prior work combining Pokemon with interpretability (none found - this is novel)
4. Identified code repos from paper references and GitHub search

### Selection Criteria
- Papers that formalize or demonstrate linear concept representations in LLMs
- Papers with code available for reproducibility
- Papers using probing/intervention methodology applicable to our binary concept ("is a pokemon")
- Prioritized recent work (2023-2025) that builds on established techniques

### Challenges Encountered
- No prior work exists on "pokemon-ness" as a concept direction - this is entirely novel research
- The paper-finder service was not fully operational; relied on web search + direct arxiv downloads
- Some confusion in paper identity (2310.06824 is "Geometry of Truth" not "RepE")

### Gaps and Workarounds
- **No existing pokemon/non-pokemon probing dataset** → Created custom datasets with 1026 pokemon, 177 non-pokemon entities across 7 categories, and 49 contrastive pairs
- **No template for entity-type probing** → Designed prompt templates for activation extraction
- **Multi-token entity names** → May need to handle tokenization carefully; some pokemon names are multi-token

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Pokemon names** (1026) as positive class
- **Non-pokemon entities** (177 across 7 categories) as negative class
- **Contrastive pairs** (49) for counterfactual direction estimation
- Use train/test split to validate generalization

### 2. Primary Method: Mass-Mean Probing
Based on literature, the recommended approach is:
1. Use TransformerLens to load GPT-2 Medium or Pythia
2. Extract residual stream activations at multiple layers for "{entity} is a" prompts
3. Compute mass-mean direction: θ = mean(act_pokemon) - mean(act_non_pokemon)
4. Project all entities onto θ to find "pokemon quality" scores
5. Rank non-pokemon entities by projection magnitude → answer the research question

### 3. Baseline Methods
- Logistic regression probe (comparison)
- Random direction (null baseline)
- "Fictional entity" direction (control - ensure we're finding pokemon-ness, not fiction-ness)
- "Animal" direction (control - ensure we're not just finding creature-ness)

### 4. Evaluation Metrics
- **Classification accuracy** on held-out pokemon/non-pokemon
- **Top-k ranking** of non-pokemon entities by pokemon quality (main result)
- **PCA visualization** of activation clusters
- **Causal validation**: add pokemon direction → observe pokemon-like completions

### 5. Code to Adapt/Reuse
- `code/geometry-of-truth/probes.py` → MMProbe class for mass-mean probing
- `code/geometry-of-truth/generate_acts.py` → activation extraction pipeline
- `code/TransformerLens/` → model loading and hook-based activation caching
- `code/linear_rep_geometry/` → causal inner product computation
