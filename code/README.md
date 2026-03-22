# Cloned Repositories

## Repo 1: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Core library for mechanistic interpretability of GPT-style models. Provides hooks into all internal activations (residual stream, attention, MLP outputs).
- **Location**: `code/TransformerLens/`
- **Key files**: `transformer_lens/` (library), `Main_Demo.ipynb` (tutorial), `demos/` (examples)
- **How to use for our research**:
  - Load GPT-2/Pythia/LLaMA models
  - Extract residual stream activations at any layer for pokemon/non-pokemon inputs
  - Cache activations for probing experiments
  - Perform activation patching to localize where pokemon-ness is encoded
- **Install**: `pip install transformer-lens`

## Repo 2: geometry-of-truth
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Purpose**: Implementation of mass-mean probes, activation patching, PCA visualization, and causal interventions for binary concept directions.
- **Location**: `code/geometry-of-truth/`
- **Key files**:
  - `probes.py` - LRProbe (logistic regression), MMProbe (mass-mean), CCSProbe (contrast-consistent search)
  - `generate_acts.py` - Extract and cache activations from LLMs
  - `patching.py` - Activation patching for localization
  - `interventions.py` - Causal intervention experiments
  - `datasets/` - True/false statement datasets
- **How to use for our research**:
  - Adapt `generate_acts.py` to extract activations for pokemon/non-pokemon statements
  - Use `MMProbe.from_data()` to find the pokemon direction via difference-in-means
  - Use `patching.py` to localize which layers encode pokemon-ness
  - Use `interventions.py` to causally validate the pokemon direction
- **Dependencies**: PyTorch, transformers

## Repo 3: representation-engineering
- **URL**: https://github.com/andyzoujm/representation-engineering
- **Purpose**: Official implementation of Representation Engineering (RepE) by Zou et al. Includes RepControl and RepReading.
- **Location**: `code/representation-engineering/`
- **Key files**: Examples for reading and controlling model representations using contrastive stimuli.
- **How to use**: Adapt contrastive stimuli templates for pokemon vs non-pokemon concepts.

## Repo 4: repeng
- **URL**: https://github.com/vgel/repeng
- **Purpose**: Lightweight library for generating control vectors via representation engineering. Train a concept vector in <60 seconds.
- **Location**: `code/repeng/`
- **Key files**: `repeng/` (library), `notebooks/` (examples)
- **How to use**: Can directly train a "pokemon" control vector using paired contrastive prompts.
- **Install**: `pip install repeng`

## Repo 5: linear_rep_geometry
- **URL**: https://github.com/KihoPark/linear_rep_geometry
- **Purpose**: Official code for "The Linear Representation Hypothesis" paper. Implements causal inner product and concept direction estimation from counterfactual pairs.
- **Location**: `code/linear_rep_geometry/`
- **Key files**:
  - `word_pairs/` - Counterfactual word pairs for 27 concepts
  - `store_matrices.py` - Extract unembedding matrices from LLaMA-2
  - `1_subspace.ipynb` through `5_sanity_check.ipynb` - Experiment notebooks
- **How to use**: Adapt the counterfactual pair methodology for pokemon/animal pairs. Use causal inner product for proper geometric analysis.
- **Requires**: GPU access for LLaMA-2 inference
