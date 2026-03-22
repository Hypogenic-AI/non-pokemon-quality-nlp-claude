# Do Any Non-Pokemon Have Non-Trivial Pokemon Quality?

Investigating how the concept "X is a Pokemon" is represented in large language models, and which non-Pokemon entities score highest along this representation direction.

## Key Findings

- **Pokemon-ness is linearly encoded** in GPT-2-XL's residual stream (88.4% classification accuracy, AUC=0.934)
- **The encoding peaks in middle layers** (10-32 of 48), consistent with prior work on concept representations
- **Top non-Pokemon by "Pokemon quality"**: Crystalix (invented name), Cactuar (FF), Frostbeak (invented), Koopa (Mario), Thunderix (invented), Kirby, Goomba, Garurumon (Digimon)
- **13 non-Pokemon entities** were classified as actual Pokemon by the probe
- **Category ordering** (most to least Pokemon-like): Pokemon-like names > game creatures > Digimon > mythical creatures > real animals > fictional characters > common objects (p=1.03×10⁻¹⁰)

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers transformer-lens numpy pandas matplotlib scikit-learn scipy seaborn tqdm einops jaxtyping

# Run experiment (requires GPU)
python src/experiment.py
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and rationale
├── src/experiment.py      # Main experiment script
├── datasets/              # Pokemon names, non-Pokemon entities, contrastive pairs
├── results/
│   ├── experiment_results.json  # Full numerical results
│   └── plots/                   # All visualizations
├── papers/                # Downloaded reference papers
├── code/                  # Cloned baseline repositories
└── literature_review.md   # Literature review
```

## Method

Uses the **mass-mean probing** technique (Marks & Tegmark, 2024) on TransformerLens. Extracts residual stream activations from GPT-2-XL for 160 Pokemon and 122 non-Pokemon entities using the prompt `"{entity} is a"`, computes the Pokemon direction as the normalized difference-in-means, and projects all entities onto this direction.

See [REPORT.md](REPORT.md) for full methodology and analysis.
