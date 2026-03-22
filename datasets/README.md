# Datasets for "Do any non-pokemon have non-trivial pokemon quality?"

This directory contains datasets for probing how LLMs encode the concept "is a pokemon."
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Pokemon Names (`pokemon_names.json`, `pokemon_full.csv`)

### Overview
- **Source**: [lgreski/pokemonData](https://github.com/lgreski/pokemonData) on GitHub
- **Size**: 1026 unique Pokemon names (Generations 1-9)
- **Format**: JSON list of names + full CSV with stats
- **License**: Public domain (fan-compiled data)

### Download Instructions

```bash
# Download Pokemon CSV
curl -o datasets/pokemon_full.csv \
  https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv

# Extract names to JSON
python3 -c "
import csv, json, io
with open('datasets/pokemon_full.csv') as f:
    names = sorted(set(row['Name'].strip() for row in csv.DictReader(f) if row['Name'].strip()))
with open('datasets/pokemon_names.json', 'w') as f:
    json.dump(names, f, indent=2)
print(f'{len(names)} Pokemon names saved')
"
```

### Sample Data
```json
["Abomasnow", "Abra", "Absol", "Accelgor", "Aegislash", "Aerodactyl", ...]
```

## Dataset 2: Non-Pokemon Entities (`non_pokemon_entities.json`)

### Overview
- **Source**: Curated for this research
- **Size**: 177 entities across 7 categories
- **Format**: JSON dict with category keys
- **Categories**:
  - `real_animals` (53): dog, cat, axolotl, platypus, ...
  - `mythical_creatures` (30): dragon, phoenix, kitsune, ...
  - `digimon` (22): Agumon, Gabumon, Patamon, ...
  - `other_game_creatures` (17): Chocobo, Slime, Yoshi, ...
  - `fictional_characters` (15): Harry Potter, Yoda, ...
  - `common_objects` (20): chair, sword, crystal, ...
  - `pokemon_like_names` (20): made-up names that sound pokemon-like

### Notes
- Categories chosen to test gradient of "pokemon-ness"
- Digimon and game creatures expected to score higher than objects
- `pokemon_like_names` are fabricated to test phonological similarity vs semantic knowledge

## Dataset 3: Contrastive Pairs (`contrastive_pairs.json`)

### Overview
- **Source**: Curated for this research
- **Size**: 49 pairs across 3 categories
- **Format**: JSON dict of (pokemon, analog) tuple lists
- **Categories**:
  - `pokemon_vs_animal` (30): (Pikachu, mouse), (Charizard, dragon), ...
  - `pokemon_vs_mythical` (9): (Articuno, phoenix), (Mewtwo, alien), ...
  - `pokemon_vs_digimon` (10): (Pikachu, Agumon), (Charizard, Greymon), ...

### Purpose
Used for the counterfactual pair method from Park et al. (2024) to estimate the
"pokemon direction" in representation space.

## Dataset 4: Prompt Templates (`prompt_templates.json`)

### Overview
- **Source**: Designed for this research
- **Format**: JSON dict with template categories
- **Categories**:
  - `direct_classification`: "{entity} is a Pokemon." / "Is {entity} a Pokemon?"
  - `context_probing`: "I caught a wild {entity}!" etc.
  - `contrastive_stimuli`: paired prompts for RepE-style reading

### Purpose
Templates for extracting model activations during probing experiments.
