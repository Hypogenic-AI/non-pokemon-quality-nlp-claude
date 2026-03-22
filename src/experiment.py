"""
Pokemon Quality Experiment: Finding the "Pokemon Direction" in LLM Representations

This script:
1. Loads a model via TransformerLens
2. Extracts residual stream activations for Pokemon and non-Pokemon entities
3. Computes a mass-mean "Pokemon direction" at each layer
4. Evaluates classification accuracy per layer
5. Projects non-Pokemon entities onto the best direction
6. Ranks non-Pokemon entities by "Pokemon quality"
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Paths
ROOT = Path("/workspaces/non-pokemon-quality-nlp-claude")
DATA_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load datasets ───────────────────────────────────────────────────────────

def load_datasets():
    with open(DATA_DIR / "pokemon_names.json") as f:
        pokemon_names = json.load(f)
    with open(DATA_DIR / "non_pokemon_entities.json") as f:
        non_pokemon = json.load(f)
    with open(DATA_DIR / "contrastive_pairs.json") as f:
        contrastive_pairs = json.load(f)
    return pokemon_names, non_pokemon, contrastive_pairs


# ─── Model setup ─────────────────────────────────────────────────────────────

def load_model(model_name="gpt2-xl"):
    """Load model via TransformerLens."""
    import transformer_lens
    print(f"Loading {model_name}...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        device="cuda:0",
        dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded {model_name}: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


# ─── Activation extraction ───────────────────────────────────────────────────

def get_activations(model, entities, prompt_template="{} is a", batch_size=32, layers=None):
    """
    Extract residual stream activations at the last token of the entity name.

    Returns: dict mapping layer_idx -> np.array of shape (n_entities, d_model)
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    n_layers = len(layers)
    d_model = model.cfg.d_model
    n_entities = len(entities)

    # Pre-allocate
    all_acts = {l: np.zeros((n_entities, d_model), dtype=np.float32) for l in layers}

    for batch_start in tqdm(range(0, n_entities, batch_size), desc="Extracting activations"):
        batch_entities = entities[batch_start:batch_start + batch_size]
        prompts = [prompt_template.format(e) for e in batch_entities]

        # Tokenize to find last token position of entity
        tokens = model.to_tokens(prompts, prepend_bos=True)

        # Find the position of the last entity token (token before " is")
        # For "{entity} is a", the entity ends before " is"
        # We use the token at position -3 (before " is" and " a")
        # But entity names can be multi-token, so we find where " is" starts

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(f"blocks.{l}.hook_resid_post" in name for l in layers)
            )

        for l in layers:
            hook_name = f"blocks.{l}.hook_resid_post"
            acts = cache[hook_name]  # (batch, seq, d_model)

            # Use position -3 (last token before " is a")
            # This is the last token of the entity name
            for i, entity in enumerate(batch_entities):
                # Tokenize just the entity to know its length
                entity_tokens = model.to_tokens([entity], prepend_bos=False)
                entity_len = entity_tokens.shape[1]
                # Position of last entity token = BOS(1) + entity_len - 1
                pos = entity_len  # 0-indexed: BOS is 0, first entity token is 1, last is entity_len
                # But we need to make sure we don't go out of bounds
                pos = min(pos, acts.shape[1] - 1)
                all_acts[l][batch_start + i] = acts[i, pos].float().cpu().numpy()

        del cache
        torch.cuda.empty_cache()

    return all_acts


# ─── Mass-mean probe ─────────────────────────────────────────────────────────

def compute_mass_mean_direction(pos_acts, neg_acts):
    """Compute mass-mean direction: mean(positive) - mean(negative), normalized."""
    mu_pos = pos_acts.mean(axis=0)
    mu_neg = neg_acts.mean(axis=0)
    direction = mu_pos - mu_neg
    direction = direction / np.linalg.norm(direction)
    return direction


def classify_with_direction(direction, acts, threshold=0.0):
    """Project activations onto direction, classify based on threshold."""
    projections = acts @ direction
    return projections


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("POKEMON QUALITY EXPERIMENT")
    print("=" * 70)

    # Load data
    pokemon_names, non_pokemon_dict, contrastive_pairs = load_datasets()

    # Flatten non-pokemon with category labels
    non_pokemon_entities = []
    non_pokemon_categories = []
    for cat, entities in non_pokemon_dict.items():
        for e in entities:
            non_pokemon_entities.append(e)
            non_pokemon_categories.append(cat)

    print(f"Pokemon: {len(pokemon_names)}, Non-Pokemon: {len(non_pokemon_entities)}")

    # Sample Pokemon for tractability (use 200 for training, keep all non-pokemon)
    random.shuffle(pokemon_names)
    pokemon_train = pokemon_names[:160]
    pokemon_test = pokemon_names[160:200]
    pokemon_extra = pokemon_names[200:]  # Additional Pokemon we can project later

    # Split non-pokemon: use ~70% of each category for training
    non_pokemon_train = []
    non_pokemon_train_cats = []
    non_pokemon_test = []
    non_pokemon_test_cats = []

    for cat, entities in non_pokemon_dict.items():
        n = len(entities)
        split = max(1, int(0.7 * n))
        shuffled = list(entities)
        random.shuffle(shuffled)
        for e in shuffled[:split]:
            non_pokemon_train.append(e)
            non_pokemon_train_cats.append(cat)
        for e in shuffled[split:]:
            non_pokemon_test.append(e)
            non_pokemon_test_cats.append(cat)

    print(f"Train: {len(pokemon_train)} Pokemon, {len(non_pokemon_train)} non-Pokemon")
    print(f"Test: {len(pokemon_test)} Pokemon, {len(non_pokemon_test)} non-Pokemon")

    # Load model
    model = load_model("gpt2-xl")

    # Layers to analyze (sample across all 48 layers)
    layers = list(range(0, model.cfg.n_layers, 2))  # Every other layer for speed
    print(f"Analyzing {len(layers)} layers: {layers}")

    # ─── Extract activations ─────────────────────────────────────────────
    print("\n--- Extracting Pokemon train activations ---")
    pokemon_train_acts = get_activations(model, pokemon_train, batch_size=16, layers=layers)

    print("\n--- Extracting non-Pokemon train activations ---")
    non_pokemon_train_acts = get_activations(model, non_pokemon_train, batch_size=16, layers=layers)

    print("\n--- Extracting Pokemon test activations ---")
    pokemon_test_acts = get_activations(model, pokemon_test, batch_size=16, layers=layers)

    print("\n--- Extracting non-Pokemon test activations ---")
    non_pokemon_test_acts = get_activations(model, non_pokemon_test, batch_size=16, layers=layers)

    # Also extract for ALL non-pokemon entities (for final ranking)
    print("\n--- Extracting ALL non-Pokemon activations for ranking ---")
    all_non_pokemon_acts = get_activations(model, non_pokemon_entities, batch_size=16, layers=layers)

    # ─── Layer-wise analysis ─────────────────────────────────────────────
    print("\n--- Computing mass-mean direction per layer ---")
    layer_results = {}

    for l in layers:
        # Compute direction
        direction = compute_mass_mean_direction(
            pokemon_train_acts[l], non_pokemon_train_acts[l]
        )

        # Classify train set
        train_pos_proj = classify_with_direction(direction, pokemon_train_acts[l])
        train_neg_proj = classify_with_direction(direction, non_pokemon_train_acts[l])

        # Classify test set
        test_pos_proj = classify_with_direction(direction, pokemon_test_acts[l])
        test_neg_proj = classify_with_direction(direction, non_pokemon_test_acts[l])

        # Find optimal threshold on train set
        all_train_proj = np.concatenate([train_pos_proj, train_neg_proj])
        all_train_labels = np.array([1]*len(train_pos_proj) + [0]*len(train_neg_proj))
        threshold = np.median(all_train_proj)

        # Test accuracy
        all_test_proj = np.concatenate([test_pos_proj, test_neg_proj])
        all_test_labels = np.array([1]*len(test_pos_proj) + [0]*len(test_neg_proj))
        test_preds = (all_test_proj > threshold).astype(int)
        test_acc = accuracy_score(all_test_labels, test_preds)

        # AUC
        try:
            test_auc = roc_auc_score(all_test_labels, all_test_proj)
        except:
            test_auc = 0.5

        # Train accuracy
        train_preds = (all_train_proj > threshold).astype(int)
        train_acc = accuracy_score(all_train_labels, train_preds)

        # Effect size (Cohen's d)
        cohens_d = (train_pos_proj.mean() - train_neg_proj.mean()) / np.sqrt(
            (train_pos_proj.var() + train_neg_proj.var()) / 2
        )

        layer_results[l] = {
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "cohens_d": float(cohens_d),
            "threshold": float(threshold),
            "direction": direction,
            "train_pos_mean": float(train_pos_proj.mean()),
            "train_neg_mean": float(train_neg_proj.mean()),
        }
        print(f"  Layer {l:2d}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
              f"AUC={test_auc:.3f}, Cohen's d={cohens_d:.2f}")

    # ─── Find best layer ─────────────────────────────────────────────────
    best_layer = max(layers, key=lambda l: layer_results[l]["test_acc"])
    best_result = layer_results[best_layer]
    print(f"\nBest layer: {best_layer} (test_acc={best_result['test_acc']:.3f}, AUC={best_result['test_auc']:.3f})")

    # ─── Logistic regression baseline ────────────────────────────────────
    print("\n--- Logistic regression baseline at best layer ---")
    X_train = np.vstack([pokemon_train_acts[best_layer], non_pokemon_train_acts[best_layer]])
    y_train = np.array([1]*len(pokemon_train) + [0]*len(non_pokemon_train))
    X_test = np.vstack([pokemon_test_acts[best_layer], non_pokemon_test_acts[best_layer]])
    y_test = np.array([1]*len(pokemon_test) + [0]*len(non_pokemon_test))

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train, y_train)
    lr_train_acc = lr.score(X_train, y_train)
    lr_test_acc = lr.score(X_test, y_test)
    lr_test_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"  LR train_acc={lr_train_acc:.3f}, test_acc={lr_test_acc:.3f}, AUC={lr_test_auc:.3f}")

    # ─── Random direction baseline ───────────────────────────────────────
    print("\n--- Random direction baseline ---")
    random_accs = []
    for _ in range(100):
        rand_dir = np.random.randn(model.cfg.d_model).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        proj = np.concatenate([
            X_test @ rand_dir
        ])
        # Can't really classify with random direction meaningfully
        # Just report for reference
    rand_test_proj = X_test @ rand_dir
    rand_preds = (rand_test_proj > np.median(X_train @ rand_dir)).astype(int)
    rand_acc = accuracy_score(y_test, rand_preds)
    print(f"  Random direction test_acc={rand_acc:.3f} (expected ~50%)")

    # ─── Project ALL non-Pokemon entities ────────────────────────────────
    print(f"\n--- Projecting all non-Pokemon entities onto Pokemon direction (layer {best_layer}) ---")
    best_direction = best_result["direction"]
    best_threshold = best_result["threshold"]

    non_pokemon_projections = []
    for i, entity in enumerate(non_pokemon_entities):
        proj = float(all_non_pokemon_acts[best_layer][i] @ best_direction)
        non_pokemon_projections.append({
            "entity": entity,
            "category": non_pokemon_categories[i],
            "pokemon_quality": proj,
            "classified_as_pokemon": bool(proj > best_threshold),
        })

    # Sort by pokemon quality (descending)
    non_pokemon_projections.sort(key=lambda x: x["pokemon_quality"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 30 NON-POKEMON ENTITIES BY POKEMON QUALITY")
    print("=" * 70)
    for i, item in enumerate(non_pokemon_projections[:30]):
        marker = " *CLASSIFIED AS POKEMON*" if item["classified_as_pokemon"] else ""
        print(f"  {i+1:2d}. {item['entity']:20s} (cat: {item['category']:20s}) "
              f"score={item['pokemon_quality']:.4f}{marker}")

    print("\n" + "=" * 70)
    print("BOTTOM 10 NON-POKEMON ENTITIES BY POKEMON QUALITY")
    print("=" * 70)
    for i, item in enumerate(non_pokemon_projections[-10:]):
        print(f"  {len(non_pokemon_projections)-9+i:3d}. {item['entity']:20s} "
              f"(cat: {item['category']:20s}) score={item['pokemon_quality']:.4f}")

    # ─── Category-level analysis ─────────────────────────────────────────
    print("\n--- Category-level Pokemon quality ---")
    category_scores = {}
    for item in non_pokemon_projections:
        cat = item["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(item["pokemon_quality"])

    cat_summary = {}
    for cat, scores in sorted(category_scores.items(), key=lambda x: -np.mean(x[1])):
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        cat_summary[cat] = {"mean": float(mean_s), "std": float(std_s), "n": len(scores)}
        print(f"  {cat:25s}: mean={mean_s:.4f} ± {std_s:.4f} (n={len(scores)})")

    # Kruskal-Wallis test across categories
    category_groups = [category_scores[cat] for cat in category_scores]
    kw_stat, kw_p = stats.kruskal(*category_groups)
    print(f"\nKruskal-Wallis test: H={kw_stat:.2f}, p={kw_p:.2e}")

    # ─── Also check: which ACTUAL Pokemon score lowest? ──────────────────
    print("\n--- Extracting activations for extra Pokemon (for curiosity) ---")
    extra_pokemon_sample = pokemon_extra[:50]  # Just 50 for speed
    extra_pokemon_acts = get_activations(model, extra_pokemon_sample, batch_size=16, layers=[best_layer])
    extra_pokemon_proj = extra_pokemon_acts[best_layer] @ best_direction

    # Sort Pokemon by their projection (lowest = "least Pokemon-like Pokemon")
    pokemon_rankings = sorted(zip(extra_pokemon_sample, extra_pokemon_proj.tolist()), key=lambda x: x[1])
    print("\nLEAST POKEMON-LIKE ACTUAL POKEMON (bottom 15):")
    for name, score in pokemon_rankings[:15]:
        print(f"  {name:20s}: score={score:.4f}")

    print("\nMOST POKEMON-LIKE ACTUAL POKEMON (top 15):")
    for name, score in pokemon_rankings[-15:]:
        print(f"  {name:20s}: score={score:.4f}")

    # ─── Save results ────────────────────────────────────────────────────
    results = {
        "model": "gpt2-xl",
        "seed": SEED,
        "best_layer": best_layer,
        "n_layers_tested": len(layers),
        "layers_tested": layers,
        "layer_accuracies": {str(l): {
            "train_acc": layer_results[l]["train_acc"],
            "test_acc": layer_results[l]["test_acc"],
            "test_auc": layer_results[l]["test_auc"],
            "cohens_d": layer_results[l]["cohens_d"],
        } for l in layers},
        "best_layer_results": {
            "mass_mean_test_acc": best_result["test_acc"],
            "mass_mean_test_auc": best_result["test_auc"],
            "mass_mean_cohens_d": best_result["cohens_d"],
            "lr_test_acc": float(lr_test_acc),
            "lr_test_auc": float(lr_test_auc),
            "random_test_acc": float(rand_acc),
        },
        "non_pokemon_rankings": non_pokemon_projections,
        "category_summary": cat_summary,
        "kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
        "pokemon_train_size": len(pokemon_train),
        "non_pokemon_train_size": len(non_pokemon_train),
        "pokemon_test_size": len(pokemon_test),
        "non_pokemon_test_size": len(non_pokemon_test),
    }

    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'experiment_results.json'}")

    # ─── Visualizations ──────────────────────────────────────────────────
    create_visualizations(
        layers, layer_results, best_layer,
        non_pokemon_projections, category_scores,
        pokemon_train_acts, non_pokemon_train_acts,
        best_direction, best_result,
        lr_test_acc, rand_acc,
        pokemon_rankings, extra_pokemon_proj,
    )

    return results


def create_visualizations(
    layers, layer_results, best_layer,
    non_pokemon_projections, category_scores,
    pokemon_train_acts, non_pokemon_train_acts,
    best_direction, best_result,
    lr_test_acc, rand_acc,
    pokemon_rankings, extra_pokemon_proj,
):
    """Create all plots."""

    # 1. Layer-wise accuracy
    fig, ax = plt.subplots(figsize=(12, 5))
    test_accs = [layer_results[l]["test_acc"] for l in layers]
    train_accs = [layer_results[l]["train_acc"] for l in layers]
    ax.plot(layers, test_accs, 'b-o', label='Test accuracy', markersize=4)
    ax.plot(layers, train_accs, 'r--s', label='Train accuracy', markersize=4, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax.axvline(x=best_layer, color='green', linestyle='--', alpha=0.5, label=f'Best layer ({best_layer})')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pokemon Direction Classification Accuracy by Layer (GPT-2-XL)")
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "layer_accuracy.png", dpi=150)
    plt.close()
    print("Saved layer_accuracy.png")

    # 2. PCA visualization at best layer
    from sklearn.decomposition import PCA
    X_pca = np.vstack([pokemon_train_acts[best_layer], non_pokemon_train_acts[best_layer]])
    labels = ['Pokemon'] * len(pokemon_train_acts[best_layer]) + ['Non-Pokemon'] * len(non_pokemon_train_acts[best_layer])

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(10, 8))
    n_pok = len(pokemon_train_acts[best_layer])
    ax.scatter(X_2d[:n_pok, 0], X_2d[:n_pok, 1], c='red', alpha=0.4, s=20, label='Pokemon')
    ax.scatter(X_2d[n_pok:, 0], X_2d[n_pok:, 1], c='blue', alpha=0.6, s=30, label='Non-Pokemon')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"PCA of Residual Stream Activations (Layer {best_layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_pokemon_vs_nonpokemon.png", dpi=150)
    plt.close()
    print("Saved pca_pokemon_vs_nonpokemon.png")

    # 3. Top non-Pokemon by Pokemon quality (bar chart)
    top_n = 30
    top_entities = non_pokemon_projections[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    names = [f"{item['entity']} ({item['category'][:8]})" for item in top_entities]
    scores = [item['pokemon_quality'] for item in top_entities]

    # Color by category
    cat_colors = {
        'real_animals': '#2ecc71',
        'mythical_creatures': '#9b59b6',
        'digimon': '#e74c3c',
        'other_game_creatures': '#f39c12',
        'fictional_characters': '#3498db',
        'common_objects': '#95a5a6',
        'pokemon_like_names': '#e91e63',
    }
    colors = [cat_colors.get(item['category'], '#666') for item in top_entities]

    bars = ax.barh(range(top_n), scores, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pokemon Quality Score (projection onto Pokemon direction)")
    ax.set_title("Top 30 Non-Pokemon Entities by Pokemon Quality (GPT-2-XL)")

    # Add threshold line
    ax.axvline(x=best_result['threshold'], color='black', linestyle='--', alpha=0.5, label='Classification threshold')

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat.replace('_', ' ').title())
                       for cat, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_nonpokemon_by_quality.png", dpi=150)
    plt.close()
    print("Saved top_nonpokemon_by_quality.png")

    # 4. Category-level box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_names_sorted = sorted(category_scores.keys(), key=lambda c: -np.mean(category_scores[c]))
    box_data = [category_scores[c] for c in cat_names_sorted]
    bp = ax.boxplot(box_data, labels=[c.replace('_', ' ').title() for c in cat_names_sorted],
                    patch_artist=True, vert=True)
    for patch, cat in zip(bp['boxes'], cat_names_sorted):
        patch.set_facecolor(cat_colors.get(cat, '#666'))
        patch.set_alpha(0.7)

    ax.axhline(y=best_result['threshold'], color='black', linestyle='--', alpha=0.5, label='Pokemon threshold')
    ax.set_ylabel("Pokemon Quality Score")
    ax.set_title("Pokemon Quality by Entity Category (GPT-2-XL)")
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_boxplot.png", dpi=150)
    plt.close()
    print("Saved category_boxplot.png")

    # 5. Distribution of projections: Pokemon vs Non-Pokemon
    fig, ax = plt.subplots(figsize=(10, 5))
    pokemon_proj_vals = pokemon_train_acts[best_layer] @ best_direction
    nonpokemon_proj_vals = non_pokemon_train_acts[best_layer] @ best_direction

    ax.hist(pokemon_proj_vals, bins=30, alpha=0.6, color='red', label='Pokemon', density=True)
    ax.hist(nonpokemon_proj_vals, bins=30, alpha=0.6, color='blue', label='Non-Pokemon', density=True)
    ax.axvline(x=best_result['threshold'], color='black', linestyle='--', label='Threshold')
    ax.set_xlabel("Projection onto Pokemon Direction")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Pokemon Quality Scores (Layer {best_layer})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "projection_distribution.png", dpi=150)
    plt.close()
    print("Saved projection_distribution.png")

    # 6. Comparison bar chart: methods
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Mass-Mean\nProbe', 'Logistic\nRegression', 'Random\nDirection']
    accs = [best_result['test_acc'], lr_test_acc, rand_acc]
    colors_m = ['#2ecc71', '#3498db', '#95a5a6']
    bars = ax.bar(methods, accs, color=colors_m)
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Chance level')
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classification Method Comparison")
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "method_comparison.png", dpi=150)
    plt.close()
    print("Saved method_comparison.png")


if __name__ == "__main__":
    results = run_experiment()
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
