# Paper Notes: Locating and Editing Factual Associations in GPT (ROME)
**arXiv: 2202.05262** | Meng, Bau, Andonian, Belinkov | NeurIPS 2022

## Core Thesis
Factual associations in autoregressive transformers correspond to localized, directly-editable computations in middle-layer feed-forward (MLP) modules, specifically when processing the last token of the subject entity name. Facts can be edited via Rank-One Model Editing (ROME) by modifying MLP weights.

## How They Locate Factual Associations: Causal Tracing

### Method: Three-Run Intervention
Each fact is represented as a knowledge tuple t = (s, r, o) -- subject, relation, object.

1. **Clean run**: Pass factual prompt through model, collect all hidden activations h_i^(l).
2. **Corrupted run**: Add Gaussian noise (3x embedding std dev) to subject token embeddings. Model loses subject info, predicts incorrectly.
3. **Corrupted-with-restoration run**: Run corrupted input but restore ONE clean activation h_i^(l) at specific token i, layer l. Measure how much this restores correct output probability.

**Average Indirect Effect (AIE)** = P_{restored}[o] - P_{corrupted}[o], averaged over 1000 factual statements.

### Key Findings (Figure 2 -- CRITICAL RESULT)
Two important sites in GPT-2 XL (48 layers):

1. **Early site** (middle layers ~10-20, at LAST SUBJECT TOKEN):
   - AIE peaks at layer 15 (8.7% at single state level)
   - **MLP contributions dominate** (AIE 6.6%) vs attention (AIE 1.6%)
   - This is the NEW discovery

2. **Late site** (last layers ~30-48, at LAST TOKEN of prompt):
   - Attention dominates here
   - Unsurprising -- directly feeds into prediction

### Modified Causal Graph Analysis (Figure 3)
- Sever MLP modules at subject token to isolate path-specific effects.
- Result: Early-layer states lose causal effect WITHOUT MLP, but higher-layer states are unaffected.
- Confirms MLP computation at middle layers is essential for factual recall.
- No such transition when severing attention instead.

### Cross-Model Consistency
- Same pattern found in GPT-2 XL (1.5B), GPT-J (6B), GPT-NeoX (20B), GPT-2 Medium (334M), GPT-2 Large (774M)
- Pattern is robust across different noise configurations and corruption rules
- Integrated Gradients (gradient-based method) does NOT reveal this pattern -- causal tracing is superior

## The Localized Factual Association Hypothesis (Section 2.3)

Proposed mechanism:
1. **MLP modules at middle layers** accept inputs encoding a subject (at last subject token)
2. They **produce outputs that recall memorized properties** about that subject
3. Middle layer MLP outputs **accumulate information** across layers
4. Late-layer attention **copies this information** to the prediction position
5. Any fact could equivalently be stored in any one of the middle MLP layers

## ROME: Rank-One Model Editing (Section 3.1)

### MLP as Linear Associative Memory
- View W_proj^(l) as a key-value memory store
- First layer W_fc forms keys, second layer W_proj retrieves values
- This differs from Geva et al.'s per-neuron view -- ROME treats it as a linear associative memory

### The ROME Algorithm

**Step 1: Choose key k* (select subject)**
- Run text containing subject s through model
- At layer l* and last subject token index i, read activation after nonlinearity inside MLP
- Average over ~50 random prefix texts for robustness:
  ```
  k* = (1/N) * sum_j k(x_j + s)
  ```

**Step 2: Choose value v* (encode fact)**
- Optimize v* = argmin_z L(z) where:
  - Term (a): Maximize probability of target object o* when z substituted as MLP output
  - Term (b): KL divergence to preserve model's understanding of subject's "essence" (predictions for "{subject} is a")
- Uses random prefix texts for context robustness
- Adam optimizer, lr=0.5, max 20 steps

**Step 3: Insert fact via rank-one update**
- W_hat = W + Lambda * (C^{-1} k*)^T
- C = KK^T is pre-cached uncentered covariance of k from Wikipedia text (100K samples)
- Lambda = (v* - Wk*) / ((C^{-1}k*)^T k*)
- Single rank-one update, takes ~2s on NVIDIA A6000

### Layer Selection
- ROME intervention at **layer 18** of GPT-2 XL (center of causal effect in MLP layers)
- Figure 5 confirms: editing at middle layers + last subject token gives best simultaneous generalization AND specificity

## Datasets

### COUNTERFACT Dataset (introduced in this paper)
- 21,919 records with counterfactual assertions
- Derived from PARAREL (hand-curated paraphrased prompts)
- Each record contains:
  - Requested rewrite: {s, r, o_c, o*, p*}
  - 2 paraphrase prompts (test generalization)
  - 10 neighborhood prompts (test specificity -- nearby subjects sharing predicates via WikiData SPARQL)
  - 3 generation prompts (test deeper generalization)
  - Reference texts (Wikipedia articles for consistency scoring)
- Much harder than zsRE: tests counterfactual changes with low initial scores

### Evaluation Metrics
- **Efficacy Score (ES)**: P[o*] > P[o_c] post-edit
- **Paraphrase Score (PS)**: Same but with rephrased prompts
- **Neighborhood Score (NS)**: Nearby subjects remain unaffected
- **Score (S)**: Harmonic mean of ES, PS, NS
- **Reference Score (RS)**: TF-IDF similarity of generated text to reference articles
- **Generation Entropy (GE)**: n-gram entropy to detect fluency degradation

### Results (Table 4)
- ROME on GPT-2 XL: S=89.2, ES=100%, PS=96.4%, NS=75.4%
- ROME on GPT-J: S=91.5, ES=99.9%, PS=99.1%, NS=78.9%
- All other methods (FT, FT+L, KE, MEND) either overfit to counterfactual (fail to generalize) or bleed over to unrelated subjects
- ROME uniquely achieves both generalization AND specificity

### Zero-Shot Relation Extraction (zsRE)
- 10,000 records from Levy et al. (2017)
- ROME: 99.8% efficacy, 88.1% paraphrase, 24.2% specificity (competitive with hypernetworks)

## Models Used
- **GPT-2 XL** (1.5B params, 48 layers)
- **GPT-J** (6B params, 28 layers)
- **GPT-2 Medium** (334M), **GPT-2 Large** (774M) -- in appendix
- **GPT-NeoX** (20B) -- for causal tracing only

## Code Availability
- **https://rome.baulab.info/** (code, dataset, interactive demo notebook, visualizations)

## Key Techniques Adaptable for "Is a Pokemon" Probing

1. **Causal Tracing methodology**: Could apply to find WHERE "pokemon knowledge" is stored. Corrupt a pokemon name, then restore individual activations to find which layer/token combination is decisive for pokemon-related predictions. This would tell us:
   - Which layers encode "is a pokemon" information
   - Whether it's at the last token of the pokemon name
   - Whether MLP or attention is more important

2. **MLP as key-value memory**: The concept of pokemon might be stored as a key (entity representation at last subject token in middle MLP layers) mapping to values (pokemon-related properties). This suggests we should look at MLP activations specifically.

3. **Rank-one editing for validation**: Could use ROME-style edits to test: can we make the model believe a non-pokemon IS a pokemon by editing a single MLP layer? If so, this confirms factual category membership is stored in that layer.

4. **Subject-token localization**: The finding that information concentrates at the LAST TOKEN of the subject name is crucial. When probing for "Pikachu is a pokemon," look at activations at the "chu" token (or whatever the last token is).

5. **COUNTERFACT-style evaluation**: Create pokemon/non-pokemon counterfactual dataset. E.g., change "Pikachu's type is Electric" to "Pikachu's type is Normal." Test generalization (works for paraphrases?) and specificity (doesn't change Raichu's type?).

6. **Covariance estimation C = E[kk^T]**: Pre-compute second moment statistics of MLP activations over Wikipedia text. This is needed for ROME edits and could also be useful for understanding the geometry of the key space.

## Important Caveats

- ROME only edits one fact at a time (MEMIT extends to multiple)
- Associations are directional ("Space Needle is in Seattle" vs "Seattle's landmark is Space Needle" stored separately)
- Model will hallucinate plausible but false facts after editing
- The paper focuses on factual subject-relation-object triples, not category membership per se
- Causal tracing depends on corruption rule, but results are robust across different noise configurations
- The "last subject token" finding is the dominant pattern but not universal (Figure 11 shows exceptions like "Windows Media Player" where "Windows" is the decisive token)
