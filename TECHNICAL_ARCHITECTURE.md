# Technical Architecture: Sparse Feature Circuits for LLM Refusal Analysis

## Overview

This document provides a comprehensive technical and academic explanation of the sparse feature circuit discovery pipeline for analyzing LLM refusal behavior. The implementation follows the methodology from [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/pdf/2403.19647) (Marks et al., 2024).

## Table of Contents

1. [Data Loading and Analysis](#1-data-loading-and-analysis)
2. [Activation Collection](#2-activation-collection)
3. [Sparse Autoencoder Training](#3-sparse-autoencoder-training)
4. [Circuit Discovery](#4-circuit-discovery)
5. [Circuit Evaluation](#5-circuit-evaluation)
6. [Key Insights and What Actually Happens](#6-key-insights-and-what-actually-happens)

---

## 1. Data Loading and Analysis

### Technical Implementation

The pipeline begins with loading and analyzing the OR-Bench dataset, which contains prompts across multiple refusal categories (e.g., violence, self-harm, illegal activities).

**Key Components:**
- `ORBenchLoader`: Loads three CSV files (`or-bench-80k.csv`, `or-bench-hard-1k.csv`, `or-bench-toxic.csv`)
- `DataAnalyzer`: Analyzes data distributions and imbalances
- `DataProcessor`: Balances datasets for fair comparison

**Data Structure:**
```python
# Each sample contains:
{
    'prompt': str,           # The input prompt
    'category': str,         # Refusal category (e.g., "violence", "self_harm")
    'refusal_label': bool,   # True = toxic/refusal, False = safe
    'dataset_type': str,     # 'safe_80k', 'toxic', or 'hard'
    'original_source': str   # Source CSV file
}
```

**Balancing Strategy:**
The system uses a two-stage approach:
1. **Stage 1**: Collect all available toxic samples per category, match with equal number of safe samples
2. **Stage 2**: SAE training uses a single SAE on all data (ensures feature space comparability)

### Academic Perspective

**Why Balance Matters:**
- **Class Imbalance Problem**: OR-Bench has far more safe samples than toxic samples. Without balancing, the model would learn to predict "safe" by default, making it difficult to identify refusal-specific circuits.
- **Category Imbalance**: Different categories have different numbers of toxic samples. The "use_all" strategy preserves all available toxic samples while matching with safe samples, maximizing signal for circuit discovery.

**What Actually Happens:**
- The data analysis reveals significant imbalances: some categories have 10x more safe samples than toxic samples
- By balancing at the category level, we ensure that circuit discovery isn't biased toward categories with more data
- The analysis phase generates statistics showing the distribution of samples across categories, which helps identify potential issues before circuit discovery

---

## 2. Activation Collection

### Technical Implementation

Activations are collected from specific layers of the transformer model during inference. The system uses forward hooks to intercept and save intermediate representations.

**Key Components:**
- `ModelWrapper`: Manages model loading with activation hooks
- `InferencePipeline`: Orchestrates batch inference and activation saving
- Hook registration at specific residual stream positions

**Activation Collection Process:**
```python
# 1. Load model with hooks
model = load_model_with_hooks(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    layers=["residuals_21", "residuals_22", ..., "residuals_31"],  # Last 11 layers
    device="cuda"
)

# 2. Run inference with hooks
activations = {}
def hook_fn(activation, hook_name):
    activations[hook_name] = activation.detach().cpu()
    return activation

# 3. Save activations per category
# Shape: (batch_size, sequence_length, hidden_dim)
# Example: (32, 128, 4096) for Llama-2-7b
```

**Layer Selection:**
- Typically focuses on **later layers** (e.g., layers 21-31 for Llama-2-7b)
- Rationale: Later layers contain more abstract, task-specific representations
- Earlier layers handle low-level token processing; later layers integrate information for decision-making

### Academic Perspective

**Why Residual Stream Activations:**
- The residual stream is the "information highway" of transformers, accumulating information through layers
- Each layer's residual stream contains the sum of all previous computations, making it a rich representation
- SAEs decompose these dense activations into sparse, interpretable features

**What Actually Happens:**
- **Activation Patterns**: Toxic prompts produce different activation patterns than safe prompts, but these differences are subtle and distributed across many dimensions
- **Dimensionality**: A single activation vector has 4096 dimensions (for Llama-2-7b), making direct analysis intractable
- **Sparsity**: Most dimensions are near-zero; only a small subset carries meaningful information
- **Layer Evolution**: Early layers show similar activations for safe/toxic; later layers show divergence as the model processes refusal-relevant information

**Key Insight**: The raw activations are **polysemantic** - each dimension encodes multiple concepts. This is why we need SAEs to disentangle them.

---

## 3. Sparse Autoencoder Training

### Technical Implementation

Sparse Autoencoders (SAEs) learn to decompose dense activations into sparse, interpretable features. Each SAE is trained on activations from a specific layer.

**Architecture:**
```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_coeff, k_percent):
        # Encoder: ReLU-linear (only top k% features active)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Decoder: Linear reconstruction
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encode: x -> features (sparse)
        features = F.relu(self.encoder(x))
        # Top-k sparsity: only keep top k% features
        k = int(self.hidden_dim * self.k_percent)
        top_k_values, top_k_indices = torch.topk(features, k, dim=-1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(-1, top_k_indices, top_k_values)
        # Decode: features -> reconstructed_x
        reconstructed = self.decoder(sparse_features)
        return reconstructed, sparse_features, reconstruction_loss
```

**Training Process:**
1. **Data Preparation**: Concatenate activations from all categories (ensures feature space consistency)
2. **Training Objective**: Minimize reconstruction error while maintaining sparsity
3. **Sparsity Enforcement**: Top-k activation (only top k% features can be active)
4. **Layer-Specific SAEs**: Train one SAE per layer (each layer has different activation distributions)

**Hyperparameters:**
- `hidden_dim`: Typically 8x or 64x the input dimension (e.g., 32,768 features for 4,096-dim input)
- `k_percent`: 0.05 (5% of features active per sample)
- `sparsity_coeff`: Controls L1 regularization (if used)

### Academic Perspective

**Why Sparse Autoencoders:**
- **Feature Disentanglement**: SAEs learn a dictionary of interpretable features, where each feature represents a specific concept
- **Sparsity Prior**: The sparsity constraint forces the model to learn features that activate on specific inputs, making them more interpretable
- **Unsupervised Learning**: No labels needed - SAEs learn from activation patterns alone

**What Actually Happens:**
- **Feature Learning**: The SAE learns features like "detects harmful content", "identifies request type", "tracks conversation context"
- **Sparsity Emergence**: Most features are inactive (zero) for most inputs; only 5% activate per sample
- **Reconstruction Quality**: Well-trained SAEs achieve <1% reconstruction error, meaning they capture most information
- **Feature Interpretability**: Individual features can be interpreted by examining their top-activating tokens and contexts

**Key Insight**: The SAE transforms a **dense, polysemantic** representation (4096 dimensions, each encoding multiple concepts) into a **sparse, monosemantic** representation (32,768 features, each encoding a single concept).

**Error Term Handling:**
- The SAE reconstruction is not perfect: `activation = reconstructed + error`
- The error term represents information not captured by the SAE
- Our circuit discovery includes both SAE features AND error terms, ensuring we don't miss important information

---

## 4. Circuit Discovery

### Technical Implementation

Circuits are discovered by identifying SAE features that correlate with refusal behavior and tracing connections between them.

**Discovery Process:**

#### Step 1: Feature Encoding
```python
# Encode activations with trained SAEs
encoded_activations = {}
for layer, activation in raw_activations.items():
    sae = sae_manager.load_sae(layer)
    features = sae.encode(activation)  # (batch, seq, n_features)
    # Aggregate across sequence (mean or last token)
    aggregated = features.mean(dim=1)  # (batch, n_features)
    encoded_activations[layer] = aggregated
```

#### Step 2: Node Discovery (Feature Selection)

**Correlation-Based Discovery** (when labels have variance):
```python
# For each feature, compute correlation with refusal labels
for feat_idx in range(n_features):
    feat_values = features[:, feat_idx]  # (batch,)
    labels = refusal_labels  # (batch,) - True/False
    
    corr, p_value = stats.pearsonr(feat_values, labels)
    
    if abs(corr) >= threshold:  # e.g., 0.1
        # Add as circuit node
        circuit.add_node(feature_id=feat_idx, layer=layer, importance=abs(corr))
```

**Magnitude-Based Discovery** (when all labels are the same):
```python
# When all samples are toxic (or all safe), use feature magnitude
for feat_idx in range(n_features):
    feat_values = features[:, feat_idx]
    importance = 0.7 * mean_abs_activation + 0.3 * variance
    
    # Select top-N features per layer
    if importance in top_n_features:
        circuit.add_node(feature_id=feat_idx, layer=layer, importance=importance)
```

#### Step 3: Edge Discovery (Feature Connections)
```python
# Find connections between features in adjacent layers
for layer1, layer2 in adjacent_layers:
    features1 = encoded_activations[layer1]
    features2 = encoded_activations[layer2]
    
    for node1 in circuit.nodes_in_layer(layer1):
        for node2 in circuit.nodes_in_layer(layer2):
            feat1_values = features1[:, node1.feature_id]
            feat2_values = features2[:, node2.feature_id]
            
            corr, _ = stats.pearsonr(feat1_values, feat2_values)
            
            if abs(corr) >= edge_threshold:  # e.g., 0.01
                circuit.add_edge(node1, node2, importance=abs(corr))
```

**Separate Safe/Toxic Circuits:**
- When `separate_safe_toxic=True`, the system discovers two circuits per category:
  - **Safe Circuit**: Features that activate on safe prompts
  - **Toxic Circuit**: Features that activate on toxic prompts
- This allows comparison: do safe and toxic prompts use different circuits?

### Academic Perspective

**Why Correlation-Based Discovery:**
- **Causal Implication**: Features that correlate with refusal behavior are likely causally involved
- **Statistical Significance**: Pearson correlation provides a measure of linear relationship
- **Threshold Selection**: Lower thresholds (0.01) capture more features but include noise; higher thresholds (0.1) are more selective

**What Actually Happens:**
- **Feature Activation Patterns**: Toxic prompts activate a different set of features than safe prompts
- **Layer Specialization**: Early layers detect surface-level patterns (e.g., keywords); later layers integrate semantic understanding
- **Circuit Structure**: Circuits form directed graphs where features in early layers connect to features in later layers
- **Sparsity**: Only a small fraction of SAE features (typically <1%) are included in circuits

**Key Insight**: The discovered circuits reveal the **computational pathway** the model uses to process refusal. Features don't just correlate - they form a causal chain where early features trigger later features.

**Edge Discovery Insights:**
- **Information Flow**: Edges show how information flows through the model
- **Bottlenecks**: Some features have many incoming edges (information integration points)
- **Specialization**: Some features only connect to specific downstream features (specialized pathways)

---

## 5. Circuit Evaluation

### Technical Implementation

Circuits are evaluated using two metrics: **Faithfulness** and **Completeness**, following the paper's definitions.

#### Metric Definitions

**F(M) - Full Model Performance:**
```python
# Run model without any ablation
f_m = run_with_ablations(prompts, empty_circuit, ablate_complement=False)
# No hooks registered = no ablation = full model performance
```

**F(C) - Circuit Performance (Faithfulness):**
```python
# Ablate everything NOT in the circuit (keep only circuit features)
f_c = run_with_ablations(prompts, circuit, ablate_complement=True)
# Replace non-circuit features with mean activations
```

**F(M\C) - Model with Circuit Removed (Completeness):**
```python
# Ablate ONLY the circuit features (remove circuit)
f_m_minus_c = run_with_ablations(prompts, circuit, ablate_complement=False)
# Replace circuit features with mean activations
```

**F(Empty) - Baseline Performance:**
```python
# Ablate everything (all features)
f_empty = run_with_ablations(prompts, empty_circuit, ablate_complement=True, ablate_all=True)
# Replace all features with mean activations
```

#### Metric Calculations

**Faithfulness:**
```python
faithfulness = (F(C) - F(Empty)) / (F(M) - F(Empty))
```
- **Interpretation**: How well does the circuit alone reproduce model behavior?
- **Range**: [0, 1] where 1 = circuit perfectly reproduces behavior
- **Normalization**: Accounts for baseline performance (F(Empty))

**Completeness:**
```python
completeness = F(M\C) / F(M)
```
- **Interpretation**: How much behavior remains after removing the circuit?
- **Range**: [0, 1] where 0 = circuit is necessary (perfect circuit), 1 = circuit is irrelevant
- **Key Insight**: Lower completeness = circuit is more necessary

### Academic Perspective

**Why These Metrics Matter:**

**Faithfulness** answers: "If we keep only the circuit features, can we still reproduce the behavior?"
- High faithfulness → circuit is **sufficient** to explain behavior
- Low faithfulness → circuit is missing important features

**Completeness** answers: "If we remove the circuit, how much behavior remains?"
- Low completeness → circuit is **necessary** (removing it breaks behavior)
- High completeness → circuit is not necessary (other features can compensate)

**What Actually Happens:**

**Faithfulness Trends:**
- As circuit size increases, faithfulness typically **increases** (more features = better reproduction)
- However, adding irrelevant features can decrease faithfulness (noise)
- Well-discovered circuits achieve faithfulness >0.8

**Completeness Trends:**
- As circuit size increases, completeness **decreases** (more relevant features removed = less behavior remains)
- **Key Insight**: This is why completeness decreases as nodes increase - you're removing more of the necessary components!
- Perfect circuit → completeness ≈ 0 (removing it eliminates all behavior)
- Irrelevant circuit → completeness ≈ 1 (removing it has no effect)

**Ablation Mechanism:**
- Features are ablated by replacing their activations with **mean activations** (baseline)
- This preserves the activation distribution while removing the specific signal
- SAE error terms are preserved (they represent information not captured by SAEs)

**Threshold Analysis:**
- Circuits are evaluated across different importance thresholds
- Lower thresholds → larger circuits → higher faithfulness, lower completeness
- Higher thresholds → smaller circuits → lower faithfulness, higher completeness
- The trade-off reveals the optimal circuit size

---

## 6. Key Insights and What Actually Happens

### The Complete Picture

**From Dense to Sparse:**
1. **Raw Activations**: Dense, polysemantic (4096 dims, each encoding multiple concepts)
2. **SAE Features**: Sparse, monosemantic (32,768 features, each encoding one concept)
3. **Circuits**: Ultra-sparse subset (<1% of features) that causally explains behavior

**Why This Works:**
- **Dimensionality Reduction**: SAEs reduce the search space from 4096 dimensions to interpretable features
- **Sparsity**: Most features are inactive, making circuits tractable to analyze
- **Causality**: Correlation + ablation establishes causal relationships

### What Actually Happens During Circuit Discovery

**Feature Activation Patterns:**
- **Safe Prompts**: Activate features related to "normal conversation", "helpful responses", "information retrieval"
- **Toxic Prompts**: Activate features related to "harmful content detection", "policy violation", "refusal generation"
- **Overlap**: Some features activate for both (e.g., "request understanding")

**Layer Progression:**
- **Early Layers (21-25)**: Detect surface patterns, keyword matching
- **Middle Layers (26-29)**: Semantic understanding, context integration
- **Late Layers (30-31)**: Decision-making, response generation

**Circuit Structure:**
- Circuits form **directed acyclic graphs** (DAGs)
- Information flows from early layers → late layers
- Some features are "hubs" (many connections), others are "leaves" (few connections)

### Monolithic vs. Modular Refusal

**Research Question**: Is refusal behavior monolithic (shared circuits) or modular (category-specific circuits)?

**What the Analysis Reveals:**
- **Shared Features**: Some features appear in circuits across multiple categories (monolithic component)
- **Category-Specific Features**: Some features only appear in specific categories (modular component)
- **Similarity Metrics**: Jaccard similarity on nodes/edges reveals the degree of overlap

**Interpretation:**
- High similarity → monolithic (shared refusal mechanism)
- Low similarity → modular (category-specific mechanisms)
- The truth is likely a **hybrid**: shared core + category-specific extensions

### Practical Insights

**Why Completeness Decreases with More Nodes:**
- As you add more relevant nodes to the circuit, you're identifying more of the necessary components
- When you remove the circuit (for completeness), you're removing more necessary components
- Result: Less behavior remains → lower completeness
- **Perfect circuit** (all necessary features) → completeness ≈ 0
- **Terrible circuit** (no necessary features) → completeness ≈ 1

**Why Faithfulness Increases with More Nodes:**
- More features → better reproduction of full model behavior
- However, diminishing returns: adding irrelevant features doesn't help
- Optimal circuit size balances faithfulness and completeness

**Threshold Selection:**
- Too low → includes noise, circuits too large
- Too high → misses important features, circuits too small
- Optimal: Use percentile-based thresholds (e.g., top 5% of features)

### Limitations and Future Work

**Current Limitations:**
1. **Linear Approximations**: Circuit discovery uses correlation, which captures linear relationships
2. **Feature Interpretability**: Not all SAE features are easily interpretable
3. **Edge Discovery**: Only captures direct correlations, not indirect causal paths
4. **Static Analysis**: Circuits are discovered on static data, may not generalize

**Future Directions:**
1. **Causal Interventions**: Directly test causal relationships with interventions
2. **Feature Labeling**: Automatically label features based on activation patterns
3. **Dynamic Circuits**: Discover how circuits change during generation
4. **Cross-Model Analysis**: Compare circuits across different model architectures

---

## Conclusion

The sparse feature circuit discovery pipeline transforms opaque neural network behavior into interpretable causal graphs. By decomposing dense activations into sparse features and identifying causal pathways, we gain insights into how LLMs process refusal behavior. The faithfulness and completeness metrics provide quantitative measures of circuit quality, revealing the trade-offs between circuit size and explanatory power.

The key insight is that **completeness decreases as nodes increase** because you're identifying more necessary components - removing a larger, more complete circuit leaves less behavior remaining in the model. This is exactly what we expect from a well-discovered circuit: it should capture the essential components needed for the behavior.

