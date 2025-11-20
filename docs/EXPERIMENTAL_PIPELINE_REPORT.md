# Experimental Pipeline Report: LLM Refusal Circuit Analysis

## Executive Summary

This report documents the complete experimental pipeline for analyzing whether Large Language Model (LLM) refusal behavior is **monolithic** (shared circuits across refusal categories) or **modular** (category-specific circuits). The pipeline employs sparse autoencoders (SAEs) to decompose model activations into interpretable features, then discovers sparse feature circuits that explain refusal behavior across different categories of harmful content.

**End Goal**: Determine if refusal circuits are shared across categories (monolithic hypothesis) or specialized per category (modular hypothesis) by comparing circuit similarity across refusal categories.

---

## 1. Data Preparation and Balancing

### 1.1 Dataset Structure

The pipeline uses the OR-Bench dataset, which contains three subsets:
- **Safe samples** (80k dataset): Prompts that should be answered normally
- **Hard samples** (1k dataset): Challenging prompts that may trigger refusal
- **Toxic samples**: Prompts requesting harmful content across multiple categories

Each sample is labeled with:
- **Category**: The type of harmful content (e.g., hate, violence, self-harm)
- **Refusal label**: Binary indicator (True = toxic/refusal expected, False = safe)

### 1.2 Two-Stage Balancing Strategy

The pipeline employs a sophisticated two-stage balancing approach to ensure fair comparison while preserving natural data distributions:

#### Stage 1: Safe-to-Toxic Balance (Activation Collection)

**Goal**: Ensure equal representation of safe and toxic samples per category for activation collection.

**Method**: For each category, the system:
1. Collects all available toxic samples for that category
2. Matches with an equal number of safe samples from the same category
3. Preserves the 1:1 safe-toxic ratio within each category

**Mathematical formulation**:
For category $c$, if $N_c^{toxic}$ toxic samples exist:
- Select all $N_c^{toxic}$ toxic samples
- Randomly sample $N_c^{toxic}$ safe samples from category $c$
- Final dataset: $2N_c^{toxic}$ samples (50% safe, 50% toxic)

**Rationale**: This ensures that activation patterns reflect both refusal and non-refusal behavior equally, preventing bias toward either class during feature learning.

#### Stage 2: Category Imbalance Handling (SAE Training)

**Problem**: Different categories have vastly different numbers of toxic samples, creating category-level imbalance.

**Example**: Category A might have 1000 toxic samples, while Category B has only 100, leading to 10:1 imbalance in the combined training set.

**Solution**: Inverse frequency weighting with weighted sampling.

**Mathematical formulation**:

1. **Compute category weights**:
   For category $c$ with toxic count $n_c$:
   $$
   w_c = \frac{N_{total}}{C \cdot n_c}
   $$
   where:
   - $N_{total} = \sum_{c} n_c$ (total toxic samples across all categories)
   - $C$ = number of categories
   - $n_c$ = toxic samples in category $c$

2. **Normalize weights**:
   $$
   w_c^{norm} = \frac{w_c}{\sum_{c'} w_{c'}}
   $$

3. **Weighted sampling**: During SAE training, samples from category $c$ are sampled with probability proportional to $w_c^{norm}$.

**Key Insight**: This weighting scheme ensures that:
- Categories with fewer samples (e.g., rare categories) receive higher weight
- Categories with many samples (e.g., common categories) receive lower weight
- The effective sample size per category becomes more balanced
- **Safe-to-toxic ratios are NOT modified** - only category-level balancing occurs

**Why not balance safe-toxic ratios?**: The 1:1 safe-toxic ratio is intentionally preserved because:
1. It ensures SAEs learn features that distinguish refusal from non-refusal
2. It maintains the natural distinction between safe and toxic behavior
3. The research question focuses on category-level differences, not safe-toxic differences

### 1.3 Data Aggregation Strategy

The pipeline supports two strategies for handling category imbalance at the data collection stage:

**"use_all" strategy**: Use all available toxic samples per category, matching with equal safe samples. This maximizes data usage but creates category imbalance.

The "use_all" strategy is preferred because category imbalance is handled via weighted sampling during SAE training, allowing maximum data utilization.

---

## 2. Activation Collection

### 2.1 Model Inference and Hook Registration

The pipeline collects internal activations from transformer models during inference. For each prompt, activations are captured at specified decoder layers using PyTorch forward hooks.

**Activation types collected**:
- **Residual stream activations**: The input to each decoder layer (residual stream)
- **MLP outputs**: Feed-forward network outputs at each layer
- **Attention outputs**: Self-attention outputs at each layer

**Layer selection**: Activations are collected from:
- All decoder layers (comprehensive analysis)

### 2.2 Activation Storage

For each category, activations are stored as tensors with shape:
- **2D case**: `(batch_size, hidden_dim)` - aggregated across sequence
- **3D case**: `(batch_size, sequence_length, hidden_dim)` - per-token activations

**Storage format**: PyTorch tensors saved to disk, preserving:
- Layer identifiers
- Activation values (typically float32 or float16)
- Sample metadata (implicitly via batch dimension)

### 2.3 Refusal Label Collection

Concurrently with activation collection, the pipeline:
1. Runs inference to generate model outputs
2. Detects refusal behavior using keyword-based heuristics
3. Stores ground-truth refusal labels (from dataset)
4. Stores detected refusal labels (from model output analysis)

This dual labeling allows comparison between expected and actual refusal behavior.

---

## 3. Sparse Autoencoder (SAE) Training

### 3.1 SAE Architecture and Objective

**Purpose**: SAEs decompose high-dimensional activations into sparse, interpretable features that capture meaningful patterns in the model's internal representations.

**Architecture**:
- **Encoder**: Linear transformation from activation dimension $d$ to feature dimension $h$ (typically $h > d$ for overcomplete representation)
- **Decoder**: Linear transformation from feature dimension $h$ back to activation dimension $d$
- **Activation function**: ReLU followed by top-$k$ sparsity

**Mathematical formulation**:

Given activation vector $\mathbf{x} \in \mathbb{R}^d$:

1. **Encoding**:
   $$
   \mathbf{z}_{pre} = \text{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)
   $$
   where $\mathbf{W}_e \in \mathbb{R}^{h \times d}$ is the encoder weight matrix.

2. **Top-$k$ sparsity**:
   $$
   \mathbf{z} = \text{TopK}(\mathbf{z}_{pre}, k)
   $$
   where TopK keeps only the $k$ largest activations, setting others to zero. Typically $k = 0.05h$ (5% of features active).

3. **Decoding**:
   $$
   \hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d
   $$
   where $\mathbf{W}_d \in \mathbb{R}^{d \times h}$ is the decoder weight matrix.

4. **Loss function**:
   $$
   \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} ||\mathbf{x}_i - \hat{\mathbf{x}}_i||_2^2
   $$
   Mean Squared Error (MSE) reconstruction loss.

### 3.2 Critical Design Choice: Single SAE for All Data

**Key Principle**: A **single SAE is trained on ALL data** (all categories, both safe and toxic samples) for each layer.

**Mathematical justification**:
- Training data: $\mathcal{D} = \bigcup_{c} \mathcal{D}_c^{safe} \cup \bigcup_{c} \mathcal{D}_c^{toxic}$
- Single SAE learns feature space: $f: \mathbb{R}^d \rightarrow \mathbb{R}^h$
- All samples are encoded using the same feature space

**Why this matters**:
1. **Feature space comparability**: Safe and toxic circuits use the same features, enabling direct comparison
2. **Consistent representation**: Features have consistent meaning across categories
3. **Fair comparison**: Circuit similarity reflects actual overlap, not feature space differences

**Alternative (rejected)**: Training separate SAEs per category would create incomparable feature spaces, making similarity comparisons meaningless.

### 3.3 Training Process

**Data preparation**:
1. Load activations from all categories
2. Apply category weights via weighted random sampling
3. Optionally balance safe-toxic samples (if enabled)
4. Create data loader with weighted sampler

**Training loop**:
- **Optimizer**: Adam with learning rate scheduling
- **Batch processing**: Process activations in batches
- **Metrics tracked**:
  - Reconstruction loss: How well SAE reconstructs activations
  - L0 norm: Average number of active features (sparsity measure)
  - Total loss: Combined objective

**Convergence**: Training continues until reconstruction loss converges (typically < 0.01) or maximum epochs reached.

### 3.4 What SAEs Learn

SAEs learn a **sparse dictionary** of interpretable features:

1. **Feature directions**: Each of the $h$ features represents a direction in activation space that captures a recurring pattern
2. **Sparse activation**: Only ~5% of features activate for any given input, ensuring interpretability
3. **Reconstruction fidelity**: Features must accurately reconstruct original activations, ensuring they capture meaningful information

**Interpretation**: Each feature can be thought of as detecting a specific "concept" or "pattern" in the model's internal representations. For example:
- Feature 42 might detect "harmful content detection"
- Feature 103 might detect "refusal trigger"
- Feature 256 might detect "category-specific patterns"

**Role in pipeline**: SAEs transform dense, uninterpretable activations into sparse, interpretable features that can be analyzed for circuit discovery.

---

## 4. Circuit Discovery

### 4.1 Circuit Representation

A **sparse feature circuit** is represented as a graph:

**Nodes**: Important SAE features (identified by feature index and layer)
- Each node represents a feature that correlates with refusal behavior
- Nodes have importance scores indicating their contribution to refusal

**Edges**: Connections between features across layers
- Edges represent information flow between features in adjacent layers
- Edge weights indicate correlation strength between connected features

**Mathematical representation**:
- Circuit $C = (V, E)$ where:
  - $V = \{v_i\}$: Set of feature nodes
  - $E = \{(v_i, v_j, w_{ij})\}$: Set of edges with weights
  - Each node $v_i$ has importance $\alpha_i$
  - Each edge $(v_i, v_j)$ has weight $w_{ij}$

### 4.2 Correlation-Based Discovery Algorithm

The circuit discovery algorithm uses **statistical correlation** to identify important features and their connections.

#### Step 1: Feature Aggregation

For each layer, activations are aggregated across the sequence dimension:
- **Mean aggregation**: $\bar{\mathbf{z}}_l = \frac{1}{T} \sum_{t=1}^{T} \mathbf{z}_{l,t}$
- **Max aggregation**: $\bar{\mathbf{z}}_l = \max_{t} \mathbf{z}_{l,t}$
- **Last token**: $\bar{\mathbf{z}}_l = \mathbf{z}_{l,T}$ (use final token only)

This produces per-sample feature vectors: $\bar{\mathbf{z}}_l \in \mathbb{R}^h$ for layer $l$.

#### Step 2: Node Discovery (Feature Selection)

For each feature $f$ in layer $l$, compute correlation with refusal labels:

**Pearson correlation coefficient**:
$$
r_f = \frac{\sum_{i=1}^{N} (z_{f,i} - \bar{z}_f)(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N}(z_{f,i} - \bar{z}_f)^2 \sum_{i=1}^{N}(y_i - \bar{y})^2}}
$$

where:
- $z_{f,i}$ = activation of feature $f$ for sample $i$
- $y_i \in \{0, 1\}$ = refusal label for sample $i$ (0 = safe, 1 = toxic/refusal)
- $N$ = number of samples
- $\bar{z}_f$ = mean activation of feature $f$
- $\bar{y}$ = mean refusal label

**Statistical significance**: Compute $p$-value using standard statistical tests to assess whether correlation is significant.

**Thresholding**: Features with $|r_f| \geq \theta_{node}$ are added as circuit nodes, where $\theta_{node}$ is the node threshold (typically 0.1).

**Alternative for uniform labels**: When all samples have the same label (e.g., all safe or all toxic), correlation cannot be computed. In this case, the algorithm uses **feature magnitude** instead:
- Importance = $0.7 \cdot \text{mean}(|\mathbf{z}_f|) + 0.3 \cdot \text{var}(\mathbf{z}_f)$
- Selects top-$N$ features by magnitude (typically top 100 per layer)

#### Step 3: Edge Discovery (Feature Connections)

For each pair of adjacent layers $(l, l+1)$, compute correlations between features:

**Cross-layer correlation**:
$$
r_{f_1, f_2} = \frac{\sum_{i=1}^{N} (z_{f_1,i} - \bar{z}_{f_1})(z_{f_2,i} - \bar{z}_{f_2})}{\sqrt{\sum_{i=1}^{N}(z_{f_1,i} - \bar{z}_{f_1})^2 \sum_{i=1}^{N}(z_{f_2,i} - \bar{z}_{f_2})^2}}
$$

where:
- $f_1$ = feature in layer $l$
- $f_2$ = feature in layer $l+1$
- $z_{f_1,i}, z_{f_2,i}$ = activations for sample $i$

**Edge creation**: If $|r_{f_1, f_2}| \geq \theta_{edge}$ (typically 0.01), create an edge between corresponding nodes.

**Interpretation**: Edges represent information flow - features that activate together across layers indicate a pathway through the model.

### 4.3 Separate Safe vs Toxic Circuit Discovery

The pipeline supports discovering **separate circuits** for safe and toxic samples:

**Rationale**: Safe and toxic samples may use different neural pathways, and comparing these circuits reveals the distinction.

**Method**:
1. Split encoded activations by refusal label
2. Discover circuit $C_{safe}$ using only safe samples
3. Discover circuit $C_{toxic}$ using only toxic samples
4. Compare $C_{safe}$ vs $C_{toxic}$ to identify refusal-specific features

**Key insight**: If refusal is modular, $C_{toxic}$ should differ significantly from $C_{safe}$. If monolithic, they should be similar.

---

## 5. Circuit Similarity and Overlap Calculation

### 5.1 Circuit Equalization

Before comparing circuits, they are **equalized** to ensure fair comparison:

**Problem**: Different categories may have different circuit sizes due to:
- Different sample sizes
- Different numbers of correlated features
- Statistical variation

**Solution**: Equalize all circuits to the same size by keeping only the top-$N$ most important features, where $N$ is the minimum circuit size across all categories.

**Mathematical formulation**:
For circuits $\{C_1, C_2, ..., C_k\}$ with sizes $\{|C_1|, |C_2|, ..., |C_k|\}$:
- Find minimum: $N_{min} = \min_i |C_i|$
- For each circuit $C_i$, keep top-$N_{min}$ nodes by importance: $C_i^{eq} = \text{TopN}(C_i, N_{min})$

**Rationale**: This ensures that comparisons reflect feature overlap, not circuit size differences.

### 5.2 Similarity Metrics

Circuit similarity is computed using **multiple complementary metrics**:

#### Metric 1: Node Overlap (Jaccard Similarity)

**Jaccard similarity** on node sets:
$$
S_{nodes}(C_1, C_2) = \frac{|V_1 \cap V_2|}{|V_1 \cup V_2|}
$$

where:
- $V_1, V_2$ = sets of node identifiers (feature + layer combinations)
- Intersection = nodes present in both circuits
- Union = nodes present in either circuit

**Range**: $[0, 1]$ where:
- $0$ = no shared nodes
- $1$ = identical node sets

#### Metric 2: Edge Overlap (Jaccard Similarity)

**Jaccard similarity** on edge sets:
$$
S_{edges}(C_1, C_2) = \frac{|E_1 \cap E_2|}{|E_1 \cup E_2|}
$$

where $E_1, E_2$ = sets of edges (source-target pairs).

**Interpretation**: Measures structural similarity - do circuits have similar connectivity patterns?

#### Metric 3: Importance Correlation

For nodes present in both circuits, compute correlation of importance scores:

**Pearson correlation**:
$$
r_{imp} = \text{corr}(\boldsymbol{\alpha}_1^{common}, \boldsymbol{\alpha}_2^{common})
$$

where:
- $\boldsymbol{\alpha}_1^{common}$ = importance scores of common nodes in circuit 1
- $\boldsymbol{\alpha}_2^{common}$ = importance scores of common nodes in circuit 2

**Interpretation**: Even if circuits share nodes, do they assign similar importance? High correlation indicates similar feature prioritization.

#### Combined Similarity Score

**Weighted combination**:
$$
S_{combined}(C_1, C_2) = 0.4 \cdot S_{nodes} + 0.3 \cdot S_{edges} + 0.3 \cdot |r_{imp}|
$$

**Rationale**:
- Node overlap (40%): Most important - do circuits use the same features?
- Edge overlap (30%): Structural similarity - do features connect similarly?
- Importance correlation (30%): Do circuits prioritize features similarly?

**Range**: $[0, 1]$ where:
- $0$ = completely different circuits
- $1$ = identical circuits

### 5.3 Similarity Matrix Construction

For $k$ categories, compute pairwise similarities:

**Similarity matrix** $S \in \mathbb{R}^{k \times k}$:
$$
S_{ij} = S_{combined}(C_i, C_j)
$$

Properties:
- **Symmetric**: $S_{ij} = S_{ji}$ (similarity is symmetric)
- **Diagonal**: $S_{ii} = 1.0$ (circuit identical to itself)

**Visualization**: Heatmap where color intensity represents similarity.

---

## 6. Analysis and Interpretation

### 6.1 Modularity Assessment

**Research Question**: Are refusal circuits monolithic (shared) or modular (category-specific)?

**Hypothesis Testing**:

**Monolithic hypothesis** ($H_0$): Refusal circuits are shared across categories
- **Prediction**: High average similarity between category pairs
- **Threshold**: Average similarity $> 0.5$ suggests monolithic behavior

**Modular hypothesis** ($H_1$): Refusal circuits are category-specific
- **Prediction**: Low average similarity between category pairs
- **Threshold**: Average similarity $< 0.5$ suggests modular behavior

**Mathematical formulation**:
$$
\bar{S} = \frac{1}{k(k-1)/2} \sum_{i < j} S_{ij}
$$

where the sum is over all unique category pairs.

### 6.2 Cross-Refusal Analysis

When separate safe and toxic circuits are discovered:

**Safe circuit similarity**: Compare safe circuits across categories
- High similarity = safe behavior uses shared pathways
- Low similarity = safe behavior is category-specific

**Toxic circuit similarity**: Compare toxic circuits across categories
- High similarity = refusal uses shared pathways (monolithic)
- Low similarity = refusal is category-specific (modular)

**Cross-refusal similarity**: Compare safe vs toxic circuits within the same category
- High similarity = safe and toxic use similar pathways
- Low similarity = refusal activates distinct pathways

**Diagonal analysis**: For category $c$, compare $C_c^{safe}$ vs $C_c^{toxic}$
- Measures how different refusal pathways are from normal processing
- High diagonal similarity = refusal is subtle modification
- Low diagonal similarity = refusal uses distinct mechanisms

### 6.3 Statistical Significance

**Bootstrap sampling**: To assess statistical significance:
1. Resample activations with replacement
2. Re-discover circuits
3. Recompute similarities
4. Repeat to get distribution of similarity scores

**Confidence intervals**: Report similarity with confidence intervals to assess robustness.

**Multiple comparisons correction**: When comparing many category pairs, apply Bonferroni correction or False Discovery Rate (FDR) control.

### 6.4 Interpretation Guidelines

**High similarity** ($S > 0.7$):
- Circuits share most important features
- Suggests shared mechanisms
- **Interpretation**: Monolithic behavior - refusal uses common pathways

**Medium similarity** ($0.3 < S < 0.7$):
- Partial overlap in features
- Some shared, some category-specific
- **Interpretation**: Hybrid behavior - both shared and specialized components

**Low similarity** ($S < 0.3$):
- Minimal feature overlap
- Distinct circuits per category
- **Interpretation**: Modular behavior - category-specific refusal mechanisms

**Zero similarity** ($S \approx 0$):
- No shared features
- Completely independent circuits
- **Interpretation**: Highly modular - each category uses unique pathways

---

## 7. Complete Experimental Pipeline

### Stage 1: Data Preparation
1. Load OR-Bench dataset (safe, hard, toxic splits)
2. Balance safe-toxic ratio per category (1:1)
3. Compute category weights for weighted sampling
4. Save balanced dataset and weights

### Stage 2: Activation Collection
1. Load transformer model
2. Register forward hooks at specified layers
3. Run inference on balanced dataset
4. Collect activations (residuals, MLPs, attention)
5. Detect refusal behavior from model outputs
6. Save activations and refusal labels

### Stage 3: SAE Training
1. Load activations from all categories
2. Create weighted sampler using category weights
3. Train single SAE per layer on all data
4. Monitor reconstruction loss and sparsity
5. Save trained SAEs and training history
6. Generate training plots

### Stage 4: Circuit Discovery
1. Load trained SAEs
2. Encode activations using SAEs
3. Compute feature-label correlations
4. Select important features (nodes)
5. Compute cross-layer feature correlations
6. Create edges between correlated features
7. Save discovered circuits

### Stage 5: Circuit Analysis
1. Load discovered circuits
2. Equalize circuits to same size
3. Compute pairwise similarities
4. Generate similarity matrices
5. Assess modularity (monolithic vs modular)
6. Create visualizations (heatmaps)
7. Generate statistical reports

---

## 8. Key Design Principles

### 8.1 Feature Space Comparability

**Principle**: All circuits must use the same feature space for fair comparison.

**Implementation**: Single SAE trained on all data ensures consistent feature meanings across categories and refusal types.

**Violation consequence**: If separate SAEs were used, feature indices would be incomparable, making similarity calculations meaningless.

### 8.2 Balanced Comparison

**Principle**: Comparisons must account for data imbalances.

**Implementation**: 
- Category-level: Weighted sampling during SAE training
- Circuit-level: Equalization before similarity computation
- Safe-toxic: Preserved 1:1 ratio (not balanced away)

**Rationale**: Ensures that differences reflect actual circuit differences, not data distribution artifacts.

### 8.3 Sparsity for Interpretability

**Principle**: Circuits must be sparse to be interpretable.

**Implementation**: 
- SAEs use top-$k$ sparsity (5% active features)
- Circuit discovery uses correlation thresholds
- Only most important features are included

**Rationale**: Dense circuits are uninterpretable; sparsity enables human understanding and analysis.

### 8.4 Statistical Rigor

**Principle**: All discoveries must be statistically validated.

**Implementation**:
- Correlation with $p$-value computation
- Threshold-based feature selection
- Bootstrap sampling for significance testing
- Multiple comparisons correction

**Rationale**: Prevents false discoveries and ensures robust conclusions.

---

## 9. Outcomes and Interpretation

### Observations
- low average similarity between toxic circuits across categories
- low average similarity between safe circuits across categories 
- low average similarity between safe and toxic circuits

This pattern indicates highly modular, category-specific processing.

1. Low toxic-toxic similarity across categories
- Each category uses a distinct refusal circuit
- No shared refusal mechanism across categories
- Refusal is category-specific, not monolithic
2. Low safe-to-safe similarity across categories
- Safe processing is also category-specific
- The model processes each category through specialized pathways even for normal (non-refusal) cases
- Category specialization exists beyond refusal
3. Low safe-to-toxic similarity
- Refusal uses different pathways than safe processing
- Refusal is a distinct mechanism, not a simple modification of normal processing
- Safe and toxic circuits within the same category are different

### Interpretation
This suggests:
- Category specialization: The model has learned category-specific representations and processing for both safe and refusal scenarios
- No universal refusal module: There isn’t a single shared refusal mechanism; refusal is implemented differently per category
- Distinct refusal mechanisms: Refusal isn’t just “turning off” normal processing; it activates category-specific refusal pathways
- Modular architecture: The model appears to have modular, category-specialized components rather than shared general-purpose circuits

### Implications
- Interventions must be category-specific: A universal intervention is unlikely to work across all categories
- Complex safety mechanisms: Refusal is not a simple binary switch but a set of category-specific mechanisms
- Model interpretability: Understanding refusal requires analyzing each category separately
- Potential for targeted attacks: Category-specific mechanisms might be more vulnerable to category-specific adversarial examples

This pattern points to a highly modular, category-specialized architecture where both normal and refusal processing are tailored to specific content categories.

---

## 10. Limitations and Considerations

### 10.1 Correlation vs Causation

**Limitation**: Correlation-based discovery identifies associations, not causal relationships.

**Mitigation**: 
- Statistical significance testing
- Cross-validation
- Ablation studies (future work)

### 10.2 SAE Feature Interpretability

**Limitation**: SAE features may not be directly interpretable without additional analysis.

**Mitigation**:
- Feature visualization
- Activation maximization
- Feature attribution analysis

### 10.3 Dataset Bias

**Limitation**: OR-Bench may not represent all refusal scenarios.

**Mitigation**:
- Multiple dataset validation
- Cross-model comparison
- Generalization testing

### 10.4 Computational Constraints

**Limitation**: Full-layer analysis requires significant computational resources.

**Mitigation**:
- Sparse layer sampling for initial experiments
- Efficient SAE training
- Distributed processing

---

## 11. Conclusion

This experimental pipeline provides a comprehensive framework for analyzing LLM refusal behavior through sparse feature circuits. By employing careful data balancing, SAE-based feature decomposition, correlation-based circuit discovery, and rigorous similarity analysis, the pipeline can determine whether refusal is monolithic or modular.

**Key innovations**:
1. Two-stage balancing (safe-toxic + category weighting)
2. Single SAE for feature space comparability
3. Correlation-based circuit discovery with statistical validation
4. Multi-metric similarity computation
5. Modularity assessment framework

**Research impact**: Understanding refusal mechanisms enables:
- Better model interpretability
- Targeted safety interventions
- Architecture improvements
- Regulatory compliance

The pipeline produces quantitative evidence for the monolithic vs modular hypothesis, advancing our understanding of how LLMs implement safety mechanisms.

