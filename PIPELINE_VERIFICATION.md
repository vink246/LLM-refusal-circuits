# Pipeline Statement Verification

This document verifies the accuracy of statements about the pipeline implementation.

## Statement-by-Statement Analysis

### ✅ Statement 1: "we built a discovery pipeline on the OR-Bench dataset"

**Status: CORRECT**

-   Verified: `src/data/orbench_loader.py` loads OR-Bench CSV files
-   Files: `or-bench-80k.csv`, `or-bench-hard-1k.csv`, `or-bench-toxic.csv`

### ✅ Statement 2: "Because the data was heavily skewed towards harmful prompts and certain categories"

**Status: PARTIALLY CORRECT - Needs Clarification**

-   **Issue**: The statement says "skewed towards harmful prompts" but OR-Bench actually has FAR MORE safe samples than toxic samples
-   **Reality**:
    -   Safe samples: ~80,000 (or-bench-80k.csv) + ~1,000 (or-bench-hard-1k.csv)
    -   Toxic samples: Much smaller per category (varies by category)
    -   The imbalance is actually **towards safe prompts**, not harmful prompts
-   **Recommendation**: Change to "skewed towards safe prompts" or "heavily imbalanced between safe and toxic prompts"

### ✅ Statement 3: "we implemented a balancing strategy to handle class and category imbalances before training"

**Status: CORRECT**

-   Verified: `src/data/data_processor.py` implements `prepare_balanced_dataset()`
-   Strategy: Matches safe samples to toxic sample count per category
-   Two-stage approach: Stage 1 balances data, Stage 2 trains single SAE on all data

### ✅ Statement 4: "we trained Sparse Autoencoders (SAEs) on the residual streams of the models' later layers, specifically layers 21 through 31"

**Status: CORRECT**

-   Verified: `configs/models/llama2_7b.yaml` specifies layers 21-31
-   Verified: `scripts/evaluate_circuits.py` line 72: `layers = [f"residuals_{i}" for i in range(21, 32)]`
-   Verified: SAEs are trained on residual stream activations (not MLP or attention outputs)

### ✅ Statement 5: "By decomposing dense, polysemantic activations into a dictionary of sparse, monosemantic features"

**Status: CORRECT**

-   Verified: SAE architecture in `src/sae/sae_model.py` uses encoder-decoder structure
-   Verified: Sparsity enforced via top-k activation (only k% features active)
-   Verified: SAE transforms dense activations (4096 dims) into sparse features (32,768 features)

### ⚠️ Statement 6: "We defined the nodes of these circuits by isolating SAE features that showed a statistically significant Pearson correlation with refusal labels"

**Status: PARTIALLY CORRECT - Needs Clarification**

-   **What the code does**:
    -   Computes Pearson correlation AND p-value: `corr, p_value = stats.pearsonr(feat_values, labels_np)`
    -   BUT: Uses correlation magnitude threshold, NOT p-value threshold
    -   Code: `if abs(corr) >= self.circuit_config.node_threshold` (line 276)
    -   The p-value is computed but not used for filtering
-   **Issue**: The statement says "statistically significant" but the code uses correlation magnitude threshold, not p-value threshold
-   **Recommendation**: Either:
    1. Change statement to: "SAE features that showed a strong Pearson correlation (above threshold) with refusal labels"
    2. OR update code to actually use p-value: `if abs(corr) >= threshold and p_value < 0.05`

### ✅ Statement 7: "edges were established based on correlations between layers"

**Status: CORRECT**

-   Verified: `src/circuits/discovery.py` lines 336-350
-   Edges computed using Pearson correlation between features in adjacent layers
-   Threshold: `edge_importance >= self.circuit_config.edge_threshold`

### ✅ Statement 8: "this framework constructs Directed Acyclic Graphs to trace the probable flow of information"

**Status: CORRECT**

-   Verified: Edges only connect from earlier layers to later layers (lines 325-327)
-   Structure: `for i in range(len(layers_list) - 1): layer1 = layers_list[i]; layer2 = layers_list[i + 1]`
-   This ensures no cycles (DAG property)
-   Direction: Early layers → Later layers (information flow)

### ✅ Statement 9: "A primary goal of this analysis was to distinguish between circuits for safe and toxic inputs"

**Status: CORRECT**

-   Verified: `separate_safe_toxic` flag in configs
-   Verified: `configs/circuits/discovery_separate_safe_toxic_llama.yaml` sets `separate_safe_toxic: true`
-   Verified: Code splits data into safe/toxic and discovers separate circuits (lines 105-144 in discovery.py)

### ✅ Statement 10: "Faithfulness tests the circuit's ability to reproduce model behavior in isolation"

**Status: CORRECT**

-   Verified: `src/circuits/evaluation.py` lines 307-311
-   Faithfulness = F(C) where everything NOT in circuit is ablated
-   Formula: `(F(C) - F(Empty)) / (F(M) - F(Empty))`
-   Measures: How well circuit alone reproduces behavior

### ✅ Statement 11: "Completeness measures how much behavior persists when the circuit is removed"

**Status: CORRECT**

-   Verified: `src/circuits/evaluation.py` lines 313-317
-   Completeness = F(M\C) / F(M) where circuit features are ablated
-   Measures: How much behavior remains after removing circuit

### ✅ Statement 12: "We observed a critical inverse relationship here where removing a circuit that captures more necessary components causes a steeper drop in model performance"

**Status: CORRECT**

-   Verified: This is exactly how completeness works
-   As circuit captures more necessary components → removing it removes more necessary components → F(M\C) decreases → Completeness decreases
-   This is the expected behavior and validates the circuit discovery

## Summary of Issues

### Critical Issues (Need Fixing):

1. **Statement 2**: Says "skewed towards harmful prompts" but should be "skewed towards safe prompts" or "imbalanced between safe and toxic"
2. **Statement 6**: Says "statistically significant" but code uses correlation magnitude threshold, not p-value threshold

### Minor Clarifications:

-   Statement 6 could be more precise about using correlation magnitude vs statistical significance
-   The p-value is computed but not used for filtering nodes

## Recommendations

### For Statement 2:

**Current**: "Because the data was heavily skewed towards harmful prompts..."
**Suggested**: "Because the data was heavily imbalanced between safe and toxic prompts, with far more safe samples than toxic samples..."

### For Statement 6:

**Option A** (Keep current code, update statement):
**Current**: "statistically significant Pearson correlation"
**Suggested**: "strong Pearson correlation (above threshold) with refusal labels"

**Option B** (Update code to use p-value):
Add p-value filtering:

```python
if abs(corr) >= self.circuit_config.node_threshold and p_value < 0.05:
```

## Overall Assessment

**Accuracy: 10/12 statements fully correct, 2 need clarification**

The pipeline description is largely accurate. The two issues are:

1. Minor factual error about data skew direction
2. Terminology mismatch (says "statistically significant" but uses correlation magnitude)

Both are easily fixable either by updating the statements or the code.
