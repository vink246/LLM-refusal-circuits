# Model Configuration Files

Configuration files for model inference and activation collection.

## Available Configurations

### Sparse Layer Sampling (Default)

-   **`llama2_7b.yaml`** - Llama-2-7B with ~7 layers (0, 5, 10, 15, 20, 25, 30)
-   **`mistral_7b.yaml`** - Mistral-7B with ~7 layers (0, 5, 10, 15, 20, 25, 30)

Good for: Quick experiments, limited memory, exploratory analysis

### All Layer Collection (Comprehensive)

-   **`llama2_7b_all_layers.yaml`** - Llama-2-7B with ALL 32 decoder layers
-   **`mistral_7b_all_layers.yaml`** - Mistral-7B with ALL 32 decoder layers

Good for: Comprehensive analysis, layer-by-layer circuit discovery, publication-quality results

⚠️ **Memory Note**: Requires ~2-3x more storage and reduces batch_size to 2

### Focused Layer Collection (Middle Layers)

-   **`llama2_7b_middle_layers.yaml`** - Llama-2-7B layers 10-21 (middle layers)

Good for: Targeting refusal circuits (often in middle-to-late layers), balanced approach

## Layer Specification Format

The `activation_layers` field supports multiple formats:

### 1. **Explicit List** (Original)

```yaml
activation_layers:
    - "residuals_0"
    - "residuals_5"
    - "residuals_10"
    - "residuals_15"
```

### 2. **"all" Shorthand** (NEW)

```yaml
activation_layers:
    - "all" # Expands to all residual layers (residuals_0 to residuals_31)
```

### 3. **Typed "all"** (NEW)

```yaml
activation_layers:
    - "residuals_all" # All residual layers (32 layers)
    - "mlp_all" # All MLP layers (32 layers)
    - "attention_all" # All attention layers (32 layers)
```

### 4. **Range Notation** (NEW)

```yaml
activation_layers:
    - "residuals_0-31" # Layers 0 through 31 (inclusive)
    - "residuals_10-21" # Middle layers only
    - "mlp_5-15" # MLP layers 5 through 15
```

### 5. **Mixed Specifications**

```yaml
activation_layers:
    - "residuals_0-10" # First 11 residual layers
    - "residuals_20-31" # Last 12 residual layers
    - "mlp_15" # Single MLP layer 15
    - "attention_15" # Single attention layer 15
```

### 6. **Comprehensive Collection** (⚠️ Very Large!)

```yaml
activation_layers:
    - "residuals_all" # 32 layers
    - "mlp_all" # 32 layers
    - "attention_all" # 32 layers
    # Total: 96 layer activations!
```

## Layer Types

### `residuals_N`

-   Residual stream input to decoder layer N
-   Contains information flowing through the model
-   **Recommended** for circuit analysis

### `mlp_N`

-   MLP (feed-forward) output at decoder layer N
-   Useful for analyzing feature transformation

### `attention_N`

-   Self-attention output at decoder layer N
-   Useful for analyzing token interactions

## Configuration Fields

```yaml
model:
    name: "meta-llama/Llama-2-7b-chat-hf" # HuggingFace model name
    device: "cuda" # "cuda" or "cpu"
    torch_dtype: "float16" # "float16" or "float32"
    trust_remote_code: false # For non-standard models
    cache_dir: null # Optional: HF cache directory
    activation_layers: [...] # Layer specifications (see above)

inference:
    batch_size: 4 # Samples per batch (reduce for more layers)
    max_new_tokens: 20 # Tokens to generate per prompt
    save_activations: true # Whether to save activations

data:
    path: "data/raw/or-bench" # Dataset path
    categories: [] # Empty = all, or list specific categories
    category_balance_strategy: "use_all" # "use_all" or "equalize"

output:
    result_dir: "results" # Output directory
```

## Memory Guidelines

### GPU Memory (During Inference)

-   **Sparse (7 layers)**: ~8-10 GB (batch_size=4)
-   **Middle (12 layers)**: ~10-12 GB (batch_size=4)
-   **All (32 layers)**: ~12-16 GB (batch_size=2)

If you get OOM errors:

1. Reduce `batch_size` (e.g., from 4 to 2 or 1)
2. Use `torch_dtype: "float16"` instead of "float32"
3. Reduce number of layers collected

### Storage (Saved Activations)

Per 1000 samples (approximate):

-   **Sparse (7 layers)**: ~200 MB
-   **Middle (12 layers)**: ~350 MB
-   **All (32 layers)**: ~1 GB

## Usage Examples

### Quick Experiment (Sparse Layers)

```bash
python scripts/collect_activations.py --config configs/models/llama2_7b.yaml
```

### Comprehensive Analysis (All Layers)

```bash
python scripts/collect_activations.py --config configs/models/llama2_7b_all_layers.yaml
```

### Targeted Analysis (Middle Layers)

```bash
python scripts/collect_activations.py --config configs/models/llama2_7b_middle_layers.yaml
```

### Custom Configuration

Create your own YAML file with specific layers:

```yaml
model:
    name: "meta-llama/Llama-2-7b-chat-hf"
    activation_layers:
        - "residuals_0-5" # Early layers
        - "residuals_28-31" # Late layers
        - "mlp_15" # Single middle MLP layer
```

## Integration with SAE Training

The SAE training pipeline (`train_saes.py`) uses the same `activation_layers` specification:

1. **Collect activations** with desired layers
2. **Train SAEs** - one SAE per layer automatically
3. **Discover circuits** - circuits use SAE features from collected layers

Example workflow:

```bash
# Step 1: Collect from all layers
python scripts/collect_activations.py --config configs/models/llama2_7b_all_layers.yaml

# Step 2: Train SAEs (automatically trains one per layer)
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml --model-name "meta-llama/Llama-2-7b-chat-hf"

# Step 3: Discover circuits (uses all trained SAEs)
python scripts/discover_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml
```

## Tips

1. **Start sparse** - Use default configs first to verify everything works
2. **Then go comprehensive** - Use `all_layers` configs for final analysis
3. **Monitor memory** - Watch GPU memory during first run with new config
4. **Adjust batch_size** - Reduce if you see OOM errors
5. **Storage planning** - All layers with many samples needs significant disk space

## Model-Specific Notes

### Llama-2-7B

-   32 decoder layers (0-31)
-   Hidden dim: 4096
-   Good starting model for experiments

### Mistral-7B

-   32 decoder layers (0-31)
-   Hidden dim: 4096
-   Sliding window attention (doesn't affect activation collection)

### Adding New Models

To add a new model:

1. Create new YAML file (e.g., `my_model.yaml`)
2. Set `model.name` to HuggingFace model identifier
3. Verify model has `model.model.layers` structure (most decoder-only transformers do)
4. Choose layer specifications based on analysis needs

The layer expansion works automatically for any model with the standard decoder architecture!
