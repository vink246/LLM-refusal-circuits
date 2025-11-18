# All Decoder Layer Collection - Implementation Guide

## Overview

The activation collection system has been enhanced to support collecting activations from **all decoder blocks** in transformer models, not just manually specified layers. This enables comprehensive layer-by-layer analysis of refusal circuits without tedious manual configuration.

## What Changed

### Before
```yaml
# Had to manually list every layer
activation_layers:
  - "residuals_0"
  - "residuals_5"
  - "residuals_10"
  - "residuals_15"
  - "residuals_20"
  - "residuals_25"
  - "residuals_30"
```

### After
```yaml
# Simple shorthand
activation_layers:
  - "all"  # Automatically expands to all 32 layers!
```

## New Layer Specification Formats

### 1. **"all" Shorthand**
Simplest way to collect from all residual layers:

```yaml
activation_layers:
  - "all"
```

Expands to: `residuals_0`, `residuals_1`, ..., `residuals_31` (for 32-layer models)

### 2. **Typed "all"**
Collect all layers of a specific type:

```yaml
activation_layers:
  - "residuals_all"  # All residual stream layers
  - "mlp_all"        # All MLP layers
  - "attention_all"  # All attention layers
```

### 3. **Range Notation**
Collect a continuous range of layers:

```yaml
activation_layers:
  - "residuals_0-31"   # First through last layer
  - "residuals_10-21"  # Middle layers only
  - "mlp_5-15"         # MLP layers 5 through 15
```

### 4. **Mixed Specifications**
Combine different formats:

```yaml
activation_layers:
  - "residuals_0-10"   # Early layers
  - "residuals_20-31"  # Late layers
  - "mlp_15"           # Single middle MLP
```

### 5. **Backward Compatible**
Old explicit lists still work:

```yaml
activation_layers:
  - "residuals_0"
  - "residuals_5"
  - "residuals_10"
```

## Files Modified

### Core Implementation

**`src/models/model_wrapper.py`**
- Added `get_num_layers()` - Detects total decoder layers
- Added `expand_layer_specs()` - Expands shorthand notation
- Enhanced `setup_activation_hooks()` - Uses expansion, adds memory warnings

**`src/models/inference_pipeline.py`**
- Enhanced `load_model()` - Shows expansion info and memory estimates

### Configuration Files

**New configs:**
- `configs/models/llama2_7b_all_layers.yaml` - Llama-2-7B with all 32 layers
- `configs/models/mistral_7b_all_layers.yaml` - Mistral-7B with all 32 layers
- `configs/models/llama2_7b_middle_layers.yaml` - Focused middle layers (10-21)

**Updated:**
- `configs/models/README.md` - Comprehensive documentation of layer specifications
- `README.md` - Added examples and memory notes

### Tests

**New test files:**
- `tests/test_layer_expansion.py` - Full unit tests (requires torch)
- `tests/test_layer_expansion_simple.py` - Standalone validation (no dependencies)

## Usage Examples

### Quick Start: All Layers

```bash
# Collect from ALL 32 decoder layers
python scripts/collect_activations.py \
    --config configs/models/llama2_7b_all_layers.yaml

# Train SAEs (one per layer automatically)
python scripts/train_saes.py \
    --config configs/sae/sae_balanced.yaml \
    --model-name "meta-llama/Llama-2-7b-chat-hf"

# Discover circuits (uses all trained SAEs)
python scripts/discover_circuits.py \
    --config configs/circuits/discovery_separate_safe_toxic.yaml
```

### Focused Analysis: Middle Layers

```bash
# Collect from middle layers where refusal often emerges
python scripts/collect_activations.py \
    --config configs/models/llama2_7b_middle_layers.yaml
```

### Custom Configuration

Create your own config with specific layer ranges:

```yaml
# custom_layers.yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  device: "cuda"
  torch_dtype: "float16"
  
  activation_layers:
    - "residuals_0-5"     # Early layers (6 layers)
    - "residuals_15-20"   # Middle layers (6 layers)
    - "residuals_28-31"   # Late layers (4 layers)
    # Total: 16 layers

inference:
  batch_size: 3  # Adjust based on GPU memory
  max_new_tokens: 20
  save_activations: true

data:
  path: "data/raw/or-bench"
  categories: []
  category_balance_strategy: "use_all"

output:
  result_dir: "results"
```

## Memory Considerations

### GPU Memory (During Inference)

| Configuration | Layers | Batch Size | GPU Memory |
|--------------|---------|------------|------------|
| Sparse (default) | 7 | 4 | ~8-10 GB |
| Middle layers | 12 | 4 | ~10-12 GB |
| All layers | 32 | 2 | ~12-16 GB |
| All types (×3) | 96 | 1 | ~18-24 GB |

**If OOM errors occur:**
1. Reduce `batch_size` (e.g., 4 → 2 → 1)
2. Use `torch_dtype: "float16"` (default)
3. Collect fewer layers

### Storage (Saved Activations)

Per 1000 samples (approximate):

- **Sparse (7 layers)**: ~200 MB
- **Middle (12 layers)**: ~350 MB
- **All (32 layers)**: ~1 GB
- **All types (96 layers)**: ~3 GB

For full OR-Bench dataset (~10K samples per category × multiple categories):
- All layers: ~10-30 GB total storage

## Integration with Pipeline

The layer expansion works seamlessly with the entire pipeline:

### 1. Activation Collection
```bash
python scripts/collect_activations.py --config configs/models/llama2_7b_all_layers.yaml
```

Output: `results/activations/{model}_{category}_activations.pt`
- Contains activations for all specified layers
- Automatically expands "all" to 32 layers

### 2. SAE Training
```bash
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml
```

Behavior:
- Reads the same `activation_layers` from activation files
- Trains one SAE per layer automatically
- For 32 layers: trains 32 SAEs
- Each SAE saved to: `results/saes/{model}/{layer}_sae.pt`

### 3. Circuit Discovery
```bash
python scripts/discover_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml
```

Behavior:
- Uses all trained SAEs
- Discovers circuits layer-by-layer
- Produces comprehensive circuit analysis across all layers

### 4. Circuit Analysis
```bash
python scripts/analyze_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml
```

Benefits:
- Compare circuit emergence across all 32 layers
- Identify which layers contain refusal circuits
- Analyze how circuits evolve through the model

## Technical Details

### Auto-Detection

The system automatically detects the number of decoder layers:

```python
# In ModelWrapper
def get_num_layers(self) -> int:
    """Get the number of decoder layers in the model."""
    if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
        raise ValueError(f"Model architecture not supported.")
    return len(self.model.model.layers)
```

Works with any model that has `model.model.layers` structure (most decoder-only transformers).

### Expansion Logic

The expansion happens before hooks are registered:

```python
def expand_layer_specs(self, layers: List[str]) -> List[str]:
    """Expand shorthand notation to explicit layer specs."""
    num_layers = self.get_num_layers()
    expanded = []
    
    for layer_spec in layers:
        if layer_spec == "all":
            expanded.extend([f"residuals_{i}" for i in range(num_layers)])
        elif layer_spec == "residuals_0-10":
            expanded.extend([f"residuals_{i}" for i in range(11)])
        # ... more expansion logic
    
    return expanded
```

### Memory Warnings

Automatic warnings when collecting many layers:

```python
if len(expanded_layers) > 20:
    print(f"  ⚠️  Warning: Collecting from {len(expanded_layers)} layers will use significant memory")
    print(f"     Consider reducing batch_size in inference config")
```

## Validation

Run validation tests to verify the expansion logic:

```bash
# Simple validation (no dependencies)
python3 tests/test_layer_expansion_simple.py

# Full unit tests (requires conda environment with torch)
conda activate llm_refusal_env
python tests/test_layer_expansion.py
```

Expected output:
```
✓ All validation tests passed!
The layer expansion logic is working correctly.
```

## Use Cases

### 1. Comprehensive Layer Analysis

**Goal**: Analyze refusal circuits across all layers

**Config**: `llama2_7b_all_layers.yaml`
```yaml
activation_layers:
  - "all"  # All 32 layers
```

**Benefits**:
- Complete picture of circuit emergence
- Identify which layers matter most
- Publication-quality comprehensive analysis

### 2. Targeted Middle Layer Analysis

**Goal**: Focus on layers where refusal behavior emerges

**Config**: `llama2_7b_middle_layers.yaml`
```yaml
activation_layers:
  - "residuals_10-21"  # 12 middle layers
```

**Benefits**:
- Reduced computational cost
- Focused on relevant layers
- Good balance of coverage vs. efficiency

### 3. Early vs Late Layer Comparison

**Goal**: Compare early and late layer circuits

**Custom config**:
```yaml
activation_layers:
  - "residuals_0-10"   # Early layers
  - "residuals_22-31"  # Late layers
```

**Benefits**:
- Test hypothesis about layer specialization
- Reduced storage compared to all layers
- Direct comparison of circuit types

### 4. Multi-Component Analysis

**Goal**: Analyze residuals, MLPs, and attention separately

**Custom config**:
```yaml
activation_layers:
  - "residuals_all"   # 32 layers
  - "mlp_all"         # 32 layers
  - "attention_all"   # 32 layers
  # Total: 96 layers!
```

**Benefits**:
- Decompose refusal circuits by component
- Understand role of MLPs vs attention
- Research-grade detailed analysis

**Warning**: Very memory-intensive!

## Troubleshooting

### Issue: OOM (Out of Memory) Errors

**Symptoms**: CUDA out of memory during inference

**Solutions**:
1. Reduce `batch_size` in config:
   ```yaml
   inference:
     batch_size: 1  # Minimum
   ```

2. Collect fewer layers:
   ```yaml
   activation_layers:
     - "residuals_10-21"  # Instead of "all"
   ```

3. Use smaller model or more GPU memory

### Issue: Slow Activation Collection

**Symptoms**: Takes hours to collect activations

**Solutions**:
1. Use sparse layer sampling for initial experiments
2. Collect all layers only for final analysis
3. Process categories separately
4. Use GPU acceleration (ensure CUDA is available)

### Issue: Large Storage Requirements

**Symptoms**: Running out of disk space

**Solutions**:
1. Collect fewer categories:
   ```yaml
   data:
     categories: ["hate", "violence"]  # Subset only
   ```

2. Use fewer samples:
   ```yaml
   data:
     max_samples_per_category: 1000  # Limit samples
   ```

3. Compress old results or use external storage

### Issue: Layer Specification Not Working

**Symptoms**: Warnings about unknown layer specifications

**Solutions**:
1. Verify YAML syntax (use quotes around specifications)
2. Check model has `model.model.layers` structure
3. Run validation tests:
   ```bash
   python3 tests/test_layer_expansion_simple.py
   ```

## Performance Tips

1. **Start Sparse**: Use default sparse configs (7 layers) for initial experiments

2. **Then Go Comprehensive**: Use all-layer configs for final analysis after you've validated everything works

3. **Use Range Notation**: More readable than explicit lists:
   ```yaml
   # Good
   activation_layers:
     - "residuals_10-21"
   
   # Tedious
   activation_layers:
     - "residuals_10"
     - "residuals_11"
     - "residuals_12"
     # ... 9 more lines
   ```

4. **Monitor First Run**: Watch GPU memory and adjust `batch_size` accordingly

5. **Parallelize Categories**: Run different categories on different GPUs if available

## Research Applications

### Layer-by-Layer Circuit Evolution

With all-layer collection, you can analyze:
- Where in the model do refusal circuits first appear?
- Which layers have the strongest refusal signals?
- Do circuits evolve or remain stable across layers?

### Component Decomposition

Collecting residuals, MLPs, and attention separately:
- Are refusal circuits primarily in MLPs or attention?
- How do components interact to produce refusal?
- Can we ablate specific components to disable refusal?

### Cross-Model Comparison

Collecting all layers from multiple models:
- Do Llama and Mistral use same layers for refusal?
- Are refusal circuits universal across architectures?
- How does model size affect circuit distribution?

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Automatic batch_size selection based on available GPU memory
- [ ] Streaming collection for very large datasets
- [ ] Distributed collection across multiple GPUs
- [ ] Activation compression for reduced storage
- [ ] Selective layer collection based on SAE reconstruction quality
- [ ] Real-time monitoring dashboard during collection

## Related Documentation

- `configs/models/README.md` - Detailed layer specification syntax
- `README.md` - Main project overview
- `docs/SAE_TRAINING_PLOTS.md` - SAE training visualization
- Test files for validation examples

## Summary

The all-layer collection enhancement makes comprehensive circuit analysis practical:

✅ **Automatic**: No manual layer enumeration  
✅ **Flexible**: Range notation, typed specifications  
✅ **Backward Compatible**: Old configs still work  
✅ **Memory Aware**: Automatic warnings and guidance  
✅ **Well Tested**: Validated expansion logic  
✅ **Integrated**: Works seamlessly with entire pipeline  

You can now analyze refusal circuits across every decoder layer with just `activation_layers: ["all"]`!

