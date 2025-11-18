# SAE Training Plots Feature - Implementation Summary

## Overview

Added automatic visualization of SAE (Sparse Autoencoder) training metrics. Training loss plots are now automatically generated after each SAE training completes.

## What Was Added

### ✅ No existing plotting scripts found
The codebase had training history saved to JSON but no visualization.

### ✅ New Files Created

1. **`src/visualization/training_plots.py`** (164 lines)
   - `plot_sae_training_history()` - Full 2x2 subplot with all metrics
   - `plot_reconstruction_loss_only()` - Focused reconstruction loss plot
   - `plot_all_layers_comparison()` - Compare metrics across layers
   - `load_and_plot_from_json()` - Load and plot from saved history files

2. **`scripts/plot_sae_training.py`** (234 lines)
   - Standalone script to regenerate plots without retraining
   - Supports batch processing of multiple models/layers
   - Can generate layer comparison plots

3. **`docs/SAE_TRAINING_PLOTS.md`** (Comprehensive documentation)
   - Usage guide
   - API documentation
   - Examples
   - Troubleshooting

### ✅ Modified Files

1. **`src/visualization/__init__.py`**
   - Added imports for new plotting functions
   - Updated `__all__` to export new functions

2. **`src/sae/sae_manager.py`**
   - Added import for plotting functions (line 17)
   - Integrated automatic plotting after training (lines 271-280)
   - Creates `training_plots/` subdirectory for each model
   - Gracefully handles plotting errors without failing training

3. **`README.md`**
   - Added documentation about automatic plotting
   - Added section on `plot_sae_training.py` utility script
   - Included usage examples

## Features

### Automatic Plotting
- **Trigger**: Automatically runs after each SAE training completes
- **Location**: `results/saes/{model_name}/training_plots/`
- **No action required**: Just run normal SAE training

### Generated Plots Per Layer
1. **Full Training History** - 2x2 grid showing:
   - Reconstruction Loss (primary metric)
   - Total Loss
   - Sparsity Loss
   - L0 Norm (active features count)

2. **Reconstruction Loss Focus** - Single plot with:
   - Epoch vs reconstruction loss curve
   - Final loss annotation
   - Cleaner visualization for presentations

### Standalone Script Features
```bash
# Plot all models
python scripts/plot_sae_training.py --sae-dir results/saes

# Plot specific model
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf"

# Add layer comparison plots
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf" --comparison

# Plot single history file
python scripts/plot_sae_training.py --history-file path/to/history.json
```

## Integration Points

### Where Plots Are Generated

**Automatic (during training):**
```python
# In src/sae/sae_manager.py, after line 269
history = trainer.train(dataloader, epochs)
trainer.save(sae_path)
with open(model_sae_dir / f"{layer}_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

# NEW: Generate training plots
print("\nGenerating training plots...")
plots_dir = model_sae_dir / "training_plots"
plots_dir.mkdir(exist_ok=True)

try:
    plot_sae_training_history(history, model_name, layer, plots_dir)
    plot_reconstruction_loss_only(history, model_name, layer, plots_dir)
except Exception as e:
    print(f"  Warning: Could not generate plots: {e}")
```

**Manual (standalone script):**
```bash
python scripts/plot_sae_training.py --sae-dir results/saes
```

## File Structure

```
results/saes/
├── {model_name}/
│   ├── training_plots/                              # NEW
│   │   ├── {model}_{layer}_training_history.png     # NEW - Full metrics
│   │   ├── {model}_{layer}_reconstruction_loss.png  # NEW - Focus plot
│   │   ├── {model}_all_layers_reconstruction_loss.png # NEW - Comparison
│   │   └── ...
│   ├── {layer}_sae.pt                               # Existing
│   └── {layer}_training_history.json                # Existing
```

## Plot Details

### Styling
- High resolution (150 DPI)
- Seaborn whitegrid style
- Color-coded metrics
- Grid lines for readability
- Annotations (e.g., final loss values)

### Metrics Plotted
1. **Reconstruction Loss** - How well SAE reconstructs activations (lower = better)
2. **Total Loss** - Combined loss with regularization
3. **Sparsity Loss** - L1 penalty on features (currently 0 due to structural sparsity)
4. **L0 Norm** - Number of active features (measures sparsity)

## Testing Checklist

- [x] Created visualization module (`training_plots.py`)
- [x] Created standalone script (`plot_sae_training.py`)
- [x] Integrated into SAE training pipeline
- [x] Updated module exports (`__init__.py`)
- [x] Updated README with usage instructions
- [x] Created comprehensive documentation
- [x] Made script executable
- [x] Verified no linting errors
- [x] Graceful error handling (plotting doesn't break training)

## Usage Example

### Quick Start
Just run normal SAE training:
```bash
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml
```

Plots will automatically appear in:
```
results/saes/{model_name}/training_plots/
```

### Regenerate Plots
If you want different styling or need to regenerate:
```bash
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf"
```

### Compare Layers
Generate overlay plots comparing all layers:
```bash
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf" --comparison
```

## Benefits

1. **Immediate Feedback** - See training convergence without manual steps
2. **Quality Assurance** - Quickly identify training issues
3. **Reproducibility** - Saved plots for reports and papers
4. **Flexibility** - Standalone script for custom plotting
5. **Non-Intrusive** - Doesn't break existing training if plotting fails

## Dependencies

All dependencies already present in the project:
- `matplotlib` - Core plotting
- `seaborn` - Styling
- `json` - Loading history files
- `pathlib` - File handling

## Next Steps

To use this feature:

1. **Existing training history**: If you already have trained SAEs with history files:
   ```bash
   python scripts/plot_sae_training.py --sae-dir results/saes
   ```

2. **New training**: Just run training normally:
   ```bash
   python scripts/train_saes.py --config configs/sae/sae_balanced.yaml
   ```
   Plots will be automatically generated!

3. **Check results**:
   ```bash
   ls results/saes/*/training_plots/
   ```

## Files Changed Summary

**New Files (3):**
- `src/visualization/training_plots.py`
- `scripts/plot_sae_training.py`
- `docs/SAE_TRAINING_PLOTS.md`

**Modified Files (3):**
- `src/visualization/__init__.py` (added exports)
- `src/sae/sae_manager.py` (integrated plotting)
- `README.md` (added documentation)

**Lines Added:** ~600 lines of code and documentation

---

**Status**: ✅ Complete and ready to use!

