# SAE Training Plots Documentation

## Overview

Automatic visualization of SAE (Sparse Autoencoder) training metrics has been integrated into the training pipeline. Training loss plots are now automatically generated after each SAE training completes, providing immediate visual feedback on training progress and convergence.

## Features

### Automatic Plot Generation

When you run SAE training, plots are automatically generated and saved without any additional steps:

```bash
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml
```

**Output location**: `results/saes/{model_name}/training_plots/`

### Generated Plots

For each model and layer, the following plots are automatically created:

1. **Full Training History** (`{model}_{layer}_training_history.png`)
   - 2x2 subplot grid showing:
     - Reconstruction Loss
     - Total Loss
     - Sparsity Loss
     - L0 Norm (Active Features)

2. **Reconstruction Loss Only** (`{model}_{layer}_reconstruction_loss.png`)
   - Focused plot of reconstruction loss over epochs
   - Includes final loss value annotation

### Plot Details

Each plot includes:
- **Epoch vs Metric**: X-axis shows training epochs, Y-axis shows the metric value
- **Grid lines**: For easier reading
- **High resolution**: 150 DPI for publication-quality images
- **Clean styling**: Using Seaborn's whitegrid style

## Standalone Plotting Script

A standalone script is provided to regenerate plots from saved training history files without retraining:

### Basic Usage

```bash
# Plot all models and all layers
python scripts/plot_sae_training.py --sae-dir results/saes

# Plot specific model
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf"

# Plot specific model with layer comparison plots
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf" --comparison

# Plot from specific training history file
python scripts/plot_sae_training.py --history-file results/saes/model/layer_training_history.json

# Custom output directory
python scripts/plot_sae_training.py --sae-dir results/saes --output-dir my_plots/
```

### Layer Comparison Plots

When using the `--comparison` flag, additional plots are generated that overlay all layers for a model:

- `{model}_all_layers_reconstruction_loss.png` - Compare reconstruction loss across layers
- `{model}_all_layers_total_loss.png` - Compare total loss across layers
- `{model}_all_layers_l0_norm.png` - Compare sparsity (L0 norm) across layers

This helps identify:
- Which layers converge faster
- Which layers achieve better reconstruction
- Layer-specific training dynamics

## File Organization

```
results/saes/
├── {model_name}/
│   ├── training_plots/                    # All training plots
│   │   ├── {model}_{layer}_training_history.png
│   │   ├── {model}_{layer}_reconstruction_loss.png
│   │   ├── {model}_all_layers_reconstruction_loss.png  # (if --comparison used)
│   │   └── ...
│   ├── {layer}_sae.pt                     # Trained SAE weights
│   └── {layer}_training_history.json      # Training metrics (raw data)
```

## Training History Format

Training history is saved as JSON with the following structure:

```json
{
  "total_loss": [0.245, 0.198, 0.156, ...],
  "reconstruction_loss": [0.245, 0.198, 0.156, ...],
  "sparsity_loss": [0.0, 0.0, 0.0, ...],
  "l0_norm": [410.5, 405.2, 398.7, ...]
}
```

Each array contains one value per epoch, allowing you to:
- Track training convergence
- Identify early stopping points
- Compare training dynamics across layers/models
- Regenerate plots with different styling

## Programmatic Usage

You can also use the plotting functions directly in your Python code:

```python
from src.visualization.training_plots import (
    plot_sae_training_history,
    plot_reconstruction_loss_only,
    plot_all_layers_comparison,
    load_and_plot_from_json
)
from pathlib import Path

# Load and plot from JSON file
history_file = Path("results/saes/model/layer10_training_history.json")
model_name = "meta-llama/Llama-2-7b-hf"
layer = "layer10"
output_dir = Path("my_plots/")

load_and_plot_from_json(history_file, model_name, layer, output_dir)

# Or use history dict directly
import json
with open(history_file) as f:
    history = json.load(f)

plot_sae_training_history(history, model_name, layer, output_dir)
plot_reconstruction_loss_only(history, model_name, layer, output_dir)

# Compare multiple layers
histories = {
    "layer10": json.load(open("results/saes/model/layer10_training_history.json")),
    "layer15": json.load(open("results/saes/model/layer15_training_history.json")),
    "layer20": json.load(open("results/saes/model/layer20_training_history.json")),
}
plot_all_layers_comparison(histories, model_name, output_dir, metric='reconstruction_loss')
```

## Integration Points

The plotting functionality is integrated at the following points:

1. **`src/sae/sae_manager.py`** (Line 271-280)
   - Automatically called after each SAE training completes
   - Generates both full history and reconstruction-only plots
   - Continues training pipeline even if plotting fails (with warning)

2. **`src/visualization/training_plots.py`**
   - Core plotting functions
   - Handles different plot types and configurations
   - Can be imported and used independently

3. **`scripts/plot_sae_training.py`**
   - Standalone script for regenerating plots
   - Supports batch processing of multiple models/layers
   - Useful for custom plotting without retraining

## Metrics Explained

### Reconstruction Loss
The primary metric for SAE quality. Measures how well the SAE can reconstruct the original activations:
```
reconstruction_loss = ||activation - reconstructed_activation||²
```
Lower is better. Target: < 0.01 for good reconstruction.

### Total Loss
Combined loss including reconstruction and any regularization:
```
total_loss = reconstruction_loss + sparsity_penalty
```
Note: In the current implementation, sparsity is enforced structurally (top-k activation), so total_loss ≈ reconstruction_loss.

### Sparsity Loss
Explicit L1 penalty on feature activations (currently 0.0 due to structural sparsity):
```
sparsity_loss = λ * ||features||₁
```

### L0 Norm
Number of active (non-zero) features:
```
l0_norm = count(features > 0)
```
This measures actual sparsity. Lower values indicate more selective feature usage.

## Troubleshooting

### Plots Not Generated

If plots are not automatically generated during training:

1. Check that matplotlib is installed: `pip install matplotlib seaborn`
2. Check the training log for error messages
3. Manually regenerate using the standalone script:
   ```bash
   python scripts/plot_sae_training.py --sae-dir results/saes
   ```

### Missing Training History Files

If `*_training_history.json` files are missing:
- The SAE training may not have completed successfully
- Check training logs for errors
- Re-run SAE training

### Invalid Plot Data

If plots look incorrect:
- Verify the training history JSON is valid
- Check that history contains expected keys: `reconstruction_loss`, `total_loss`, `sparsity_loss`, `l0_norm`
- Try regenerating with the standalone script

## Examples

### Monitor Training Progress

After running:
```bash
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml --model-name "meta-llama/Llama-2-7b-hf"
```

Check the plots:
```bash
ls results/saes/meta-llama-Llama-2-7b-hf/training_plots/
```

Expected output:
```
meta-llama-Llama-2-7b-hf_layer10_training_history.png
meta-llama-Llama-2-7b-hf_layer10_reconstruction_loss.png
meta-llama-Llama-2-7b-hf_layer15_training_history.png
meta-llama-Llama-2-7b-hf_layer15_reconstruction_loss.png
...
```

### Compare Models

To compare training across different models:

```bash
# Train multiple models
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml --model-name "meta-llama/Llama-2-7b-hf"
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml --model-name "mistralai/Mistral-7B-v0.1"

# Generate comparison plots for each
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf" --comparison
python scripts/plot_sae_training.py --model-name "mistralai/Mistral-7B-v0.1" --comparison
```

Then visually compare the plots in each model's `training_plots/` directory.

## Future Enhancements

Potential improvements for future versions:

- [ ] Real-time plotting during training (with live updates)
- [ ] Interactive plots (using plotly)
- [ ] Automatic anomaly detection (training failures, non-convergence)
- [ ] Multi-model comparison plots
- [ ] Tensorboard integration
- [ ] Export plots in multiple formats (PDF, SVG)
- [ ] Customizable plot styling via config files

## Related Documentation

- [SAE Training Guide](SAE_TRAINING.md) - General SAE training documentation
- [Visualization Guide](VISUALIZATION.md) - Other visualization features
- [Pipeline Overview](../README.md) - Full analysis pipeline

## Contact

For questions or issues related to SAE training plots, please open an issue on the project repository.

