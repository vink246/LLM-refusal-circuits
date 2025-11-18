# LLM Refusal Circuits Analysis

Class project for Georgia Tech's CS 4650: investigating monolith vs modular refusal behavior of LLMs across refusal categories using sparse feature circuits.

## Project Overview

This project analyzes whether LLM refusal behavior is **monolithic** (shared circuits across categories) or **modular** (category-specific circuits) by:

1. Collecting model activations on OR-Bench dataset
2. Training Sparse Autoencoders (SAEs) to decompose activations into interpretable features
3. Discovering sparse feature circuits for each refusal category using SAE-encoded features
4. Comparing circuits across categories with statistical significance testing

## Project Structure

```
llm-refusal-circuits/
├── configs/           # Configuration files (YAML)
├── src/               # Source code
│   ├── data/          # Data processing
│   ├── models/        # Model inference
│   ├── sae/           # Sparse Autoencoders
│   ├── circuits/      # Circuit discovery
│   ├── visualization/ # Visualization
│   ├── analysis/      # Statistics and reporting
│   └── utils/         # Utilities
├── scripts/           # Workflow scripts
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── docs/              # Documentation
├── data/              # Data directory
└── results/           # Results (gitignored)
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f llm_refusal_env.yml
conda activate llm_refusal_env

# Or use pip
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download OR-Bench dataset
hf download bench-llm/or-bench --repo-type dataset --local-dir data/raw/or-bench
```

### 3. Run Analysis Pipeline

```bash
# Step 1: Analyze data splits
python scripts/analyze_data.py --config configs/data/orbench_default.yaml

# Step 2: Collect activations
# Option A: Sparse layers (quick, 7 layers)
python scripts/collect_activations.py --config configs/models/llama2_7b.yaml

# Option B: All decoder layers (comprehensive, 32 layers)
python scripts/collect_activations.py --config configs/models/llama2_7b_all_layers.yaml

# Option C: Middle layers only (focused, 12 layers)
python scripts/collect_activations.py --config configs/models/llama2_7b_middle_layers.yaml

# Step 3: Train SAEs
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml

# Step 4: Discover circuits
python scripts/discover_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 5: Analyze circuits
python scripts/analyze_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 6: Generate report
python scripts/generate_report.py --results-dir results/
```

## Workflow Scripts

The analysis pipeline consists of six main scripts:

1. **`analyze_data.py`** - Analyze data splits (safe/toxic/hard) and identify imbalances
2. **`collect_activations.py`** - Run inference on models and collect activations from specified layers
3. **`train_saes.py`** - Train sparse autoencoders on collected activations
    - **Automatic plotting**: Training loss plots are automatically generated after training completes
    - Plots saved to: `results/saes/{model_name}/training_plots/`
4. **`discover_circuits.py`** - Discover sparse feature circuits for each category
5. **`analyze_circuits.py`** - Compare circuits across categories and generate visualizations
6. **`generate_report.py`** - Generate comprehensive final analysis report

### Additional Utilities

-   **`plot_sae_training.py`** - Regenerate SAE training plots from saved history files

```bash
# Plot all models
python scripts/plot_sae_training.py --sae-dir results/saes

# Plot specific model with layer comparisons
python scripts/plot_sae_training.py --model-name "meta-llama/Llama-2-7b-hf" --comparison

# Plot from specific history file
python scripts/plot_sae_training.py --history-file results/saes/model/layer_training_history.json
```

Each script can be run independently or as part of the complete pipeline. See individual script help for options:

```bash
python scripts/<script_name>.py --help
```

## Configuration

All parameters are configured via YAML files in `configs/`:

-   `configs/data/` - Data processing settings
-   `configs/models/` - Model inference settings (**NEW**: Supports layer shortcuts like `"all"`, `"residuals_0-31"`, `"mlp_all"`)
-   `configs/sae/` - SAE training settings
-   `configs/circuits/` - Circuit discovery settings

See `configs/models/README.md` for detailed layer specification syntax.

### Layer Collection Options

The activation collection now supports flexible layer specifications:

```yaml
# Collect from ALL decoder layers (32 layers for Llama-2/Mistral)
activation_layers:
  - "all"  # Shorthand for all residual layers

# Collect from a range of layers
activation_layers:
  - "residuals_0-31"   # All layers
  - "residuals_10-21"  # Middle layers only

# Collect specific layer types
activation_layers:
  - "residuals_all"  # All residual layers
  - "mlp_all"        # All MLP layers
  - "attention_all"  # All attention layers

# Traditional explicit list (still supported)
activation_layers:
  - "residuals_0"
  - "residuals_5"
  - "residuals_10"
```

**Memory Note**: Collecting from all 32 layers requires more memory. The default batch_size is reduced from 4 to 2 in `*_all_layers.yaml` configs.

## Documentation

See `docs/` for detailed documentation:

-   `docs/setup.md` - Setup instructions
-   `docs/usage.md` - Usage guide
-   `docs/architecture.md` - Architecture overview
-   `docs/api.md` - API documentation

## License

See LICENSE file for details.
