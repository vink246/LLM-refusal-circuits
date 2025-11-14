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
├── scripts/           # Workflow scripts (01-06)
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
python scripts/01_analyze_data.py --config configs/data/orbench_default.yaml

# Step 2: Collect activations
python scripts/02_collect_activations.py --config configs/models/llama2_7b.yaml

# Step 3: Train SAEs
python scripts/03_train_saes.py --config configs/sae/sae_balanced.yaml

# Step 4: Discover circuits
python scripts/04_discover_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 5: Analyze circuits
python scripts/05_analyze_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 6: Generate report
python scripts/06_generate_report.py --results-dir results/
```

## Workflow Scripts

1. **`01_analyze_data.py`** - Analyze data splits (safe/toxic/hard)
2. **`02_collect_activations.py`** - Run inference and collect activations
3. **`03_train_saes.py`** - Train sparse autoencoders
4. **`04_discover_circuits.py`** - Discover circuits for each category
5. **`05_analyze_circuits.py`** - Compare circuits and generate visualizations
6. **`06_generate_report.py`** - Generate comprehensive final report

## Configuration

All parameters are configured via YAML files in `configs/`:

- `configs/data/` - Data processing settings
- `configs/models/` - Model inference settings
- `configs/sae/` - SAE training settings
- `configs/circuits/` - Circuit discovery settings

## Documentation

See `docs/` for detailed documentation:
- `docs/setup.md` - Setup instructions
- `docs/usage.md` - Usage guide
- `docs/architecture.md` - Architecture overview
- `docs/api.md` - API documentation

## License

See LICENSE file for details.
