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
python scripts/collect_activations.py --config configs/models/llama2_7b.yaml

# Step 3: Train SAEs
python scripts/train_saes.py --config configs/sae/sae_balanced.yaml

# Step 4: Discover circuits
python scripts/discover_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 5: Analyze circuits
python scripts/analyze_circuits.py --config configs/circuits/discovery_separate_safe_toxic.yaml

# Step 6: Evaluate circuits (faithfulness & completeness)
python scripts/evaluate_circuits.py --model "meta-llama/Llama-2-7b-chat-hf"

# Step 7: Generate report
python scripts/generate_report.py --results-dir results/
```

## Workflow Scripts

The analysis pipeline consists of seven main scripts:

1. **`analyze_data.py`** - Analyze data splits (safe/toxic/hard) and identify imbalances
2. **`collect_activations.py`** - Run inference on models and collect activations from specified layers
3. **`train_saes.py`** - Train sparse autoencoders on collected activations
4. **`discover_circuits.py`** - Discover sparse feature circuits for each category
5. **`analyze_circuits.py`** - Compare circuits across categories and generate visualizations
6. **`evaluate_circuits.py`** - Evaluate circuit faithfulness and completeness metrics across different thresholds
7. **`generate_report.py`** - Generate comprehensive final analysis report

Each script can be run independently or as part of the complete pipeline. See individual script help for options:
```bash
python scripts/<script_name>.py --help
```

### Circuit Evaluation (`evaluate_circuits.py`)

Evaluates the faithfulness and completeness of discovered circuits across different importance thresholds:

- **Faithfulness**: Measures how well the circuit alone can reproduce the model's refusal behavior (F(C) - F(Empty)) / (F(M) - F(Empty))
- **Completeness**: Measures what fraction of the full model's behavior the circuit captures (F(C) / F(M))

The script:
- Loads circuits for each category from `results/circuits/`
- Filters circuits by importance thresholds (automatically computed from percentiles)
- Evaluates each filtered circuit using ablation experiments
- Generates plots showing faithfulness and completeness vs. circuit size
- Saves metrics and statistical analysis to `results/evaluation/`

**Usage:**
```bash
python scripts/evaluate_circuits.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --circuit_dir "results/circuits" \
    --output_dir "results/evaluation" \
    --device "cuda" \
    --cache_dir "/path/to/huggingface/cache"  # Optional
```

**Outputs:**
- `{model}_{category}_metrics.png` - Plot of faithfulness and completeness vs. number of nodes
- `{model}_{category}_metrics.json` - Raw metrics data
- `{model}_{category}_stats.json` - Statistical significance tests

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
