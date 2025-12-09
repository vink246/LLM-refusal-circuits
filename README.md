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
│   ├── data/          # Data processing configurations
│   ├── models/         # Model inference configurations
│   ├── sae/            # SAE training configurations
│   └── circuits/       # Circuit discovery configurations
├── src/                # Source code
│   ├── data/           # Data processing (loaders, analyzers, processors)
│   ├── models/          # Model inference (wrappers, pipelines)
│   ├── sae/            # Sparse Autoencoders (models, trainers, managers)
│   ├── circuits/       # Circuit discovery (discovery, evaluation, similarity)
│   ├── visualization/  # Visualization (heatmaps, plots)
│   ├── analysis/       # Statistics and reporting
│   └── utils/          # Utilities (config, logging)
├── scripts/            # Workflow scripts
├── tests/              # Unit tests
├── data/               # Data directory
│   └── raw/            # Raw datasets (OR-Bench)
└── results/            # Results (gitignored)
    ├── activations/    # Collected model activations
    ├── saes/           # Trained SAE models
    ├── circuits/       # Discovered circuits
    ├── evaluation/     # Circuit evaluation metrics
    ├── evaluation_stats/ # Statistical test results
    ├── visualizations/ # Generated plots and heatmaps
    ├── data_analysis/   # Data split analysis
    └── reports/        # Final analysis reports
```

## Quick Start

### 0. Load the Anaconda Module
PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
> ```bash
> module avail anaconda
> ```

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate llm_refusal_env

# Or use pip (if requirements.txt exists)
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download OR-Bench dataset
hf download bench-llm/or-bench --repo-type dataset --local-dir data/raw/or-bench
```

### 3. Download Models

```bash
huggingface-cli login
# This will prompt you for your Hugging Face API token
``` 

```bash
export HF_HOME=/home/hice1/vkulkarni46/scratch/huggingface
```

```bash
hf download meta-llama/Llama-2-7b-chat-hf --cache-dir $HF_HOME
```

```bash
hf download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir $HF_HOME
```

### 4. Run Analysis Pipeline

```bash
# Step 1: Analyze data splits
python scripts/analyze_data.py \
    --dataset-dir data/raw/or-bench \
    --output-dir results/data_analysis \
    --config configs/data/orbench_analysis.yaml  # Optional

# Step 2: Collect activations
python scripts/collect_activations.py \
    --config configs/models/llama2_7b.yaml \
    --data-config configs/data/orbench_balanced.yaml  # Optional

# Step 3: Train SAEs
python scripts/train_saes.py \
    --config configs/sae/sae_balanced_llama.yaml \
    --result-dir results \
    --model-name "meta-llama/Llama-2-7b-chat-hf"  # Optional, overrides config
    --categories violence deception  # Optional, overrides config

# Step 4: Discover circuits
python scripts/discover_circuits.py \
    --config configs/circuits/discovery_separate_safe_toxic_llama.yaml \
    --result-dir results \
    --model-name "meta-llama/Llama-2-7b-chat-hf"  # Optional
    --categories violence deception  # Optional

# Step 5: Analyze circuits
python scripts/analyze_circuits.py \
    --config configs/circuits/discovery_separate_safe_toxic_llama.yaml \
    --circuits-dir results/circuits  # Optional
    --no-visualizations  # Optional, skip visualization generation

# Step 6: Compute circuit statistics (permutation tests)
python scripts/compute_circuit_stats.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --circuit_dir results/circuits \
    --output_dir results/evaluation_stats \
    --categories violence deception  # Optional, default: all found

# Step 7: Evaluate circuits (faithfulness & completeness)
python scripts/evaluate_circuits.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --circuit_dir results/circuits \
    --output_dir results/evaluation \
    --device cuda \
    --cache_dir /path/to/huggingface/cache  # Optional

# Step 8: Generate report
python scripts/generate_report.py \
    --results-dir results \
    --output results/reports/final_report.md  # Optional
    --format markdown  # Options: markdown, html, pdf, all
```

## Workflow Scripts

The analysis pipeline consists of eight main scripts:

1. **`analyze_data.py`** - Analyze data splits (safe/toxic/hard) and identify imbalances
2. **`collect_activations.py`** - Run inference on models and collect activations from specified layers
3. **`train_saes.py`** - Train sparse autoencoders on collected activations
4. **`discover_circuits.py`** - Discover sparse feature circuits for each category
5. **`analyze_circuits.py`** - Compare circuits across categories and generate visualizations
6. **`compute_circuit_stats.py`** - Compute statistical significance tests (permutation tests) for discovered circuits
7. **`evaluate_circuits.py`** - Evaluate circuit faithfulness and completeness metrics across different thresholds
8. **`generate_report.py`** - Generate comprehensive final analysis report

Each script can be run independently or as part of the complete pipeline. See individual script help for options:

```bash
python scripts/<script_name>.py --help
```

## Configuration

All parameters are configured via YAML files in `configs/`. Each config directory contains README files with detailed explanations.

### Config File Structure

#### `configs/data/` - Data Processing Settings

**Files:**

-   `orbench_analysis.yaml` - Configuration for data split analysis
-   `orbench_balanced.yaml` - Configuration for balanced data preparation

**Key Parameters:**

-   `dataset.path`: Path to OR-Bench dataset directory
-   `dataset.categories`: Categories to analyze (empty = all)
-   `analysis.min_toxic_ratio_warning`: Minimum toxic ratio threshold for warnings
-   `analysis.generate_plots`: Whether to generate visualizations
-   `output.dir`: Output directory for analysis results

#### `configs/models/` - Model Inference Settings

**Files:**

-   `llama2_7b.yaml` - Configuration for Llama-2-7b-chat-hf
-   `mistral_7b.yaml` - Configuration for Mistral-7B-Instruct-v0.1

**Key Parameters:**

-   `model.name`: Model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
-   `model.device`: Device to use ("cuda" or "cpu")
-   `model.torch_dtype`: Data type ("float16" or "float32")
-   `model.cache_dir`: HuggingFace cache directory
-   `model.activation_layers`: List of layers to collect activations from (e.g., ["residuals_21", "residuals_22", ...])
-   `inference.batch_size`: Batch size for inference
-   `inference.max_new_tokens`: Number of tokens to generate
-   `data.path`: Path to dataset directory
-   `data.categories`: Categories to process (empty = all)
-   `data.category_balance_strategy`: "use_all" or "equalize"
-   `output.result_dir`: Result directory

#### `configs/sae/` - SAE Training Settings

**Files:**

-   `sae_balanced_llama.yaml` - SAE training config for LLaMA
-   `sae_balanced_mistral.yaml` - SAE training config for Mistral

**Key Parameters:**

-   `model.name`: Model name (must match activation collection model)
-   `model.activation_layers`: Activation layers to train SAEs for (optional, inferred if not provided)
-   `data.categories`: Categories to use (optional, inferred if not provided)
-   `sae.hidden_dim`: SAE hidden dimension (e.g., 8192 for 8x expansion)
-   `sae.sparsity_coeff`: Sparsity coefficient (e.g., 0.01)
-   `sae.k_percent`: Top-k sparsity percentage (e.g., 0.05 for 5%)
-   `training.epochs`: Number of training epochs
-   `training.batch_size`: Batch size for training
-   `training.max_samples`: Maximum samples to use for training
-   `training.balance_safe_toxic`: Whether to balance safe/toxic during training (CRITICAL)
-   `training.use_category_weights`: Whether to weight categories inversely to frequency
-   `training.learning_rate`: Learning rate
-   `output.save_dir`: Directory to save trained SAEs

**Important:** These configs train a SINGLE SAE on ALL data (safe + toxic, all categories) to ensure feature space comparability.

#### `configs/circuits/` - Circuit Discovery Settings

**Files:**

-   `discovery_separate_safe_toxic_llama.yaml` - Circuit discovery for LLaMA with separate safe/toxic circuits
-   `discovery_separate_safe_toxic_mistral.yaml` - Circuit discovery for Mistral with separate safe/toxic circuits
-   `discovery_separate_safe_toxic_debug.yaml` - Debug configuration

**Key Parameters:**

-   `model.name`: Model name (must match activation collection model)
-   `circuit.node_threshold`: Correlation threshold for important features (used when labels have variance)
-   `circuit.edge_threshold`: Correlation threshold for edges (0.001-0.05 recommended)
-   `circuit.top_n_per_layer`: Number of top features to select per layer (used when labels have no variance)
-   `circuit.aggregation_method`: Aggregation method ("mean", "max", or "none")
-   `circuit.attribution_method`: Attribution method ("stats", "ig", "atp")
-   `discovery.separate_safe_toxic`: Whether to discover separate circuits for safe and toxic samples
-   `discovery.categories`: Categories to discover circuits for (empty = all available)
-   `analysis.equalize_circuits`: Whether to equalize circuits before comparison (recommended)
-   `analysis.compare_categories`: Whether to compare circuits across categories
-   `analysis.generate_visualizations`: Whether to generate visualizations
-   `output.circuits_dir`: Directory to save discovered circuits
-   `output.visualizations_dir`: Directory to save visualizations
-   `output.reports_dir`: Directory to save reports

## Output Directories

Results are saved to the `results/` directory (gitignored):

-   `results/activations/` - Collected model activations (`.pt` files)
-   `results/saes/` - Trained SAE models and training histories
-   `results/circuits/` - Discovered circuits (JSON files)
    -   Pattern: `{model_name}_{category}_toxic_circuit.json` or `{model_name}_{category}_safe_circuit.json`
-   `results/evaluation/` - Circuit evaluation metrics (faithfulness/completeness)
-   `results/evaluation_stats/` - Statistical test results (permutation tests)
-   `results/visualizations/` - Generated plots and heatmaps
-   `results/data_analysis/` - Data split analysis results
-   `results/reports/` - Final analysis reports

## Circuit File Naming

Circuits are saved with the following naming patterns:

-   **Toxic circuits**: `{model_name}_{category}_toxic_circuit.json`
    -   Example: `meta-llama-Llama-2-7b-chat-hf_violence_toxic_circuit.json`
-   **Safe circuits**: `{model_name}_{category}_safe_circuit.json`
    -   Example: `meta-llama-Llama-2-7b-chat-hf_violence_safe_circuit.json`
-   **Combined circuits**: `{model_name}_{category}_circuit.json`
    -   Example: `meta-llama-Llama-2-7b-chat-hf_violence_circuit.json`

Where `{model_name}` is the model name with `/` replaced by `-` and spaces replaced by `_`.
