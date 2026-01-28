# RBM Python Project - AI Agent Guidelines

## Project Overview
Research codebase for **Restricted Boltzmann Machines (RBMs)** applied to crowd learning and ensemble classifier aggregation. The project uses energy-based models to learn from noisy multi-annotator data where each data point has predictions from multiple classifiers.

**Core concept**: Train RBMs to aggregate predictions from multiple weak classifiers (crowd annotations) into accurate consensus labels by modeling their joint distribution as an energy function.

## Architecture & Key Components

### Directory Structure
- **`src/`** - Main source code (code execution happens here, not `rbm_python/`)
  - `models/` - RBM implementations (abstract base classes → specific variants)
  - `utils/` - Data loading, metrics, Hungarian algorithm for label mapping
  - `experiment_configs/` - YAML-based experiment definitions
  - `benchmarks/` - Baseline methods (majority vote, best classifier)
  - `losses/` - Energy-based model losses
  - Root scripts: `run_optuna.py`, `run_experiment.py`, `algorithm.py`, `training.py`
- **`datasets/`** - .mat files containing crowd annotations
- **`tests/`** - pytest suite (run with `pytest` from root, pythonpath=src in pytest.ini)
- **`rbm_python/`** - Legacy/auxiliary code, mostly unused in main workflow

### Data Format & Flow
**Input data** (`.mat` files in `datasets/`):
- `f` matrix: `(n_samples, n_classifiers)` with classifier predictions (integers 0 to k-1) or -1 for missing
- `y` vector: True labels (may contain -1 for unlabeled data)
- `k`: Number of classes (optional, auto-inferred if not provided)

**Dataset sources**:
- Curated benchmarks in `datasets/` (e.g., tree3k.mat, mnist_e_v1.mat, csgo.mat)
- Custom datasets can be loaded via `--path` argument with `default_config.yaml`
- TabRepo datasets in `datasets/tabrepo1/` subdirectory
- YAML configs in `experiment_configs/` map experiment names to dataset paths

**Loading pipeline** (`src/utils/datasets.py`):
```python
load_from_mat(args) → train_loader, val_loader, test_loader, updated_args
```
- Filters out samples with all -1 predictions (rows where all classifiers failed)
- Automatically determines `k` (from data or config), `in_dim`, `input_size=[k, in_dim]`
- Returns PyTorch DataLoaders with `TensorDatasetWithShape`
- Supports 3D soft predictions (probability distributions) converted to 2D argmax for evaluation

### RBM Model Hierarchy

**Abstract base**: `src/models/abstract_rbms.py`
- `RBM` class: Core interface for all RBM variants
- `MultinomialRBM` mixin: Energy calculation, CD algorithm, sampling methods

**Concrete implementations** (most used: `MultinomialRBMGwg`):
1. **`MultinomialRBMGwg`** (`models/multinomial_rbm_gwg.py`) - Gibbs-with-gradients, production model
2. **`MultinomialRBMManual`** (`models/multinomial_rbm_manual.py`) - Manual CD implementation
3. **`MultinomialRBMLangevin`** (`models/multinomial_rbm_langevin_new.py`) - Langevin dynamics sampling
4. **`MultinomialDBN`** (`models/multinomial_dbn.py`) - Deep belief network wrapper

**Key RBM concepts**:
- `dx`: Visible dimension (number of classifiers)
- `dh`: Hidden dimension (learned representation size)
- `cd_k`: Contrastive divergence steps
- `deterministic`: Use probabilities instead of sampling during training
- Energy function: Lower energy = higher probability configuration

### Training Flow

**Primary workflow** (`src/run_predict.py`) - **Use this for running experiments**:
```bash
cd src
python run_predict.py <experiment_name> [--path <dataset.mat>] [--seq]
```

This is the **production script** that:
- Runs experiments 5 times with different seeds in parallel (use `--seq` for sequential)
- Averages results across runs for statistical validity
- Uses hyperparameter prediction model to automatically set optimal hyperparameters
- Extracts meta-features from datasets to predict best hyperparameters
- Handles both predefined experiments (YAML configs) and custom datasets via `--path`

**Hyperparameter prediction**: The model uses trained ML models in `saved_hyp_models_v1/` to predict optimal hyperparameters (lr, batch_size, num_layers, etc.) based on dataset meta-features (n_samples, sequence_length, token_density, etc.)

**Workflow**:
1. Load experiment config from `experiment_configs/<name>.yaml` OR use `default_config.yaml` with custom path
2. Extract meta-features from dataset (28 features: n_samples, token_density, entropy, etc.)
3. Predict optimal hyperparameters using trained sklearn models in `saved_hyp_models_v1/`
4. Initialize model via `define_model()` with predicted/configured parameters
5. **Train using INLINE LOOP** (lines 378-398 in run_predict.py) - NOT calling training.py
6. Evaluate with Hungarian algorithm to solve label permutation problem
7. Average results across 5 runs (parallel multiprocessing with spawn) and compare against baselines

**CRITICAL**: The actual training code is **inline in run_predict.py**, NOT in `training.py`:
- ✅ `run_predict.py` lines 378-398: Production training loop (correct, no break)
- ❌ `training.py` with `train_rbm()`: LEGACY, contains break bug, NOT USED

**Legacy scripts** (still functional but not primary workflow):
- `training.py`: LEGACY training function (has break bug, not called in production)
- `run_experiment.py`: Single-run experiments with manual hyperparameters
- `run_optuna.py`: Hyperparameter search with Optuna (for finding optimal configs)

**Critical training detail**: The `deterministic` flag affects whether sampling or probabilities are used during CD. Set in YAML configs under `model.deterministic`.

## Development Workflows

### Running Experiments

**Production workflow** (recommended - 5 runs averaged):
```bash
cd src
python run_predict.py tree3k  # Parallel execution (default)
python run_predict.py csgo --seq  # Sequential execution
python run_predict.py default --path datasets/custom_data.mat  # Custom dataset
```

**Hyperparameter search** (for finding optimal configs):
```bash
python run_optuna.py mniste --mode random --trials 100
```

**Single-run experiments** (legacy, for debugging):
```bash
python run_experiment.py tree3k --verbose --seed 42
```

**Available experiments** (defined in `EXPERIMENTS` dict across run scripts):
- **Standard benchmarks**: `tree3k`, `condind`, `mniste`
- **Multimodal datasets**: `csgo`, `hs3`, `hsac`, `petfinder`, `pk1`
- **Large-scale vision**: `imagenet_filter`, `cifar100_experts`, `cifar10`
- **Random forest ensembles**: `pendigits`, `eye_movements`, `artifical_chars`, `gesture_phase`
- **WRENCH datasets**: `agnews`, `census`, `imdb`, `yelp`, `youtube`, `sms`, `spouse`, etc.
- **Custom paths**: Use `default` with `--path` for any .mat file

### SLURM Cluster Jobs

**Submit GPU job**:
```bash
cd src
sbatch run_imagenet_benchmarks.slurm  # or run_optuna.slurm
```

SLURM scripts:
- Activate venv: `source /home/dsi/maymona3/rbm_python/.venv/bin/activate`
- Request GPU: `#SBATCH --gres=gpu:1`
- Logs go to `logs/` directory

### Testing

```bash
pytest  # Runs from root, automatically uses pythonpath=src from pytest.ini
```

Tests cover:
- RBM forward/backward pass (`test_multinomial_rbm.py`)
- GWG sampler (`test_gwg.py`)
- Langevin dynamics (`test_langevin.py`)

## Key Patterns & Conventions

### Experiment Configuration (YAML)

All experiments defined via YAML in `experiment_configs/`. Critical fields:
```yaml
data:
  path: datasets/tree3k.mat
  batch_size: 128

model:
  deterministic: true  # Critical: affects CD sampling
  cd_k: 10             # Contrastive divergence steps
  init_method: "rand"  # Weight init: rand/mv/zero
  sampler:
    steps: 10
    oh_mode: false     # One-hot mode for sampling
  multinomial_net:     # Optional preprocessing layer
    use_deep_net: false

training:
  epochs: 100
  learning_rate: 0.001
  momentum: 0.9

seed:
  start: 0
  end: 5               # Run multiple seeds for statistical significance
```

### Hungarian Algorithm Pattern

**Critical for evaluation**: RBM hidden units have no inherent ordering. Use Hungarian algorithm to match predicted clusters to true labels.

```python
from utils.hungarian import get_hungarian_solution, vectorize_predictions

predictions = model.predict(data)  # Shape: (n, dh)
class_map = get_hungarian_solution(predictions, true_labels, k)
aligned_predictions = vectorize_predictions(predictions, class_map)
accuracy = (aligned_predictions == true_labels).mean()
```

Always use `get_hungarian_solution()` before computing accuracy metrics.

### Energy Monitoring

Track energy during training as sanity check (should generally decrease):
```python
rbm.print_energy_metrics(data)  # Prints sum and mean energy
energies.extend(rbm.energies_sum)  # Store for plotting
```

### Model Initialization Methods

Specified in configs via `init_method`:
- **`"rand"`**: Random initialization (default)
- **`"mv"`**: Initialize to majority vote statistics
- **`"zero"`**: Zero initialization (use with caution)

For `multinomial_net`, use `init_method` under `model.multinomial_net`.

### Missing Values

Data commonly has -1 for missing classifier predictions. Handled in:
- `load_from_mat()`: Filters rows with all -1s
- RBM models: Energy calculation skips -1 entries

## Common Tasks

### Adding a New Dataset
1. Convert to .mat format with `f` (predictions) and `y` (labels) keys
2. Place in `datasets/`
3. Create YAML config in `experiment_configs/` (copy existing, modify path)
4. Add entry to `EXPERIMENTS` dict in run scripts
5. Run: `python run_experiment.py <new_name>`

### Debugging Training Issues
1. Check deterministic flag in YAML
2. Monitor energy: should decrease (if increasing, check learning rate)
3. Verify data loading: `args.k`, `args.in_dim` should match dataset
4. Check for dead units: `extract_dead_classes_indicies()` in evaluation

### Modifying RBM Architecture
- Extend `MultinomialRBM` mixin from `abstract_rbms.py`
- Implement `calc_visible_prob()`, `calc_hidden_prob()`, `energy()`
- Override `contrastive_divergence()` if needed for custom sampling
- See `MultinomialRBMGwg` as reference implementation

## Critical Caveats

1. **Python path**: Code must run from `src/` directory or with `PYTHONPATH=src`
2. **Evaluation requires Hungarian**: Never compute accuracy directly on predictions without Hungarian alignment
3. **Config vs args**: Configs are YAML objects, `args` are Namespace objects from `config_to_args()`
4. **Device handling**: Always check `config.device` or `torch.cuda.is_available()`
5. **Deterministic mode**: Changes both training behavior and inference - keep consistent
6. **Seed management**: Multiple seeds required for statistical validity - `run_predict.py` does this automatically (5 seeds), or use `seed.start/end` in config
7. **-1 handling**: Missing values represented as -1, not NaN or None
8. **Manual flags and variables**: The codebase contains various manual flags, commented code, and experimental features throughout - **DO NOT remove or "clean up" these without explicit permission**. They support different experiment configurations and are intentionally kept for reproducibility.
9. **Multiprocessing**: `run_predict.py` uses spawn method for CUDA compatibility - ensure proper process handling
python run_predict.py default --path ../datasets/my_data.mat

# Production: Sequential execution (for debugging)
python run_predict.py tree3k --seq

# Hyperparameter search (for finding optimal configs)
python run_optuna.py csgo --mode bayesian --trials 50

# Legacy: Single-run experiment (for quick tests)
python run_experiment.py mniste --seed 42 --verbose

# SLURM job submission
cd src && sbatch run_imagenet_benchmarks.slurm

# Tests
pytest  # From project root, uses pythonpath=src from pytest.ini
pytest tests/test_multinomial_rbm.py -v  # Specific test

# Quick data inspection
python -c "import scipy.io; print(scipy.io.loadmat('datasets/tree3k.mat').keys())"
```
# SLURM job
cd src && sbatch run_optuna.slurm

# Tests
pytest tests/test_multinomial_rbm.py -v

# Quick data inspection
python -c "import scipy.io; print(scipy.io.loadmat('datasets/tree3k.mat').keys())"
```

## External Dependencies
- **Optuna**: Hyperparameter optimization framework
- **Comet ML**: Experiment tracking (API key in run scripts, optional)
- **Hungarian algorithm**: Label alignment (`scipy.optimize.linear_sum_assignment`)
- **PyTorch**: Backend for all models and training

---

*For research context, see experiment results in `analysis_results/` and best model configurations in `best_*` files.*
