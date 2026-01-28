# Changelog

All notable changes to the DEEM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-28

### Added
- **Bundled AutoML Models**: All 8 trained hyperparameter prediction models now ship with the package
  - No longer requires separate `model_dir` parameter
  - Works out-of-the-box after `pip install deem`
  - Models stored in `deem/automl/models/` (15MB total)
- **Comprehensive Documentation**: Enhanced README with 600+ lines of new content
  - Hyperparameters Reference: 5 tables documenting 30+ parameters
  - Soft Labels Guide: Hard vs soft label formats, auto-detection, conversion functions
  - Preprocessing Guide: Activation functions, initialization methods, YAML equivalence
  - Troubleshooting: 9 common issues with detailed solutions
  - Updated installation and requirements sections

### Changed
- **AutoML Dependencies Now Core**: `scikit-learn`, `pandas`, `joblib` moved from optional to required dependencies
  - Simplifies installation: no longer need `pip install deem[automl]`
  - Just `pip install deem` includes everything
- **Hyperparameter Prediction**: `HyperparameterPredictor.__init__(model_dir=None)` now uses bundled models by default
  - Uses `importlib.resources` for package resource loading
  - Custom `model_dir` still supported for experiments
- **Optuna Optimization**: Removed accuracy from objective function in hyperparameter search
  - Changed from 3 objectives `(accuracy, train_time, loss)` to 2 objectives `(train_time, loss)`
  - Changed study directions from `['maximize', 'minimize', 'minimize']` to `['minimize', 'minimize']`
  - Rationale: Accuracy not relevant for hyperparameter selection

### Fixed
- **Package Distribution**: Updated `MANIFEST.in` and `pyproject.toml` to properly include `.pkl` model files
  - Added `recursive-include deem/automl/models *.pkl` to MANIFEST.in
  - Added `"deem.automl.models"` to packages list
  - Added `automl/models/*.pkl` to package-data

### Technical Details
- Package size increased to ~15MB (due to bundled models)
- Verified wheel distribution includes all model files
- All 105 tests passing
- Compatible with Python >=3.9

## [0.1.1] - 2026-01-20

### Fixed
- Corrected package dependencies for PyPI distribution
- Added `entmax>=1.0` as core dependency (required for multinomial preprocessing)
- Moved `pandas>=1.3.0` to automl optional dependencies
- Removed unnecessary dependencies (omegaconf, optuna, optunahub)

### Changed
- Core dependencies: torch, numpy, scipy, entmax
- Optional dependencies [automl]: scikit-learn, pandas, joblib

## [0.1.0] - 2026-01-20

### Added
- Initial release of DEEM (Deep Ensemble Energy Models)
- Core `DEEM` class with scikit-learn compatible API
- `MultinomialRBMGwg` model with DLP/GWG sampling
- Energy-based contrastive divergence training
- Hungarian algorithm for label alignment
- Support for hard and soft labels
- Missing value handling (use -1 for missing predictions)
- GPU acceleration via PyTorch
- Automatic hyperparameter selection (optional)
- Model save/load functionality
- Comprehensive test suite (23+ tests)

### Features
- 3-line API for ensemble aggregation
- Unsupervised training (no labels required)
- Automatic label permutation handling
- Deterministic and stochastic training modes
- Persistent replay buffer for efficient sampling
- Optional multinomial preprocessing layer

### Components
- `deem.DEEM`: High-level API
- `deem.core.models`: RBM implementations
- `deem.core.samplers`: DLP and GWG samplers
- `deem.core.training`: Training loop
- `deem.core.losses`: Energy-based loss functions
- `deem.core.evaluation`: Hungarian algorithm and metrics
- `deem.data`: Data loading and preprocessing
- `deem.automl`: Hyperparameter prediction

### Documentation
- README with quickstart guide
- API reference
- Usage examples for crowd learning and ensemble aggregation
- MIT License

### Known Issues
- Soft label support has minor bugs (being addressed)
- Automatic hyperparameter selection requires additional trained models

## [Unreleased]

### Planned
- Complete soft label support
- Extended documentation and tutorials
- Additional baseline methods
- Hyperparameter search utilities
- Pre-trained models for common scenarios
