"""Tests for hyperparameter prediction system.

Tests cover:
- MetaFeatureExtractor: 28 meta-features extraction
- HyperparameterPredictor: Hyperparameter prediction from features
- Integration with real saved models
"""

import numpy as np
import pytest
from pathlib import Path

from deem.automl import MetaFeatureExtractor, HyperparameterPredictor
from deem.automl.meta_features import FEATURE_NAMES, compute_gini_coefficient


# =============================================================================
# MetaFeatureExtractor Tests
# =============================================================================

class TestMetaFeatureExtractor:
    """Tests for MetaFeatureExtractor class."""
    
    def test_extract_basic_features(self):
        """Test extraction of basic features."""
        extractor = MetaFeatureExtractor(num_classes=3)
        predictions = np.random.randint(0, 3, size=(100, 15))
        
        features = extractor.extract(predictions)
        
        assert features['n_samples'] == 100
        assert features['sequence_length'] == 15
        assert features['global_num_classes'] <= 3
    
    def test_extract_returns_all_28_features(self):
        """Test that all 28 features are extracted."""
        extractor = MetaFeatureExtractor()
        predictions = np.random.randint(0, 5, size=(200, 10))
        
        features = extractor.extract(predictions)
        
        assert len(features) == 29  # 28 features + 1 (we have 29 in dict)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
    
    def test_extract_with_missing_values(self):
        """Test extraction handles -1 (missing) values."""
        extractor = MetaFeatureExtractor(num_classes=3)
        predictions = np.random.randint(0, 3, size=(100, 15))
        # Add some missing values
        predictions[0:10, 0] = -1
        predictions[5:15, 5] = -1
        
        features = extractor.extract(predictions)
        
        assert features['n_samples'] <= 100  # Some samples may be invalid
        assert 'token_density' in features
    
    def test_extract_infers_num_classes(self):
        """Test that num_classes is inferred from data."""
        extractor = MetaFeatureExtractor()  # No num_classes specified
        predictions = np.random.randint(0, 7, size=(100, 12))
        
        features = extractor.extract(predictions)
        
        assert features['global_num_classes'] <= 7
    
    def test_extract_as_vector(self):
        """Test extraction as ordered vector."""
        extractor = MetaFeatureExtractor(num_classes=3)
        predictions = np.random.randint(0, 3, size=(100, 15))
        
        vector = extractor.extract_as_vector(predictions)
        
        assert vector.shape == (len(FEATURE_NAMES),)
        assert vector.dtype == np.float64
    
    def test_extract_deterministic(self):
        """Test that extraction is deterministic for same input."""
        np.random.seed(42)
        extractor = MetaFeatureExtractor(num_classes=3)
        predictions = np.random.randint(0, 3, size=(50, 10))
        
        features1 = extractor.extract(predictions)
        features2 = extractor.extract(predictions)
        
        # Deterministic features should match
        assert features1['n_samples'] == features2['n_samples']
        assert features1['sequence_length'] == features2['sequence_length']
        assert features1['token_frequency_entropy'] == features2['token_frequency_entropy']
    
    def test_extract_raises_on_invalid_input(self):
        """Test that extraction raises on invalid input."""
        extractor = MetaFeatureExtractor()
        
        # 1D input
        with pytest.raises(ValueError, match="must be 2D"):
            extractor.extract(np.array([1, 2, 3]))
        
        # All negative values
        with pytest.raises(ValueError, match="No valid predictions"):
            extractor.extract(np.full((10, 5), -1))
    
    def test_feature_names_static_method(self):
        """Test get_feature_names() static method."""
        names = MetaFeatureExtractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) == len(FEATURE_NAMES)
        assert names == FEATURE_NAMES


class TestComputeGiniCoefficient:
    """Tests for Gini coefficient computation."""
    
    def test_perfect_equality(self):
        """Test Gini = 0 for perfect equality."""
        arr = np.array([10.0, 10.0, 10.0, 10.0])
        gini = compute_gini_coefficient(arr)
        assert abs(gini) < 0.01  # Should be close to 0
    
    def test_perfect_inequality(self):
        """Test Gini close to 1 for perfect inequality."""
        arr = np.array([0.0, 0.0, 0.0, 100.0])
        gini = compute_gini_coefficient(arr)
        assert gini > 0.5  # Should be high
    
    def test_empty_array(self):
        """Test empty array returns 0."""
        gini = compute_gini_coefficient(np.array([]))
        assert gini == 0.0
    
    def test_zero_array(self):
        """Test array of zeros returns 0."""
        gini = compute_gini_coefficient(np.zeros(5))
        assert gini == 0.0


# =============================================================================
# HyperparameterPredictor Tests
# =============================================================================

class TestHyperparameterPredictor:
    """Tests for HyperparameterPredictor class."""
    
    def test_init_with_missing_dir_no_raise(self):
        """Test initialization with missing model dir doesn't raise by default."""
        predictor = HyperparameterPredictor(
            model_dir='nonexistent_dir',
            raise_on_missing=False,
        )
        
        assert not predictor.is_ready()
        assert len(predictor.models) == 0
    
    def test_init_with_missing_dir_raises(self):
        """Test initialization with missing model dir raises when requested."""
        with pytest.raises(FileNotFoundError):
            HyperparameterPredictor(
                model_dir='nonexistent_dir',
                raise_on_missing=True,
            )
    
    def test_default_hyperparameters(self):
        """Test default hyperparameters are returned when models missing."""
        predictor = HyperparameterPredictor(
            model_dir='nonexistent_dir',
            raise_on_missing=False,
        )
        
        defaults = predictor.get_default_hyperparameters()
        
        assert 'batch_size' in defaults
        assert 'epochs' in defaults
        assert 'learning_rate' in defaults
        assert 'init_method' in defaults
        assert 'num_layers' in defaults
        assert 'activation_func' in defaults
        assert 'momentum' in defaults
        assert 'scheduler' in defaults
    
    def test_predict_returns_defaults_when_no_models(self):
        """Test prediction returns defaults when no models loaded."""
        predictor = HyperparameterPredictor(
            model_dir='nonexistent_dir',
            raise_on_missing=False,
        )
        predictions = np.random.randint(0, 3, size=(100, 15))
        
        hyperparams = predictor.predict(predictions)
        
        defaults = predictor.get_default_hyperparameters()
        for key, value in defaults.items():
            assert hyperparams[key] == value
    
    def test_predict_as_list_order(self):
        """Test predict_as_list returns correct order."""
        predictor = HyperparameterPredictor(
            model_dir='nonexistent_dir',
            raise_on_missing=False,
        )
        predictions = np.random.randint(0, 3, size=(100, 15))
        
        hyp_list = predictor.predict_as_list(predictions)
        
        # Order: [batch_size, epochs, lr, init_method, num_layers, activation_func, momentum, scheduler]
        assert len(hyp_list) == 8
        assert isinstance(hyp_list[0], int)  # batch_size
        assert isinstance(hyp_list[1], int)  # epochs
        assert isinstance(hyp_list[2], float)  # learning_rate
        assert isinstance(hyp_list[3], str)  # init_method
    
    def test_get_available_parameters(self):
        """Test get_available_parameters returns loaded models."""
        predictor = HyperparameterPredictor(
            model_dir='nonexistent_dir',
            raise_on_missing=False,
        )
        
        params = predictor.get_available_parameters()
        
        assert isinstance(params, list)
        # When no models loaded, should be empty
        assert len(params) == 0


class TestHyperparameterPredictorIntegration:
    """Integration tests with real saved models (if available)."""
    
    @pytest.fixture
    def real_model_dir(self):
        """Get path to real model directory if it exists."""
        # Try various possible locations
        paths = [
            Path('src/saved_hyp_models_v1'),
            Path('saved_hyp_models_v1'),
            Path('../src/saved_hyp_models_v1'),
        ]
        
        for p in paths:
            if p.exists():
                return p
        
        pytest.skip("Model directory not found")
    
    def test_load_real_models(self, real_model_dir):
        """Test loading real sklearn models."""
        predictor = HyperparameterPredictor(
            model_dir=real_model_dir,
            raise_on_missing=False,
        )
        
        assert predictor.is_ready()
        assert len(predictor.models) > 0
    
    def test_predict_with_real_models(self, real_model_dir):
        """Test prediction with real models."""
        predictor = HyperparameterPredictor(
            model_dir=real_model_dir,
            raise_on_missing=False,
        )
        predictions = np.random.randint(0, 3, size=(100, 15))
        
        hyperparams = predictor.predict(predictions)
        
        # Check that predictions are reasonable
        assert 'batch_size' in hyperparams
        assert isinstance(hyperparams['batch_size'], int)
        assert hyperparams['batch_size'] > 0
        
        assert 'learning_rate' in hyperparams
        assert isinstance(hyperparams['learning_rate'], float)


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end tests for the full pipeline."""
    
    def test_full_pipeline_without_models(self):
        """Test full pipeline works even without models."""
        # Create synthetic data
        np.random.seed(42)
        predictions = np.random.randint(0, 5, size=(500, 20))
        
        # Create predictor (will use defaults)
        predictor = HyperparameterPredictor(
            model_dir='nonexistent',
            raise_on_missing=False,
        )
        
        # Predict
        hyperparams = predictor.predict(predictions)
        
        # Verify output format
        assert isinstance(hyperparams, dict)
        assert all(key in hyperparams for key in [
            'batch_size', 'epochs', 'learning_rate', 'init_method',
            'num_layers', 'activation_func', 'momentum', 'scheduler'
        ])
    
    def test_feature_extraction_stability(self):
        """Test that feature extraction is stable across runs."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, size=(100, 10))
        
        extractor = MetaFeatureExtractor(num_classes=3)
        
        # Extract multiple times
        results = [extractor.extract(predictions) for _ in range(3)]
        
        # Deterministic features should be identical
        for key in ['n_samples', 'sequence_length', 'global_num_classes']:
            values = [r[key] for r in results]
            assert all(v == values[0] for v in values)
