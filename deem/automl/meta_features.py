"""Meta-feature extraction for hyperparameter prediction.

This module extracts 28 meta-features from ensemble prediction data that
describe the statistical properties of the predictions. These features are
used by HyperparameterPredictor to predict optimal hyperparameters.

The 28 features capture:
- Basic dataset statistics (n_samples, sequence_length, global_num_classes)
- Token/prediction patterns (density, entropy, kurtosis, repetition)
- Dimensionality measures (dimension_ratio, effective_rank)
- Ensemble-specific features (confidence, agreement, prediction margins)
- Distribution shape (skewness, Gini coefficient)
- Spatial/structural features (hamming distance, spectral entropy)
- Class imbalance metrics
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import entropy, kurtosis, skew

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Ordered list of feature names matching the trained models
FEATURE_NAMES = [
    'n_samples',
    'sequence_length', 
    'global_num_classes',
    'avg_unique_per_sample',
    'token_density',
    'class_frequency_std',
    'token_frequency_entropy',
    'token_frequency_kurtosis',
    'mean_pairwise_overlap',
    'mean_token_repetition',
    'autocorrelation_lag1',
    'token_position_entropy',
    'dimension',
    'effective_dimension',
    'dimension_ratio',
    'avg_confidence',
    'std_confidence',
    'avg_prediction_margin',
    'avg_agreement',
    'freq_skewness',
    'freq_gini',
    'normalized_entropy',
    'avg_hamming_distance',
    'effective_rank',
    'spectral_entropy',
    'position_variance',
    'avg_positional_disagreement',
    'class_imbalance_ratio',
    'minority_class_ratio',
]


def compute_gini_coefficient(arr: NDArray[np.floating]) -> float:
    """Compute Gini coefficient for measuring inequality in class distribution.
    
    Parameters
    ----------
    arr : array-like
        Array of values (e.g., class frequencies)
        
    Returns
    -------
    float
        Gini coefficient in [0, 1], where 0 = perfect equality, 1 = perfect inequality
    """
    if len(arr) == 0:
        return 0.0
    
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(sorted_arr)
    
    if cumsum[-1] == 0:
        return 0.0
    
    return float(
        (2 * np.sum((np.arange(1, n + 1) * sorted_arr))) / (n * cumsum[-1]) 
        - (n + 1) / n
    )


class MetaFeatureExtractor:
    """Extract 28 meta-features from ensemble prediction data.
    
    These meta-features characterize the statistical properties of the
    predictions and are used to predict optimal hyperparameters for RBM training.
    
    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If None, inferred from data as max(predictions) + 1.
    max_samples_for_pca : int, default=100
        Maximum samples to use for PCA-based features (for efficiency).
    max_samples_for_overlap : int, default=50
        Maximum samples to use for pairwise overlap computation.
        
    Examples
    --------
    >>> import numpy as np
    >>> from deem.automl import MetaFeatureExtractor
    >>> 
    >>> # Create synthetic ensemble predictions
    >>> predictions = np.random.randint(0, 3, size=(1000, 15))
    >>> 
    >>> # Extract features
    >>> extractor = MetaFeatureExtractor(num_classes=3)
    >>> features = extractor.extract(predictions)
    >>> print(features['n_samples'])  # 1000
    >>> print(features['sequence_length'])  # 15
    """
    
    def __init__(
        self,
        num_classes: int | None = None,
        max_samples_for_pca: int = 100,
        max_samples_for_overlap: int = 50,
    ):
        self.num_classes = num_classes
        self.max_samples_for_pca = max_samples_for_pca
        self.max_samples_for_overlap = max_samples_for_overlap
        
    def extract(self, predictions: NDArray[np.integer]) -> dict[str, Any]:
        """Extract 28 meta-features from ensemble predictions.
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Ensemble predictions where each row is a sample and each column
            is a classifier's prediction. Values should be integers 0 to k-1,
            or -1 for missing values.
            
        Returns
        -------
        dict
            Dictionary containing 28 meta-features with keys matching FEATURE_NAMES.
        """
        predictions = np.asarray(predictions)
        
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions must be 2D (n_samples, n_classifiers), got shape {predictions.shape}"
            )
        
        n_samples, sequence_length = predictions.shape
        
        # Determine number of classes
        valid_preds = predictions[predictions >= 0]
        if len(valid_preds) == 0:
            raise ValueError("No valid predictions found (all values are < 0)")
            
        num_classes = self.num_classes or int(valid_preds.max()) + 1
        
        # Initialize accumulators
        all_token_counts: Counter[int] = Counter()
        unique_per_sample: list[int] = []
        token_repetitions: list[int] = []
        overlaps: list[float] = []
        autocorr_numerator = 0
        autocorr_denominator = 0
        token_position_counts = np.zeros((sequence_length, num_classes), dtype=np.int64)
        
        # New feature collectors
        all_sequences: list[list[int]] = []
        confidence_scores: list[float] = []
        prediction_margins: list[float] = []
        agreement_scores: list[float] = []
        
        total_samples = 0
        
        for seq in predictions:
            # Filter valid tokens
            valid_tokens = [int(t) for t in seq if 0 <= t < num_classes]
            if len(valid_tokens) == 0:
                continue
            
            total_samples += 1
            all_sequences.append(list(seq))
            
            # Unique tokens per sample
            unique_tokens = set(valid_tokens)
            unique_per_sample.append(len(unique_tokens))
            
            # Token counts
            all_token_counts.update(valid_tokens)
            
            # Token repetitions
            repetition = len(valid_tokens) - len(unique_tokens)
            token_repetitions.append(repetition)
            
            # Autocorrelation (consecutive same predictions)
            autocorr_numerator += np.sum(seq[:-1] == seq[1:])
            autocorr_denominator += len(seq) - 1
            
            # Position-token counts
            for pos, token in enumerate(seq):
                if 0 <= token < num_classes:
                    token_position_counts[pos, int(token)] += 1
            
            # Ensemble-specific features
            seq_counts = Counter(valid_tokens)
            mode_freq = max(seq_counts.values())
            confidence_scores.append(mode_freq / len(valid_tokens))
            
            # Prediction margin (difference between top 2 predictions)
            sorted_counts = sorted(seq_counts.values(), reverse=True)
            if len(sorted_counts) >= 2:
                prediction_margins.append((sorted_counts[0] - sorted_counts[1]) / len(valid_tokens))
            else:
                prediction_margins.append(1.0)  # Perfect agreement
            
            # Agreement score (entropy-based)
            probs = np.array(list(seq_counts.values())) / len(valid_tokens)
            n_unique = len(seq_counts)
            if n_unique > 1:
                max_entropy = np.log(min(n_unique, num_classes))
                if max_entropy > 0:
                    agreement_scores.append(1 - entropy(probs) / max_entropy)
                else:
                    agreement_scores.append(1.0)
            else:
                agreement_scores.append(1.0)
        
        if total_samples == 0:
            raise ValueError("No valid samples to process.")
        
        # Compute pairwise overlaps (sample for efficiency)
        sample_size = min(total_samples, self.max_samples_for_overlap)
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
        for i_idx, i in enumerate(sample_indices):
            a = set(all_sequences[i])
            for j_idx in range(i_idx + 1, sample_size):
                j = sample_indices[j_idx]
                b = set(all_sequences[j])
                inter = len(a & b)
                union = len(a | b)
                if union > 0:
                    overlaps.append(inter / union)
        
        # Frequency-based features
        freq_arr = np.zeros(num_classes)
        for token, count in all_token_counts.items():
            if 0 <= token < num_classes:
                freq_arr[token] = count
        
        freq_sum = freq_arr.sum()
        freq_probs = freq_arr / freq_sum if freq_sum > 0 else np.ones_like(freq_arr) / num_classes
        
        # Token position entropy
        token_position_entropy = np.mean([
            entropy(row / np.sum(row)) if np.sum(row) > 0 else 0.0
            for row in token_position_counts
        ])
        
        autocorr_lag1 = autocorr_numerator / autocorr_denominator if autocorr_denominator > 0 else 0.0
        
        # Dimensionality features
        dimension = sequence_length
        effective_dimension = int((freq_arr > 0).sum())
        dimension_ratio = effective_dimension / dimension if dimension > 0 else 0.0
        
        # Ensemble-specific averages
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        std_confidence = float(np.std(confidence_scores)) if confidence_scores else 0.0
        avg_prediction_margin = float(np.mean(prediction_margins)) if prediction_margins else 0.0
        avg_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0
        
        # Distribution shape features
        freq_skewness = float(skew(freq_arr))
        freq_gini = compute_gini_coefficient(freq_arr)
        
        # Spatial/structural features using PCA
        all_sequences_array = np.array(all_sequences)
        if len(all_sequences_array) > 1:
            try:
                from sklearn.decomposition import PCA
                from sklearn.metrics import pairwise_distances
                
                # Sample for efficiency
                sample_seqs = all_sequences_array[:min(self.max_samples_for_pca, len(all_sequences_array))]
                
                # Hamming distance
                pairwise_dists = pairwise_distances(sample_seqs, metric='hamming')
                avg_hamming_distance = float(np.mean(pairwise_dists[np.triu_indices(len(pairwise_dists), k=1)]))
                
                # PCA for intrinsic dimensionality
                pca = PCA()
                pca.fit(sample_seqs)
                explained_var_ratio = pca.explained_variance_ratio_
                effective_rank = int(np.sum(explained_var_ratio > 0.01))
                spectral_entropy = float(entropy(explained_var_ratio + 1e-10))
            except Exception:
                avg_hamming_distance = 0.0
                effective_rank = dimension
                spectral_entropy = 0.0
        else:
            avg_hamming_distance = 0.0
            effective_rank = dimension
            spectral_entropy = 0.0
        
        # Positional features
        position_variance = float(np.var([np.var(token_position_counts[i]) for i in range(sequence_length)]))
        
        # Positional disagreement
        if len(all_sequences_array) > 1:
            disagreement_matrix = np.zeros((sequence_length, sequence_length))
            sample_seqs = all_sequences_array[:min(self.max_samples_for_pca, len(all_sequences_array))]
            for seq in sample_seqs:
                for i in range(sequence_length):
                    for j in range(i + 1, sequence_length):
                        if seq[i] != seq[j]:
                            disagreement_matrix[i, j] += 1
            avg_positional_disagreement = float(np.mean(disagreement_matrix[np.triu_indices(sequence_length, k=1)]))
        else:
            avg_positional_disagreement = 0.0
        
        # Class imbalance features
        class_imbalance_ratio = float(np.max(freq_arr) / (np.mean(freq_arr) + 1e-10))
        minority_class_ratio = float(np.sum(freq_arr > 0) / num_classes)
        
        # Normalized entropy
        normalized_entropy = float(entropy(freq_probs) / np.log(num_classes)) if num_classes > 1 else 0.0
        
        return {
            # Basic statistics
            'n_samples': total_samples,
            'sequence_length': sequence_length,
            'global_num_classes': int((freq_arr > 0).sum()),
            'avg_unique_per_sample': float(np.mean(unique_per_sample)),
            'token_density': float(np.mean(unique_per_sample) / sequence_length) if sequence_length > 0 else 0.0,
            'class_frequency_std': float(np.std(freq_arr)),
            'token_frequency_entropy': float(entropy(freq_probs)),
            'token_frequency_kurtosis': float(kurtosis(freq_arr)),
            'mean_pairwise_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
            'mean_token_repetition': float(np.mean(token_repetitions)),
            'autocorrelation_lag1': float(autocorr_lag1),
            'token_position_entropy': float(token_position_entropy),
            
            # Dimensionality features
            'dimension': dimension,
            'effective_dimension': effective_dimension,
            'dimension_ratio': float(dimension_ratio),
            
            # Ensemble-specific features
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'avg_prediction_margin': avg_prediction_margin,
            'avg_agreement': avg_agreement,
            
            # Distribution shape
            'freq_skewness': freq_skewness,
            'freq_gini': freq_gini,
            'normalized_entropy': normalized_entropy,
            
            # Spatial/structural
            'avg_hamming_distance': avg_hamming_distance,
            'effective_rank': effective_rank,
            'spectral_entropy': spectral_entropy,
            
            # Positional
            'position_variance': position_variance,
            'avg_positional_disagreement': avg_positional_disagreement,
            
            # Class imbalance
            'class_imbalance_ratio': class_imbalance_ratio,
            'minority_class_ratio': minority_class_ratio,
        }
    
    def extract_as_vector(self, predictions: NDArray[np.integer]) -> NDArray[np.floating]:
        """Extract features as a 1D numpy array in canonical order.
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Ensemble predictions
            
        Returns
        -------
        NDArray
            Feature vector of shape (28,) in order of FEATURE_NAMES
        """
        features = self.extract(predictions)
        return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    
    @staticmethod
    def get_feature_names() -> list[str]:
        """Get ordered list of feature names."""
        return FEATURE_NAMES.copy()
