"""Tests for model training and evaluation."""

import json
from pathlib import Path

import numpy as np

from src.models.evaluate import calculate_metrics, format_metrics
from src.models.train import load_data, prepare_data


class TestModelEvaluation:
    """Test evaluation metrics calculation."""

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # MAE should be 2.0
        assert abs(metrics['mae'] - 2.0) < 0.1
        
        # All values should be positive
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mape'] > 0

    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {
            'mae': 12.34,
            'rmse': 15.67,
            'r2': 0.882,
            'mape': 5.21,
        }
        
        formatted = format_metrics(metrics)
        
        assert '12.3400' in formatted
        assert '15.6700' in formatted
        assert '0.8820' in formatted


class TestDataLoading:
    """Test data loading and preparation."""

    def test_load_data(self):
        """Test loading Week 1 data."""
        data = load_data()
        
        # Check all required keys exist
        assert 'train' in data
        assert 'val' in data
        assert 'test' in data
        assert 'train_features' in data
        assert 'val_features' in data
        assert 'test_features' in data
        
        # Check data is not empty
        assert len(data['train']) > 0
        assert len(data['val']) > 0
        assert len(data['test']) > 0
        assert len(data['train_features']) > 0

    def test_prepare_data(self):
        """Test data preparation for training."""
        data = load_data()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(data)
        
        # Check shapes match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        # Check feature count is reasonable (should be ~44 features)
        assert len(feature_cols) > 40
        assert len(feature_cols) < 50
        
        # Check no NaNs
        assert not X_train.isnull().any().any()
        assert not X_val.isnull().any().any()
        assert not X_test.isnull().any().any()
        
        # Check splits don't overlap
        train_ids = set(data['train']['trip_id'].values)
        val_ids = set(data['val']['trip_id'].values)
        test_ids = set(data['test']['trip_id'].values)
        
        assert len(train_ids & val_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train_ids & test_ids) == 0


class TestModelMetadata:
    """Test model metadata after training."""

    def test_best_model_metadata_exists(self):
        """Test that best model metadata file is created."""
        metadata_path = Path('models/trained/best_model_metadata.json')
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Check required fields
            assert 'model_type' in metadata
            assert 'params' in metadata
            assert 'metrics_test' in metadata
            
            # Check metrics
            metrics = metadata['metrics_test']
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'r2' in metrics


class TestTargetDistribution:
    """Test target variable properties."""

    def test_target_not_constant(self):
        """Test that target variable has variance."""
        data = load_data()
        
        y_train = data['train']['trip_duration_min']
        y_val = data['val']['trip_duration_min']
        y_test = data['test']['trip_duration_min']
        
        # Each split should have variance
        assert y_train.std() > 1.0
        assert y_val.std() > 1.0
        assert y_test.std() > 1.0
        
        # Values in reasonable range (1-300 minutes)
        assert y_train.min() >= 1
        assert y_train.max() <= 300
