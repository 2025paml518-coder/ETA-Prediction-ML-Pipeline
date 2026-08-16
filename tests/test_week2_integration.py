"""Integration tests for Week 2 handoff.

These tests verify that Week 1 outputs are ready to feed into model training.
Run after `dvc repro` to ensure the feature matrix is in the expected format.
"""

import json

import numpy as np
import pandas as pd
import pytest


class TestWeek2Integration:
    """Verify Week 1 outputs are ready for Week 2 model training."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load all required outputs."""
        self.train_features = pd.read_parquet('data/processed/train_features.parquet')
        self.val_features = pd.read_parquet('data/processed/val_features.parquet')
        self.test_features = pd.read_parquet('data/processed/test_features.parquet')
        
        self.train = pd.read_parquet('data/interim/train.parquet')
        self.val = pd.read_parquet('data/interim/val.parquet')
        self.test = pd.read_parquet('data/interim/test.parquet')
        
        with open('models/feature_pipeline/feature_spec.json') as f:
            self.feature_spec = json.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Matrix Integrity
    # ─────────────────────────────────────────────────────────────────────────

    def test_feature_matrix_dimensions(self):
        """Feature matrices must have consistent dimensions across splits."""
        assert len(self.train_features.columns) > 0, "Train features empty"
        assert len(self.val_features.columns) > 0, "Val features empty"
        assert len(self.test_features.columns) > 0, "Test features empty"
        
        assert len(self.train_features.columns) == len(self.val_features.columns), \
            f"Train has {len(self.train_features.columns)}, Val has {len(self.val_features.columns)}"
        
        assert len(self.train_features.columns) == len(self.test_features.columns), \
            f"Train has {len(self.train_features.columns)}, Test has {len(self.test_features.columns)}"

    def test_feature_names_match_spec(self):
        """Feature column names must match feature_spec.json."""
        expected_cols = set(self.feature_spec['feature_columns'])
        actual_cols = set(self.train_features.columns)
        
        # Remove metadata columns that might be included alongside features
        metadata_cols = {'trip_id', 'pickup_datetime', 'trip_duration_min', 'target'}
        actual_feature_cols = actual_cols - metadata_cols
        
        missing = expected_cols - actual_feature_cols
        extra = actual_feature_cols - expected_cols
        
        assert not missing, f"Missing columns in train_features: {missing}"
        assert not extra, f"Extra columns in train_features: {extra}"

    def test_feature_column_order(self):
        """Feature column order must be deterministic (matches spec for the feature columns)."""
        expected_order = self.feature_spec['feature_columns']
        
        # Get actual features (may have metadata cols at end)
        actual_all = list(self.train_features.columns)
        
        # Extract just the features (in order) from the actual columns
        actual_features = [col for col in actual_all if col in expected_order]
        
        assert actual_features == expected_order, \
            f"Feature order mismatch. Expected: {expected_order[:5]}..., Got: {actual_features[:5]}..."

    def test_feature_dtypes_numeric(self):
        """All expected feature columns must be numeric (float64 or int64)."""
        # Only check the expected features from spec, not all columns
        expected_features = self.feature_spec['feature_columns']
        actual_features = self.train_features[expected_features]
        non_numeric = actual_features.select_dtypes(exclude=['number']).columns.tolist()
        assert not non_numeric, f"Non-numeric columns found in expected features: {non_numeric}"

    # ─────────────────────────────────────────────────────────────────────────
    # NaN and Null Leakage
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_nulls_in_features(self):
        """Feature matrices must have no NaN values."""
        train_nulls = self.train_features.isnull().sum().sum()
        val_nulls = self.val_features.isnull().sum().sum()
        test_nulls = self.test_features.isnull().sum().sum()
        
        assert train_nulls == 0, f"Train features have {train_nulls} NaNs"
        assert val_nulls == 0, f"Val features have {val_nulls} NaNs"
        assert test_nulls == 0, f"Test features have {test_nulls} NaNs"

    def test_no_inf_in_features(self):
        """Feature matrices must have no infinite values."""
        # Select only numeric columns to avoid type errors
        numeric_train = self.train_features.select_dtypes(include=['number'])
        numeric_val = self.val_features.select_dtypes(include=['number'])
        numeric_test = self.test_features.select_dtypes(include=['number'])
        
        train_infs = np.isinf(numeric_train.values).sum()
        val_infs = np.isinf(numeric_val.values).sum()
        test_infs = np.isinf(numeric_test.values).sum()
        
        assert train_infs == 0, f"Train features have {train_infs} infs"
        assert val_infs == 0, f"Val features have {val_infs} infs"
        assert test_infs == 0, f"Test features have {test_infs} infs"

    def test_target_no_nulls(self):
        """Target variable must be complete (no NaNs)."""
        train_nulls = self.train['trip_duration_min'].isnull().sum()
        val_nulls = self.val['trip_duration_min'].isnull().sum()
        test_nulls = self.test['trip_duration_min'].isnull().sum()
        
        assert train_nulls == 0, f"Train target has {train_nulls} NaNs"
        assert val_nulls == 0, f"Val target has {val_nulls} NaNs"
        assert test_nulls == 0, f"Test target has {test_nulls} NaNs"

    # ─────────────────────────────────────────────────────────────────────────
    # Target Range and Distribution
    # ─────────────────────────────────────────────────────────────────────────

    def test_target_range(self):
        """Target must be in expected range (1-300 minutes)."""
        min_val, max_val = 1.0, 300.0
        
        train_min = self.train['trip_duration_min'].min()
        train_max = self.train['trip_duration_min'].max()
        
        assert train_min >= min_val, f"Train target min {train_min} < {min_val}"
        assert train_max <= max_val, f"Train target max {train_max} > {max_val}"

    def test_target_has_variance(self):
        """Target must have sufficient variance for modeling."""
        train_std = self.train['trip_duration_min'].std()
        assert train_std > 1.0, f"Train target std {train_std} too low (constant data?)"

    def test_target_not_constant(self):
        """Target values must not all be identical."""
        train_unique = self.train['trip_duration_min'].nunique()
        assert train_unique > 100, f"Train target has only {train_unique} unique values"

    # ─────────────────────────────────────────────────────────────────────────
    # Split Proportions
    # ─────────────────────────────────────────────────────────────────────────

    def test_split_proportions(self):
        """Train/val/test split must be approximately 70/15/15."""
        total = len(self.train_features) + len(self.val_features) + len(self.test_features)
        
        train_pct = len(self.train_features) / total
        val_pct = len(self.val_features) / total
        test_pct = len(self.test_features) / total
        
        # Allow ±2% deviation
        assert 0.68 < train_pct < 0.72, f"Train % {train_pct:.1%} not ~70%"
        assert 0.13 < val_pct < 0.17, f"Val % {val_pct:.1%} not ~15%"
        assert 0.13 < test_pct < 0.17, f"Test % {test_pct:.1%} not ~15%"

    def test_no_overlap_in_splits(self):
        """Rows must not appear in multiple splits."""
        train_ids = set(self.train['trip_id'].values)
        val_ids = set(self.val['trip_id'].values)
        test_ids = set(self.test['trip_id'].values)
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        assert not train_val_overlap, f"{len(train_val_overlap)} rows in both train and val"
        assert not train_test_overlap, f"{len(train_test_overlap)} rows in both train and test"
        assert not val_test_overlap, f"{len(val_test_overlap)} rows in both val and test"

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Correctness
    # ─────────────────────────────────────────────────────────────────────────

    def test_temporal_split_order(self):
        """Splits must be in temporal order (train < val < test)."""
        train_max = pd.to_datetime(self.train['pickup_datetime']).max()
        val_min = pd.to_datetime(self.val['pickup_datetime']).min()
        val_max = pd.to_datetime(self.val['pickup_datetime']).max()
        test_min = pd.to_datetime(self.test['pickup_datetime']).min()
        
        assert train_max <= val_min, "Train dates overlap with Val"
        assert val_max <= test_min, "Val dates overlap with Test"

    # ─────────────────────────────────────────────────────────────────────────
    # Row Counts
    # ─────────────────────────────────────────────────────────────────────────

    def test_row_counts_match(self):
        """Feature matrix rows must match raw data rows."""
        assert len(self.train_features) == len(self.train), \
            f"Train features has {len(self.train_features)} rows, train has {len(self.train)}"
        
        assert len(self.val_features) == len(self.val), \
            f"Val features has {len(self.val_features)} rows, val has {len(self.val)}"
        
        assert len(self.test_features) == len(self.test), \
            f"Test features has {len(self.test_features)} rows, test has {len(self.test)}"

    def test_minimum_rows_for_training(self):
        """Training set must have sufficient rows."""
        min_rows = 10000
        assert len(self.train_features) >= min_rows, \
            f"Train set has {len(self.train_features)} rows, need at least {min_rows}"

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Spec Validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_feature_spec_completeness(self):
        """feature_spec.json must have required fields."""
        required_fields = ['version', 'feature_columns', 'imputation_strategies', 'n_zone_clusters']
        
        for field in required_fields:
            assert field in self.feature_spec, f"Missing field in feature_spec: {field}"

    def test_feature_spec_version(self):
        """feature_spec.json version must be defined."""
        assert 'version' in self.feature_spec, "No version in feature_spec"
        assert self.feature_spec['version'], "Version is empty"

    def test_zone_centroids_exist(self):
        """Zone cluster configuration must be persisted for serving."""
        assert 'n_zone_clusters' in self.feature_spec, "No n_zone_clusters in feature_spec"
        assert self.feature_spec['n_zone_clusters'] > 0, "n_zone_clusters is zero"

    def test_imputation_values_exist(self):
        """Imputation strategies must be persisted for serving."""
        assert 'imputation_strategies' in self.feature_spec, "No imputation_strategies in feature_spec"
        assert len(self.feature_spec['imputation_strategies']) > 0, "imputation_strategies empty"

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Value Ranges
    # ─────────────────────────────────────────────────────────────────────────

    def test_feature_values_reasonable(self):
        """Features must have reasonable ranges (not all zeros or ones)."""
        # Only check numeric features
        expected_features = self.feature_spec['feature_columns']
        features_to_check = self.train_features[expected_features].select_dtypes(include=['number'])
        
        stds = features_to_check.std()
        
        # Each feature must have some variation
        for col in features_to_check.columns:
            assert stds[col] > 0.001, f"Feature {col} has near-zero variance"
            # Check for common issues: all zeros, all ones, all same value
            unique_vals = features_to_check[col].nunique()
            assert unique_vals > 1, f"Feature {col} is constant"


class TestDataQualityMetrics:
    """Verify data quality metrics from Week 1."""

    def test_validated_data_exists(self):
        """Validated data must exist and be non-empty."""
        validated = pd.read_parquet('data/interim/trips_validated.parquet')
        assert len(validated) > 0, "Validated data is empty"

    def test_quarantine_data_exists(self):
        """Quarantine data must exist (may be empty, but file should exist)."""
        quarantine = pd.read_parquet('data/quarantine/quarantined_trips.parquet')
        # OK if empty, but file should exist and be readable
        assert isinstance(quarantine, pd.DataFrame)

    def test_quarantine_rate_reasonable(self):
        """Quarantine rate should be in expected range."""
        raw = pd.read_parquet('data/raw/trips_raw.parquet')
        quarantine = pd.read_parquet('data/quarantine/quarantined_trips.parquet')
        
        quarantine_rate = len(quarantine) / len(raw)
        
        # Should be between 1% and 15% (per params.yaml)
        assert 0.01 <= quarantine_rate <= 0.15, \
            f"Quarantine rate {quarantine_rate:.1%} outside expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
