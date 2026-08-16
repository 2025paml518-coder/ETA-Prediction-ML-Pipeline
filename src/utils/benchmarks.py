"""Benchmark utilities for Week 1 pipeline.

Measures per-stage runtime, data volumes, and feature pipeline latency.
Results are stored in reports/benchmarks.json for reproducibility tracking.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

BENCHMARK_OUTPUT = Path("reports/benchmarks.json")


class Benchmark:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000


def measure_data_volumes() -> dict[str, Any]:
    """Measure rows and file sizes at each stage."""
    stages = {
        'raw': 'data/raw/trips_raw.parquet',
        'validated': 'data/interim/trips_validated.parquet',
        'quarantined': 'data/quarantine/quarantined_trips.parquet',
        'train': 'data/interim/train.parquet',
        'val': 'data/interim/val.parquet',
        'test': 'data/interim/test.parquet',
        'train_features': 'data/processed/train_features.parquet',
        'val_features': 'data/processed/val_features.parquet',
        'test_features': 'data/processed/test_features.parquet',
    }

    volumes = {}
    for name, path in stages.items():
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                file_size_mb = p.stat().st_size / (1024 * 1024)
                volumes[name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file_size_mb': round(file_size_mb, 2),
                    'bytes_per_row': round((p.stat().st_size / len(df)), 0) if len(df) > 0 else 0,
                }
            except Exception as e:
                volumes[name] = {'error': str(e)}
        else:
            volumes[name] = {'error': 'File not found'}

    return volumes


def measure_feature_pipeline_latency() -> dict[str, Any]:
    """Measure single-row and batch feature pipeline latency."""
    try:
        from src.features.build_features import FeaturePipeline
        
        pipeline = FeaturePipeline.load('models/feature_pipeline')
        
        # Create a synthetic single row
        sample_row = pd.DataFrame({
            'pickup_datetime': ['2024-01-15 09:30:00'],
            'pickup_latitude': [40.7580],
            'pickup_longitude': [-73.9855],
            'dropoff_latitude': [40.7614],
            'dropoff_longitude': [-73.9776],
            'vendor_id': [1],
            'passenger_count': [1],
            'store_and_forward': [False],
            'trip_distance_miles': [0.5],
            'traffic_index': [0.5],
            'temperature_c': [15.0],
            'precipitation_mm': [0.0],
            'wind_kph': [10.0],
            'weather': ['Clear'],
        })
        
        # Warm up
        _ = pipeline.transform(sample_row)
        
        # Time single row
        start = time.perf_counter()
        for _ in range(100):
            _ = pipeline.transform(sample_row)
        single_row_ms = (time.perf_counter() - start) * 1000 / 100
        
        # Create batch
        batch = pd.concat([sample_row] * 1000, ignore_index=True)
        
        start = time.perf_counter()
        _ = pipeline.transform(batch)
        batch_ms = (time.perf_counter() - start) * 1000
        
        return {
            'single_row_ms': round(single_row_ms, 4),
            'batch_1k_ms': round(batch_ms, 2),
            'batch_1k_rows_per_sec': round(1000 / (batch_ms / 1000), 0),
        }
    except Exception as e:
        return {'error': str(e)}


def load_benchmarks() -> dict[str, Any]:
    """Load existing benchmarks or return empty."""
    if BENCHMARK_OUTPUT.exists():
        with open(BENCHMARK_OUTPUT) as f:
            return json.load(f)
    return {}


def save_benchmarks(benchmarks: dict[str, Any]) -> None:
    """Save benchmarks to reports/benchmarks.json."""
    BENCHMARK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(BENCHMARK_OUTPUT, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"✓ Benchmarks saved to {BENCHMARK_OUTPUT}")


def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmarks and save to file."""
    print("⏱️  Running Week 1 benchmarks...")
    
    benchmarks = {
        'timestamp': datetime.now().isoformat(),
        'data_volumes': measure_data_volumes(),
        'feature_pipeline': measure_feature_pipeline_latency(),
    }
    
    save_benchmarks(benchmarks)
    
    # Print summary
    print("\n📊 Data Volumes:")
    for stage, metrics in benchmarks['data_volumes'].items():
        if 'error' not in metrics:
            print(
                f"  {stage:20} "
                f"{metrics['rows']:>9,} rows  "
                f"{metrics['columns']:>3} cols  "
                f"{metrics['file_size_mb']:>7.1f} MB"
            )
    
    print("\n⚡ Feature Pipeline Latency:")
    if 'error' not in benchmarks['feature_pipeline']:
        fp = benchmarks['feature_pipeline']
        print(f"  Single row:        {fp['single_row_ms']:.4f} ms")
        print(f"  Batch (1k rows):   {fp['batch_1k_ms']:.2f} ms")
        print(f"  Throughput:        {fp['batch_1k_rows_per_sec']:.0f} rows/sec")
    else:
        print(f"  Error: {benchmarks['feature_pipeline']['error']}")
    
    return benchmarks


if __name__ == '__main__':
    run_all_benchmarks()
