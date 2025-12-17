"""
Compare Distribution: 10K Sample vs 11M Full Training Data
Script untuk membuktikan bahwa sample 10,000 baris representatif terhadap 11 juta baris
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FULL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
SAMPLE_DATA_PATH = PROJECT_ROOT / "src" / "serving" / "reference_sample.parquet"

# Features to compare
DRIFT_FEATURES = ['trip_distance', 'passenger_count', 'pickup_hour', 'pickup_dayofweek']


def compare_distributions():
    """Compare statistics between full data and sample."""
    
    print("=" * 70)
    print("PERBANDINGAN DISTRIBUSI: 10K SAMPLE vs 11M FULL DATA")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    
    print(f"    Loading full data: {FULL_DATA_PATH}")
    full_df = pd.read_parquet(FULL_DATA_PATH)
    print(f"    Full data rows: {len(full_df):,}")
    
    print(f"    Loading sample data: {SAMPLE_DATA_PATH}")
    sample_df = pd.read_parquet(SAMPLE_DATA_PATH)
    print(f"    Sample data rows: {len(sample_df):,}")
    
    # Filter full data to only include drift features
    full_df = full_df[[f for f in DRIFT_FEATURES if f in full_df.columns]]
    
    print("\n" + "=" * 70)
    print("[2] PERBANDINGAN STATISTIK PER FITUR")
    print("=" * 70)
    
    results = []
    
    for feature in DRIFT_FEATURES:
        if feature not in full_df.columns or feature not in sample_df.columns:
            print(f"\n    [SKIP] {feature} - tidak ada di salah satu dataset")
            continue
        
        full_values = full_df[feature].dropna()
        sample_values = sample_df[feature].dropna()
        
        # Calculate statistics
        full_mean = full_values.mean()
        sample_mean = sample_values.mean()
        mean_diff_pct = abs(full_mean - sample_mean) / full_mean * 100 if full_mean != 0 else 0
        
        full_std = full_values.std()
        sample_std = sample_values.std()
        std_diff_pct = abs(full_std - sample_std) / full_std * 100 if full_std != 0 else 0
        
        full_median = full_values.median()
        sample_median = sample_values.median()
        median_diff_pct = abs(full_median - sample_median) / full_median * 100 if full_median != 0 else 0
        
        # Kolmogorov-Smirnov test (statistical test untuk kesamaan distribusi)
        # p-value > 0.05 berarti distribusi SAMA
        ks_stat, ks_pvalue = stats.ks_2samp(
            full_values.sample(n=min(10000, len(full_values)), random_state=42),
            sample_values
        )
        
        results.append({
            'feature': feature,
            'mean_diff_pct': mean_diff_pct,
            'std_diff_pct': std_diff_pct,
            'ks_pvalue': ks_pvalue
        })
        
        print(f"\n    FITUR: {feature}")
        print(f"    " + "-" * 50)
        print(f"    {'Statistik':<20} {'Full (11M)':<15} {'Sample (10K)':<15} {'Diff %':<10}")
        print(f"    " + "-" * 50)
        print(f"    {'Mean':<20} {full_mean:<15.4f} {sample_mean:<15.4f} {mean_diff_pct:<10.2f}%")
        print(f"    {'Std Dev':<20} {full_std:<15.4f} {sample_std:<15.4f} {std_diff_pct:<10.2f}%")
        print(f"    {'Median':<20} {full_median:<15.4f} {sample_median:<15.4f} {median_diff_pct:<10.2f}%")
        print(f"    {'Min':<20} {full_values.min():<15.4f} {sample_values.min():<15.4f}")
        print(f"    {'Max':<20} {full_values.max():<15.4f} {sample_values.max():<15.4f}")
        print(f"    " + "-" * 50)
        print(f"    KS Test p-value: {ks_pvalue:.4f}", end="")
        if ks_pvalue > 0.05:
            print(" --> DISTRIBUSI SAMA (p > 0.05)")
        else:
            print(" --> Distribusi berbeda (p <= 0.05)")
    
    # Summary
    print("\n" + "=" * 70)
    print("[3] RINGKASAN")
    print("=" * 70)
    
    avg_mean_diff = np.mean([r['mean_diff_pct'] for r in results])
    avg_std_diff = np.mean([r['std_diff_pct'] for r in results])
    same_distribution_count = sum(1 for r in results if r['ks_pvalue'] > 0.05)
    
    print(f"\n    Rata-rata perbedaan Mean:    {avg_mean_diff:.2f}%")
    print(f"    Rata-rata perbedaan Std Dev: {avg_std_diff:.2f}%")
    print(f"    Fitur dengan distribusi sama: {same_distribution_count}/{len(results)}")
    
    print("\n" + "=" * 70)
    print("[4] KESIMPULAN")
    print("=" * 70)
    
    if avg_mean_diff < 5 and same_distribution_count >= len(results) * 0.75:
        print("\n    HASIL: Sample 10K REPRESENTATIF terhadap data 11M")
        print("    - Perbedaan mean < 5%")
        print("    - Mayoritas fitur memiliki distribusi yang sama secara statistik")
        print("    - Sample dapat digunakan untuk drift detection dengan confidence tinggi")
    else:
        print("\n    HASIL: Sample 10K KURANG REPRESENTATIF")
        print("    - Pertimbangkan untuk membuat sample baru dengan ukuran lebih besar")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_distributions()
