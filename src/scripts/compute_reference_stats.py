
import pandas as pd
import json
import numpy as np
from pathlib import Path

def compute_stats():
    # Paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_PATH = ROOT_DIR / "data" / "processed" / "train.parquet"
    OUTPUT_PATH = ROOT_DIR / "src" / "serving" / "reference_stats.json"
    
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found!")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    # Features to monitor
    features = ['trip_distance', 'passenger_count', 'pickup_hour', 'pickup_dayofweek', 'fare_amount']
    
    stats = {}
    
    for feat in features:
        if feat in df.columns:
            # Basic stats
            col_data = df[feat]
            mean = float(col_data.mean())
            std = float(col_data.std())
            
            # Histogram for distribution (simplified for lightweight JSON)
            hist, bin_edges = np.histogram(col_data, bins=10, density=True)
            
            stats[feat] = {
                "mean": mean,
                "std": std,
                "hist": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "min": float(col_data.min()),
                "max": float(col_data.max())
            }
            print(f"Computed stats for {feat}")
            
    # Function to convert simple stats for frontend
    # We save comprehensive stats, but frontend might just need bins
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving stats to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    compute_stats()
