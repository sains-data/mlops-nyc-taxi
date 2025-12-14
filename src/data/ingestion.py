"""
Data Ingestion Module
Downloads and loads NYC Taxi Yellow Tripdata
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DATA_DIR = Path("data/raw")


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        logger.info(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_nyc_taxi_data(
    year: int = 2024,
    months: list[int] = [1, 2, 3],
    data_dir: Optional[Path] = None
) -> list[Path]:
    """
    Download NYC Yellow Taxi trip data for specified year and months.
    
    Args:
        year: Year of data (2009-2024)
        months: List of months (1-12)
        data_dir: Directory to save data
        
    Returns:
        List of downloaded file paths
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for month in months:
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            downloaded_files.append(filepath)
            continue
        
        url = f"{BASE_URL}/{filename}"
        logger.info(f"Downloading: {url}")
        
        if download_file(url, filepath):
            downloaded_files.append(filepath)
    
    return downloaded_files


def load_parquet_data(
    filepath: Path,
    columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load parquet file into pandas DataFrame."""
    logger.info(f"Loading: {filepath}")
    
    if columns:
        df = pd.read_parquet(filepath, columns=columns)
    else:
        df = pd.read_parquet(filepath)
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_multiple_parquet(
    filepaths: list[Path],
    columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load and concatenate multiple parquet files."""
    dfs = []
    
    for filepath in filepaths:
        df = load_parquet_data(filepath, columns)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df):,} rows")
    
    return combined_df


if __name__ == "__main__":
    # Example usage
    files = download_nyc_taxi_data(year=2024, months=[1])
    if files:
        df = load_parquet_data(files[0])
        print(df.head())
        print(df.info())
