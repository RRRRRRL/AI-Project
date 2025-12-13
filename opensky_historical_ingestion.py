#!/usr/bin/env python3
"""
OpenSky Historical Ingestion Script

This script ingests historical flight data from OpenSky parquet files,
preprocesses it, and integrates it with the existing pipeline.

The script:
1. Loads historical data from OpenSky parquet files
2. Parses columns: icao24, latitude, longitude, altitude, time_position
3. Preprocesses data (drops missing values, converts timestamps, filters invalid values)
4. Saves processed data as CSV in data/processed/ directory

Output CSV format compatible with prepare_sequences.py:
- flight_id (string): aircraft identifier (icao24)
- timestamp (int): unix seconds
- lat (float): latitude in degrees
- lon (float): longitude in degrees
- alt (float): altitude in meters
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_pyarrow():
    """Check if pyarrow is installed for reading parquet files."""
    try:
        import pyarrow
        return True
    except ImportError:
        logger.error("pyarrow is not installed. Please install it using: pip install pyarrow")
        return False


def load_parquet_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from an OpenSky parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame with the loaded data, or None if loading fails
    """
    logger.info(f"Loading parquet file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        logger.info("Please download OpenSky historical data from: https://opensky-network.org/datasets/states/")
        logger.info("Expected format: OpenSky State Vectors (parquet files)")
        logger.info("Required columns: icao24, lat/latitude, lon/longitude, altitude/geoaltitude/baroaltitude, time/timestamp")
        return None
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        logger.info(f"Columns found: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        return None


def parse_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and preprocess OpenSky data.
    
    Expected OpenSky columns (with variations):
    - icao24: aircraft identifier
    - lat/latitude: latitude in degrees
    - lon/longitude: longitude in degrees  
    - altitude/baroaltitude/geoaltitude: altitude in meters
    - velocity: velocity in m/s
    - heading/track: heading in degrees
    - time/timestamp/time_position: unix timestamp
    
    Args:
        df: Raw DataFrame from parquet file
        
    Returns:
        Preprocessed DataFrame with columns: flight_id, timestamp, lat, lon, alt
    """
    logger.info("Starting data preprocessing...")
    
    original_rows = len(df)
    
    # Map column names to standard names (handle variations)
    column_mapping = {}
    
    # Map icao24 column
    if 'icao24' in df.columns:
        column_mapping['icao24'] = 'flight_id'
    else:
        logger.error("Required column 'icao24' not found")
        return pd.DataFrame()
    
    # Map latitude column
    if 'lat' in df.columns:
        column_mapping['lat'] = 'lat'
    elif 'latitude' in df.columns:
        column_mapping['latitude'] = 'lat'
    else:
        logger.error("Required column 'lat' or 'latitude' not found")
        return pd.DataFrame()
    
    # Map longitude column
    if 'lon' in df.columns:
        column_mapping['lon'] = 'lon'
    elif 'longitude' in df.columns:
        column_mapping['longitude'] = 'lon'
    else:
        logger.error("Required column 'lon' or 'longitude' not found")
        return pd.DataFrame()
    
    # Map altitude column (prefer geoaltitude, then baroaltitude, then altitude)
    if 'geoaltitude' in df.columns:
        column_mapping['geoaltitude'] = 'alt'
    elif 'baroaltitude' in df.columns:
        column_mapping['baroaltitude'] = 'alt'
    elif 'altitude' in df.columns:
        column_mapping['altitude'] = 'alt'
    else:
        logger.error("Required column 'altitude', 'baroaltitude', or 'geoaltitude' not found")
        return pd.DataFrame()
    
    # Map timestamp column
    if 'time' in df.columns:
        column_mapping['time'] = 'timestamp'
    elif 'timestamp' in df.columns:
        column_mapping['timestamp'] = 'timestamp'
    elif 'time_position' in df.columns:
        column_mapping['time_position'] = 'timestamp'
    else:
        logger.error("Required column 'time', 'timestamp', or 'time_position' not found")
        return pd.DataFrame()
    
    # Select and rename columns
    columns_to_keep = list(column_mapping.keys())
    df_processed = df[columns_to_keep].copy()
    df_processed = df_processed.rename(columns=column_mapping)
    
    logger.info(f"Mapped columns: {column_mapping}")
    
    # Drop rows with missing essential values
    logger.info("Dropping rows with missing essential values...")
    df_processed = df_processed.dropna(subset=['flight_id', 'lat', 'lon', 'alt', 'timestamp'])
    logger.info(f"Dropped {original_rows - len(df_processed)} rows with missing values")
    
    if len(df_processed) == 0:
        logger.warning("No data remaining after dropping missing values")
        return df_processed
    
    # Convert timestamp to int (unix seconds)
    logger.info("Converting timestamps to unix seconds...")
    # Handle both unix timestamps and datetime objects
    if df_processed['timestamp'].dtype == 'object' or 'datetime' in str(df_processed['timestamp'].dtype):
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp']).astype(int) // 10**9
    else:
        # For numeric types, round to nearest second (preserves sub-second precision better than truncation)
        df_processed['timestamp'] = pd.to_numeric(df_processed['timestamp'], errors='coerce').round().astype(int)
    
    # Ensure flight_id is string
    df_processed['flight_id'] = df_processed['flight_id'].astype(str)
    
    # Convert numeric columns to float
    logger.info("Converting numeric columns to float...")
    for col in ['lat', 'lon', 'alt']:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Drop any rows where conversion failed
    before_numeric = len(df_processed)
    df_processed = df_processed.dropna(subset=['lat', 'lon', 'alt'])
    if before_numeric - len(df_processed) > 0:
        logger.info(f"Dropped {before_numeric - len(df_processed)} rows with invalid numeric values")
    
    # Filter out invalid coordinates and altitudes
    logger.info("Filtering out invalid coordinates and altitudes...")
    before_filter = len(df_processed)
    df_processed = df_processed[
        (df_processed['lat'] >= -90) & (df_processed['lat'] <= 90) &
        (df_processed['lon'] >= -180) & (df_processed['lon'] <= 180) &
        (df_processed['alt'] >= -500)  # Filter obviously invalid readings (allow low-altitude/sea-level flights)
    ]
    logger.info(f"Filtered out {before_filter - len(df_processed)} rows with invalid values")
    
    # Sort by flight_id and timestamp
    logger.info("Sorting data by flight_id and timestamp...")
    df_processed = df_processed.sort_values(['flight_id', 'timestamp'])
    
    # Remove duplicates (keep last to preserve most recent measurement for same timestamp)
    logger.info("Removing duplicate records...")
    before_dedup = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=['flight_id', 'timestamp'], keep='last')
    logger.info(f"Removed {before_dedup - len(df_processed)} duplicate records")
    
    # Log statistics
    logger.info(f"Preprocessing complete: {len(df_processed)} rows remaining")
    logger.info(f"Data reduction: {original_rows} -> {len(df_processed)} ({100*len(df_processed)/original_rows:.1f}% retained)")
    logger.info(f"Unique flights: {df_processed['flight_id'].nunique()}")
    logger.info(f"Time range: {pd.to_datetime(df_processed['timestamp'].min(), unit='s')} to {pd.to_datetime(df_processed['timestamp'].max(), unit='s')}")
    logger.info(f"Altitude range: {df_processed['alt'].min():.1f}m to {df_processed['alt'].max():.1f}m")
    
    return df_processed


def save_csv(df: pd.DataFrame, output_path: str):
    """
    Save processed data to CSV file.
    
    Args:
        df: Preprocessed DataFrame
        output_path: Path to output CSV file
    """
    logger.info(f"Saving processed data to: {output_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")
    
    # Select only the required columns in the correct order
    columns_to_save = ['flight_id', 'timestamp', 'lat', 'lon', 'alt']
    df_to_save = df[columns_to_save]
    
    # Save to CSV
    df_to_save.to_csv(output_path, index=False)
    
    logger.info(f"Successfully saved {len(df_to_save)} rows to {output_path}")
    logger.info(f"Output format compatible with prepare_sequences.py")


def main():
    """Main function to run the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest historical flight data from OpenSky parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single parquet file
  python opensky_historical_ingestion.py --input data/opensky_states_2024-01.parquet --output data/processed/flights.csv
  
  # Process with custom output location
  python opensky_historical_ingestion.py --input /path/to/opensky_data.parquet --output data/processed/custom_flights.csv

Note:
  OpenSky historical data can be downloaded from:
  https://opensky-network.org/datasets/states/
  
  The output CSV will be compatible with prepare_sequences.py and the rest of the AI pipeline.
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to OpenSky parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/opensky_historical.csv',
        help='Path to output CSV file (default: data/processed/opensky_historical.csv)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenSky Historical Data Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_pyarrow():
        sys.exit(1)
    
    # Load data
    df = load_parquet_file(args.input)
    if df is None:
        logger.error("Failed to load parquet file")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error("Loaded parquet file is empty")
        sys.exit(1)
    
    # Preprocess data
    df_processed = parse_and_preprocess(df)
    
    if len(df_processed) == 0:
        logger.error("No data remaining after preprocessing")
        sys.exit(1)
    
    # Save to CSV
    save_csv(df_processed, args.output)
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Next step: Use the output CSV with prepare_sequences.py")
    logger.info(f"Example: python src/prepare_sequences.py --input_csv {args.output} --output_npz data/sequences.npz")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
