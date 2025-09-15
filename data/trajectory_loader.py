"""
trajectory_loader.py

This module provides utilities to load flight trajectory data from a CSV file
and prepare it for model training. Specifically, it extracts position information
(x, y, z) from flights that meet a minimum length requirement, converts
latitude/longitude to meters, concatenates multiple flights side by side, truncates
to the shortest flight to avoid NaNs, and adds a sequential block index for easier
plotting or analysis.
"""

import pandas as pd
from utils.coordinate_converter import latlon_to_meters


def load_and_concat_flights(
    csv_path,
    min_rows=1000,
    num_flights=3,
    position_columns=None,
) -> pd.DataFrame:
    """
    Load flight data from CSV, filter flights with at least `min_rows` rows,
    convert latitude/longitude to meters, and concatenate `num_flights` of them
    side by side using the shortest flight length.
    Adds a sequential block index column.

    Args:
        csv_path (str): Path to the CSV file.
        min_rows (int): Minimum number of rows a flight must have to be selected.
        num_flights (int): Number of flights to concatenate side by side.
        position_columns (list, optional): Columns to extract (default ['position_x', 'position_y', 'position_z']).

    Returns:
        pd.DataFrame: Concatenated DataFrame with selected flights side by side.
                    Columns are renamed as {column}_flight{i}, and an additional
                    'block_index' column is added with sequential integers.

    Raises:
        ValueError: If not enough flights meet the minimum row requirement.
    """

    if position_columns is None:
        position_columns = ["position_x", "position_y", "position_z"]

    # Load CSV
    df = pd.read_csv(csv_path)

    # Convert lat/lon to meters using first point of each flight as reference
    grouped = df.groupby("flight")
    converted_flights = []
    for _, group in grouped:
        if len(group) < min_rows:
            continue

        # Convert lat/lon to meters
        x_m, y_m = latlon_to_meters(
            group["position_y"],  # latitude
            group["position_x"],  # longitude
            ref_lat=group["position_y"].iloc[0],
            ref_lon=group["position_x"].iloc[0],
        )

        group_copy = group.copy()
        group_copy["position_x"] = x_m
        group_copy["position_y"] = y_m

        # Keep only the requested position columns
        converted_flights.append(group_copy[position_columns].reset_index(drop=True))

    if len(converted_flights) < num_flights:
        raise ValueError(
            f"Not enough flights with at least {min_rows} rows. Found {len(converted_flights)}."
        )

    concatenated_blocks = []

    # Create multiple blocks in sliding window style
    for block_idx, i in enumerate(
        range(0, len(converted_flights) - num_flights + 1), start=1
    ):
        block_flights = converted_flights[i : i + num_flights]
        min_len = min(f.shape[0] for f in block_flights)
        truncated_dfs = [f.iloc[:min_len] for f in block_flights]

        # Concatenate side by side
        concatenated = pd.concat(truncated_dfs, axis=1)

        # Rename columns to indicate flight number
        new_columns = []
        for j, f in enumerate(truncated_dfs, start=1):
            new_columns.extend([f"{col}_flight{j}" for col in position_columns])
        concatenated.columns = new_columns

        # Add sequential block index
        concatenated["trajectory_index"] = block_idx
        concatenated_blocks.append(concatenated)

    # Combine all blocks into a single DataFrame with sequential index
    final_df = pd.concat(concatenated_blocks, ignore_index=True)
    return final_df
