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
    add_zurich_csv=False,
    zurich_csv_path=None,
) -> pd.DataFrame:
    """
    Load flight data from CSV, filter flights with at least `min_rows` rows,
    convert latitude/longitude to meters, and concatenate `num_flights` of them
    side by side using the shortest flight length.
    Adds a sequential block index column.
    
    Optionally, include the Zurich dataset as pseudo-trajectories split by min_rows.

    Args:
        csv_path (str): Path to the CSV file.
        min_rows (int): Minimum number of rows a flight must have to be selected.
        num_flights (int): Number of flights to concatenate side by side.
        position_columns (list, optional): Columns to extract (default ['position_x', 'position_y', 'position_z']).
        add_zurich_csv (bool): If True, append Zurich CSV as pseudo-trajectories.
        zurich_csv_path (str, optional): Path to Zurich CSV file.

    Returns:
        pd.DataFrame: Concatenated DataFrame with selected flights and optionally Zurich blocks.
    """

    if position_columns is None:
        position_columns = ["position_x", "position_y", "position_z"]

    # --- Process main CSV ---
    df = pd.read_csv(csv_path)
    grouped = df.groupby("flight")
    converted_flights = []

    for _, group in grouped:
        if len(group) < min_rows:
            continue

        x_m, y_m = latlon_to_meters(
            group["position_y"],  # latitude
            group["position_x"],  # longitude
            ref_lat=group["position_y"].iloc[0],
            ref_lon=group["position_x"].iloc[0],
        )

        group_copy = group.copy()
        group_copy["position_x"] = x_m
        group_copy["position_y"] = y_m
        converted_flights.append(group_copy[position_columns].reset_index(drop=True))

    if len(converted_flights) < num_flights:
        raise ValueError(
            f"Not enough flights with at least {min_rows} rows. Found {len(converted_flights)}."
        )

    concatenated_blocks = []
    trajectory_counter = 1

    # --- Create blocks from main CSV ---
    for i in range(0, len(converted_flights) - num_flights + 1):
        block_flights = converted_flights[i : i + num_flights]
        min_len = min(f.shape[0] for f in block_flights)
        truncated_dfs = [f.iloc[:min_len] for f in block_flights]
        concatenated = pd.concat(truncated_dfs, axis=1)

        # Rename columns
        new_columns = []
        for j, f in enumerate(truncated_dfs, start=1):
            new_columns.extend([f"{col}_flight{j}" for col in position_columns])
        concatenated.columns = new_columns

        concatenated["trajectory_index"] = trajectory_counter
        trajectory_counter += 1
        concatenated_blocks.append(concatenated)

    # --- Optionally process Zurich dataset ---
    if add_zurich_csv and zurich_csv_path is not None:
        zurich_df = pd.read_csv(zurich_csv_path, sep=None, engine="python")
        zurich_df.columns = zurich_df.columns.str.strip()
        print(zurich_df.columns.tolist())

        # Convert lat/lon to meters using first row as reference
        x_m, y_m = latlon_to_meters(
            zurich_df["lat"],
            zurich_df["lon"],
            ref_lat=zurich_df["lat"].iloc[0],
            ref_lon=zurich_df["lon"].iloc[0],
        )
        zurich_df = zurich_df.copy()
        zurich_df["position_x"] = x_m
        zurich_df["position_y"] = y_m
        zurich_df["position_z"] = zurich_df["alt"]

        # Only keep positional columns
        zurich_positions = zurich_df[position_columns].reset_index(drop=True)
        total_rows = zurich_positions.shape[0]

        # Split Zurich sequentially into non-overlapping pseudo-trajectories
        pseudo_trajectories = [
            zurich_positions.iloc[i : i + min_rows].reset_index(drop=True)
            for i in range(0, total_rows, min_rows)
            if i + min_rows <= total_rows  # drop last incomplete chunk
        ]

        # Create blocks from sequential pseudo-trajectories
        for block_start in range(0, len(pseudo_trajectories) - num_flights + 1):
            block_flights = pseudo_trajectories[block_start : block_start + num_flights]
            min_len = min(f.shape[0] for f in block_flights)
            truncated_dfs = [f.iloc[:min_len] for f in block_flights]
            concatenated = pd.concat(truncated_dfs, axis=1)

            # Rename columns
            new_columns = []
            for j in range(1, num_flights + 1):
                new_columns.extend([f"{col}_flight{j}" for col in position_columns])
            concatenated.columns = new_columns

            concatenated["trajectory_index"] = trajectory_counter
            trajectory_counter += 1
            concatenated_blocks.append(concatenated)

    # Combine all blocks into single DataFrame
    final_df = pd.concat(concatenated_blocks, ignore_index=True)
    return final_df
