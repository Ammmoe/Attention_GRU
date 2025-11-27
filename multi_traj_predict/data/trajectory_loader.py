"""
trajectory_loader.py

This module provides dataset-specific loaders for flight trajectory data:
- Quadcopter dataset (data/flights.csv)
- Zurich MAV dataset (data/zurich.csv)
- Mixed dataset (combine both)

Each loader prepares the data for model training:
- Converts lat/lon to meters
- Extracts position information (x, y, z)
- Concatenates multiple flights side by side
- Truncates to the shortest flight to avoid NaNs
- Adds a sequential trajectory index for plotting/analysis
"""

import pandas as pd
from multi_traj_predict.utils.coordinate_converter import latlon_to_meters


def _concat_flights_into_blocks(flights, num_flights, position_columns):
    """
    Helper to concatenate flights into blocks of num_flights side by side.
    """
    concatenated_blocks = []
    trajectory_counter = 1

    for i in range(0, len(flights) - num_flights + 1):
        block_flights = flights[i : i + num_flights]
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

    return pd.concat(concatenated_blocks, ignore_index=True)


def load_quadcopter_dataset(
    csv_path="data/flights.csv",
    min_rows=800,
    num_flights=3,
    position_columns=None,
) -> pd.DataFrame:
    """
    Load and process the Quadcopter dataset into a ready-to-use DataFrame.
    """
    if position_columns is None:
        position_columns = ["position_x", "position_y", "position_z"]

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

    return _concat_flights_into_blocks(converted_flights, num_flights, position_columns)


def load_zurich_dataset(
    csv_path="data/zurich_flights_downsampled_2.csv",
    min_rows=800,
    num_flights=3,
    position_columns=None,
) -> pd.DataFrame:
    """
    Load and process the Zurich MAV dataset into a ready-to-use DataFrame.
    """
    if position_columns is None:
        position_columns = ["position_x", "position_y", "position_z"]

    zurich_df = pd.read_csv(csv_path, sep=None, engine="python")
    zurich_df.columns = zurich_df.columns.str.strip()

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

    zurich_positions = zurich_df[position_columns].reset_index(drop=True)
    total_rows = zurich_positions.shape[0]

    # Split into non-overlapping pseudo-trajectories
    pseudo_trajectories = [
        zurich_positions.iloc[i : i + min_rows].reset_index(drop=True)
        for i in range(0, total_rows, min_rows)
        if i + min_rows <= total_rows
    ]

    if len(pseudo_trajectories) < num_flights:
        raise ValueError(
            f"Not enough Zurich pseudo-trajectories with at least {min_rows} rows. Found {len(pseudo_trajectories)}."
        )

    return _concat_flights_into_blocks(
        pseudo_trajectories, num_flights, position_columns
    )


def load_mixed_dataset(
    quadcopter_csv="data/flights.csv",
    zurich_csv="data/zurich_flights_downsampled_2.csv",
    min_rows=800,
    num_flights=3,
    position_columns=None,
) -> pd.DataFrame:
    """
    Load and combine Quadcopter + Zurich datasets into a single DataFrame.
    """
    if position_columns is None:
        position_columns = ["position_x", "position_y", "position_z"]

    quad_df = load_quadcopter_dataset(
        quadcopter_csv, min_rows, num_flights, position_columns
    )
    zurich_df = load_zurich_dataset(zurich_csv, min_rows, num_flights, position_columns)

    return pd.concat([quad_df, zurich_df], ignore_index=True)


def load_simulated_dataset(
    csv_path="data/drone_states.csv",
    min_rows=800,
    num_flights=3,
    features_per_agent=6,
) -> pd.DataFrame:
    """
    Load and process the Simulated Multi-Drone dataset into a ready-to-use DataFrame.

    The dataset format should include columns:
    [flight_id, time_stamp, drone_id, role, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]

    Each flight may contain multiple drones (agents). The loader:
    - Groups by flight_id, then by drone_id
    - Ensures each drone trajectory has at least `min_rows` rows
    - Truncates all drones in a flight to the shortest trajectory length
    - Concatenates multiple drone trajectories side by side (num_flights per flight)
    - Optionally computes acceleration columns if `features_per_agent >= 9`
    """

    # Validate feature count
    if features_per_agent not in [3, 6, 9]:
        raise ValueError("features_per_agent must be one of [3, 6, 9].")

    df = pd.read_csv(csv_path)

    # Sort by flight, drone, and timestamp
    df = df.sort_values(by=["flight_id", "drone_id", "time_stamp"]).reset_index(
        drop=True
    )

    # Define feature columns
    position_columns = ["pos_x", "pos_y", "pos_z"]
    velocity_columns = ["vel_x", "vel_y", "vel_z"]
    accel_columns = ["acc_x", "acc_y", "acc_z"]

    # Select columns based on features_per_agent
    if features_per_agent == 3:
        selected_columns = position_columns
    elif features_per_agent == 6:
        selected_columns = position_columns + velocity_columns
    else:  # 9 features → compute acceleration
        df[accel_columns] = (
            df.groupby(["flight_id", "drone_id"])
            .apply(
                lambda g: g[velocity_columns].diff().div(g["time_stamp"].diff(), axis=0)
            )
            .reset_index(level=[0, 1], drop=True)
        )
        df[accel_columns] = df[accel_columns].fillna(0)
        selected_columns = position_columns + velocity_columns + accel_columns

    grouped_by_flight = df.groupby("flight_id")
    all_flight_dfs = []

    for flight_id, flight_data in grouped_by_flight:
        drones = []
        for _, drone_data in flight_data.groupby("drone_id"):
            if len(drone_data) < min_rows:
                continue
            drones.append(drone_data[selected_columns].reset_index(drop=True))

        if len(drones) < num_flights:
            continue

        # Truncate all drones to shortest length
        min_len = min(d.shape[0] for d in drones)
        truncated = [d.iloc[:min_len] for d in drones]

        # Concatenate drones side-by-side
        concatenated = pd.concat(truncated, axis=1)

        # Rename columns with flight number suffix
        new_columns = []
        for i in range(1, len(truncated) + 1):
            new_columns.extend([f"{col}_flight{i}" for col in selected_columns])
        concatenated.columns = new_columns

        # Add trajectory_index based on flight_id
        concatenated["trajectory_index"] = flight_id

        all_flight_dfs.append(concatenated)

    if not all_flight_dfs:
        raise ValueError(
            f"No valid flights found with at least {num_flights} drones having >= {min_rows} rows."
        )

    return pd.concat(all_flight_dfs, ignore_index=True)


def load_dataset(
    data_type: str,
    min_rows=1000,
    num_flights=3,
    position_columns=None,
    features_per_agent=6,
) -> pd.DataFrame:
    """
    General loader to select dataset type.
    """

    if data_type.lower() == "quadcopter":
        return load_quadcopter_dataset(
            min_rows=min_rows,
            num_flights=num_flights,
            position_columns=position_columns,
        )
    elif data_type.lower() == "zurich":
        return load_zurich_dataset(
            min_rows=min_rows,
            num_flights=num_flights,
            position_columns=position_columns,
        )
    elif data_type.lower() == "mixed":
        return load_mixed_dataset(
            min_rows=min_rows,
            num_flights=num_flights,
            position_columns=position_columns,
        )
    elif data_type.lower() == "simulated":
        return load_simulated_dataset(
            min_rows=min_rows,
            num_flights=num_flights,
            features_per_agent=features_per_agent,
        )
    else:
        raise ValueError(
            f"Unknown DATA_TYPE '{data_type}'. Choose from 'quadcopter', 'zurich', 'mixed'."
        )
