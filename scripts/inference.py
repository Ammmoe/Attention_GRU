"""
inference.py

Run multi-agent trajectory inference using a trained model.

This script performs the following steps:

1. Loads configuration parameters from a JSON file.
2. Dynamically loads the trained model class and weights.
3. Loads a dataset and selects a trajectory for inference.
4. Scales input features using MinMaxScaler (fitted on the trajectory itself).
5. Iteratively predicts the full trajectory using the model (supports teacher forcing).
6. Inverse-scales predicted and ground-truth trajectories to original units.
7. Plots the full trajectory of all agents (ground-truth vs predicted).

Notes:
- Supports variable number of agents at inference time.
- Currently uses runtime scaling based on the selected trajectory instead of reusing training scalers.
- Teacher forcing is applied: the next input step is the ground-truth scaled step.
- Uses GPU if available; otherwise falls back to CPU.

Outputs:
- PNG visualization of predicted vs true trajectories per agent.
- Prints shape of predicted and ground-truth trajectories.
"""

from pathlib import Path
from importlib import import_module
import json
import argparse
import time
import torch
import joblib
import numpy as np
from multi_traj_predict.utils.plot_generator import (
    plot_inference_trajectory,
    plot_3d_trajectories_subplots,
)
from multi_traj_predict.utils.scaler import scale_per_agent
from multi_traj_predict.utils.logger import (
    get_inference_logger,
    log_metrics_for_features,
    get_latest_experiment_dir,
)
from multi_traj_predict.data.trajectory_loader import load_dataset


def main(exp_dir: str | None = None):
    # pylint: disable=all
    # Set number of drones (agents) for inference
    AGENTS = 6

    # Set number of subplots for sequential plotter
    NUM_SUBPLOTS = 1

    # Predict in sequence or in single coordinate points
    SEQUENTIAL_PREDICTION = True

    # Paths & Config
    # Auto-detect latest experiment if not provided
    experiment_dir = Path(exp_dir) if exp_dir else get_latest_experiment_dir()
    print("Using experiment:", experiment_dir)

    CONFIG_PATH = experiment_dir / "config.json"
    MODEL_PATH = experiment_dir / "last_model.pt"

    # Set up logger
    logger = get_inference_logger(exp_dir=str(experiment_dir))

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Config params
    DATA_TYPE = config["DATA_TYPE"]
    LOOK_BACK = config["LOOK_BACK"]
    # FORWARD_LEN = config["FORWARD_LEN"]
    FORWARD_LEN = 5
    FEATURES_PER_AGENT = config["FEATURES_PER_AGENT"]

    # Log config info
    logger.info("Dataset used: %s", DATA_TYPE)
    logger.info("Number of drones: %d", AGENTS)

    # Dynamically load model class
    model_module_name = config[
        "model_module"
    ]  # e.g., "multi_traj_predict.models.modified_attention_bi_gru_predictor"
    # Prefix with root package
    model_class_name = config["model_class"]  # e.g., "TrajPredictor"
    model_params = config["model_params"]  # dict of parameters

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    # Import module
    model_module = import_module(model_module_name)

    # Get class from module
    ModelClass = getattr(model_module, model_class_name)

    # Instantiate model and load weights
    model = ModelClass(**model_params).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load dataset
    df = load_dataset(DATA_TYPE, min_rows=800, num_flights=AGENTS)
    
    # Save the Full DataFrame to CSV inside the experiment directory
    df.to_csv(experiment_dir / "full_dataframe.csv", index=False)

    # Take only the last 20% which is unused during training
    split_idx = int(len(df) * 0.8)
    df = df.iloc[split_idx:].reset_index(drop=True)
    
    # Save the Inference DataFrame to CSV inside the experiment directory
    df.to_csv(experiment_dir / "inference_dataframe.csv", index=False)

    # Pick a random trajectory for inference
    traj_idx = np.random.choice(df["trajectory_index"].unique())
    traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)
    traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
    total_len = traj_data.shape[0]

    # Scale input sequence
    scaler_X_path = experiment_dir / "scaler_X.pkl"
    scaler_y_path = experiment_dir / "scaler_y.pkl"
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    traj_scaled_X = scale_per_agent(traj_data, scaler_X, FEATURES_PER_AGENT)
    traj_scaled_y = scale_per_agent(traj_data, scaler_y, FEATURES_PER_AGENT)

    # Predict full trajectory iteratively
    y_pred_scaled = []
    input_seq = traj_scaled_X[:LOOK_BACK].copy()  # use X-scaler for model inputs
    start_time = time.time()

    with torch.no_grad():
        for i in range(total_len - LOOK_BACK - FORWARD_LEN + 1):
            # Prepare input tensor
            X_tensor = (
                torch.from_numpy(input_seq[-LOOK_BACK:].reshape(1, LOOK_BACK, -1))
                .float()
                .to(device)
            )

            # Model prediction
            if SEQUENTIAL_PREDICTION:
                pred = (
                    model(X_tensor, pred_len=FORWARD_LEN).cpu().numpy()
                )  # (1, FORWARD_LEN, features)
            else:
                pred = model(X_tensor, pred_len=1).cpu().numpy()  # (1, 1, features)

            # Append prediction
            y_pred_scaled.append(pred)

            # Teacher forcing: append ground truth NEXT step (scaled with X for inputs)
            next_step = traj_scaled_X[i + LOOK_BACK]
            input_seq = np.vstack([input_seq, next_step])

    # Log inference time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info("Total inference time: %.4f seconds", total_time)
    logger.info(
        "Average time per step: %.6f seconds",
        total_time / (total_len - LOOK_BACK - FORWARD_LEN + 1),
    )

    # Convert predictions to array
    y_pred_scaled = np.array(y_pred_scaled).squeeze(1)  # (timesteps, features)

    # Ground truth aligned (scaled with y-scaler)
    # y_true_scaled = traj_scaled_y[LOOK_BACK + FORWARD_LEN - 1 :]
    y_true_scaled = []
    seq_count = total_len - LOOK_BACK - FORWARD_LEN + 1
    for i in range(seq_count):
        if SEQUENTIAL_PREDICTION:
            seq_y = traj_scaled_y[
                i + LOOK_BACK : i + LOOK_BACK + FORWARD_LEN
            ]  # shape (FORWARD_LEN, features)
        else:
            seq_y = traj_scaled_y[
                i + LOOK_BACK + FORWARD_LEN - 1 : i + LOOK_BACK + FORWARD_LEN
            ]  # (1, features)

        y_true_scaled.append(seq_y)

    y_true_scaled = np.array(
        y_true_scaled
    )  # shape: (num_sequences, FORWARD_LEN, features)

    # Convert to tensor because log metrics function accepts tensors
    y_true_tensor = torch.tensor(y_true_scaled, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred_scaled, dtype=torch.float32)

    log_metrics_for_features(
        y_true_tensor, y_pred_tensor, scaler_y, AGENTS, FEATURES_PER_AGENT, logger
    )

    if FEATURES_PER_AGENT == 3:
        # inverse scale for plotting
        y_true = scale_per_agent(
            y_true_scaled, scaler_y, FEATURES_PER_AGENT, inverse=True
        )
        y_pred = scale_per_agent(
            y_pred_scaled, scaler_y, FEATURES_PER_AGENT, inverse=True
        )

        # Plot full trajectory
        plot_inference_trajectory(
            y_true=y_true,
            y_pred=y_pred,
            agents=AGENTS,
            save_dir=str(experiment_dir),
            filename=f"inference_trajectory_{traj_idx}.png",
        )

        # Number of sequences to visualize (e.g., 4 randomly chosen timesteps)
        NUM_SUBPLOTS = min(NUM_SUBPLOTS, y_true.shape[0])

        # Select random indices for plotting
        plot_indices = np.random.choice(
            np.arange(LOOK_BACK, y_true.shape[0]),  # indices starting from 50
            NUM_SUBPLOTS,
            replace=False,
        )

        trajectory_sets = []

        for idx in plot_indices:
            # For inference, "past" is the LOOK_BACK window ending at this sequence
            start_idx = max(0, idx - LOOK_BACK)

            past = traj_scaled_X[start_idx:idx]  # scaled input
            past_orig = scale_per_agent(
                past, scaler_X, FEATURES_PER_AGENT, inverse=True
            )

            true_future = y_true[idx]
            pred_future = y_pred[idx]

            # Make continuous lines
            true_line = np.vstack([past_orig[-1:], true_future])
            pred_line = np.vstack([past_orig[-1:], pred_future])

            trajectory_sets.append((past_orig, true_line, pred_line))

        # Generate short timestamp for filenames
        timestamp = time.strftime("%H%M%S")

        # Define save path
        plot_path = (
            Path(experiment_dir) / f"inference_subplots_{traj_idx}_{timestamp}.png"
        )

        # Use the same plotting function as in train.py
        plot_3d_trajectories_subplots(
            trajectory_sets,
            per_agent=False,
            title=f"Multi-Drone Inference Trajectory {traj_idx}",
            save_path=str(plot_path),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Path to experiment directory. If omitted, loads latest experiment."
    )
    args = parser.parse_args()

    # Pass CLI argument to main
    main(exp_dir=args.exp_dir)
