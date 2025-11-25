"""
train.py

This script handles the end-to-end training, evaluation, and visualization of a
multi-agent 3D trajectory prediction model using an attention-based BiGRU network.

Main functionalities include:
1. Loading and preprocessing multi-agent trajectory datasets, including sequence
generation and MinMax scaling.
2. Defining and training the TrajPredictor model with early stopping and logging.
3. Evaluating the trained model using multiple metrics such as MSE, RMSE, MAE,
and Euclidean Distance Error (EDE), both overall and per-axis averaged across agents.
4. Logging training progress, evaluation results, and saving model checkpoints.
5. Saving experiment configurations and plotting selected predicted vs. true trajectories
for qualitative inspection.

The script supports GPU acceleration if available and is configurable for different
datasets, numbers of agents, sequence lengths, and training hyperparameters.
"""

import os
import time
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multi_traj_predict.data.trajectory_loader import load_dataset
from multi_traj_predict.models.attention_bi_gru_predictor import TrajPredictor
from multi_traj_predict.utils.logger import get_logger, log_metrics_for_features
from multi_traj_predict.utils.plot_generator import (
    plot_multiagent_trajectories,
    plot_3d_trajectories_subplots,
)
from multi_traj_predict.utils.scaler import scale_per_agent


def main():
    # pylint: disable=all
    # Data settings and parameters
    DATA_TYPE = "simulated"  # Options: "zurich", "quadcopter", "mixed", "simulated"
    SEQUENTIAL_PREDICTION = (
        True  # If False, model predicts only the last point for FORWARD_LEN steps
    )
    EMBEDDING_EXTRACTION = False
    AGENTS = 6  # Number of agents or drones
    LOOK_BACK = 50  # Number of past time steps to use as input

    if EMBEDDING_EXTRACTION:
        FORWARD_LEN = 50  # Prediction is the same as LOOK_BACK for embedding extraction
    else:
        FORWARD_LEN = 5  # Number of future time steps to predict

    if DATA_TYPE == "simulated":
        FEATURES_PER_AGENT = 6  # x, y, z, vx, vy, vz
    else:
        FEATURES_PER_AGENT = 3  # x, y, z

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 500
    LEARNING_RATE = 1e-3

    # Plotting parameters
    NUM_PLOTS = 2  # number of plots to generate
    NUM_SUBPLOTS = 2  # number of subplots to generate

    # Setup logger and experiment folder
    logger, exp_dir = get_logger()
    os.makedirs(exp_dir, exist_ok=True)

    logger.info("Experiment started")
    logger.info("Experiment folder: %s", exp_dir)
    logger.info("Dataset: %s", DATA_TYPE)
    logger.info("Number of drones (agents): %d", AGENTS)
    logger.info("LOOK_BACK (past steps): %d", LOOK_BACK)
    logger.info("FORWARD_LEN (future steps): %d", FORWARD_LEN)
    logger.info(
        "SEQUENTIAL_PREDICTION: %s", "True" if SEQUENTIAL_PREDICTION else "False"
    )

    # Load DataFrame
    df = load_dataset(
        DATA_TYPE,
        min_rows=800,
        num_flights=AGENTS,
        features_per_agent=FEATURES_PER_AGENT,
    )
    print(df.shape)

    # Prepare sequences
    # Track trajectory indices to be used in plotting later
    X, y, trajectory_ids = [], [], []

    for traj_idx in df["trajectory_index"].unique():
        traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)

        # Drop trajectory_index for features
        traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
        n_rows = traj_data.shape[0]

        seq_count = n_rows - LOOK_BACK - FORWARD_LEN + 1
        for i in range(seq_count):
            seq_X = traj_data[i : i + LOOK_BACK]  # shape (LOOK_BACK, features)

            # For sequential prediction, predict FORWARD_LEN steps; else just the last step
            if EMBEDDING_EXTRACTION is False:
                if SEQUENTIAL_PREDICTION:
                    seq_y = traj_data[
                        i + LOOK_BACK : i + LOOK_BACK + FORWARD_LEN
                    ]  # shape (FORWARD_LEN, features)
                else:
                    seq_y = traj_data[
                        i + LOOK_BACK + FORWARD_LEN - 1 : i + LOOK_BACK + FORWARD_LEN
                    ]  # (1, features)
            else:
                seq_y = (
                    seq_X.copy()
                )  # For embedding extraction, target is same as input

            X.append(seq_X)
            y.append(seq_y)
            trajectory_ids.append(traj_idx)

    # Convert to NumPy arrays
    X = np.array(X, dtype=np.float32)  # (num_sequences, LOOK_BACK, features)
    y = np.array(
        y, dtype=np.float32
    )  # (num_sequences, 1, features) or (num_sequences, FORWARD_LEN, features)
    trajectory_ids = np.array(trajectory_ids)

    # Split train/test
    X_train, X_test, y_train, y_test, traj_train, traj_test = train_test_split(
        X, y, trajectory_ids, test_size=0.2, shuffle=False
    )

    # Scale data to [0, 1]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Fit scalers on training data only
    X_train_scaled = scale_per_agent(X_train, scaler_X, FEATURES_PER_AGENT, fit=True)
    X_test_scaled = scale_per_agent(X_test, scaler_X, FEATURES_PER_AGENT, fit=False)

    y_train_scaled = scale_per_agent(y_train, scaler_y, FEATURES_PER_AGENT, fit=True)
    y_test_scaled = scale_per_agent(y_test, scaler_y, FEATURES_PER_AGENT, fit=False)

    # Save scalers
    joblib.dump(scaler_X, os.path.join(exp_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(exp_dir, "scaler_y.pkl"))

    # Convert to tensors
    X_train_tensor = torch.tensor(
        X_train_scaled, dtype=torch.float32
    )  # (num_sequences, LOOK_BACK, features)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Log dataset sizes
    total_sequences = X_train_tensor.shape[0] + X_test_tensor.shape[0]
    logger.info("Total sequences: %d", total_sequences)
    logger.info("Train sequences: %s", X_train_tensor.shape)
    logger.info("Test sequences: %s", X_test_tensor.shape)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Model, criterion, optimizer
    model_params = {
        "input_size": FEATURES_PER_AGENT,
        "enc_hidden_size": 64,
        "dec_hidden_size": 64,
        "num_layers": 1,
    }
    model = TrajPredictor(**model_params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Log model info
    logger.info("Model module: %s", model.__class__.__module__)
    logger.info("Model architecture:\n%s", model)

    # Log training time
    training_start = time.time()

    # Early stopping parameters
    patience = 10
    best_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False

    # Training Loop
    model.train()
    try:
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for batch_x, batch_y in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"
            ):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                # Add prediction_len argument if sequential prediction
                if SEQUENTIAL_PREDICTION:
                    pred = model(batch_x, pred_len=FORWARD_LEN)
                else:
                    pred = model(batch_x, pred_len=1)

                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info("Epoch %d/%d - Train Loss: %.7f", epoch + 1, EPOCHS, avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
            else:
                epochs_no_improve += 1

            # If no improvement for 'patience' epochs, stop training
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                early_stop = True
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user! Running evaluation...")

    # Save last-epoch model
    finally:
        torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pt"))

    # If training completed without early stopping
    if not early_stop:
        logger.info("Training finished without early stopping.")

    # Log total training time
    training_end_time = time.time()
    elapsed_time = training_end_time - training_start
    logger.info("Total training time: %.2f seconds", elapsed_time)

    # Evaluation parameters
    all_preds, all_trues = [], []
    inference_times = []
    test_loss = 0.0
    total_sequences = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            total_sequences += batch_x.size(0)

            # Measure inference time
            start_time = time.time()

            # Add prediction_len argument if sequential prediction
            if SEQUENTIAL_PREDICTION:
                outputs = model(batch_x, pred_len=FORWARD_LEN)
            else:
                outputs = model(batch_x, pred_len=1)

            end_time = time.time()

            # Record inference time per batch
            inference_times.append(end_time - start_time)

            # Compute test loss
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            # Store predictions and true values
            all_preds.append(outputs.cpu())
            all_trues.append(batch_y.cpu())

    # Compute average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Compute total inference time, average inference time per batch and per sequence
    total_inf_time = sum(inference_times)
    avg_inf_time_per_seq = total_inf_time / total_sequences
    avg_inf_time_per_batch = total_inf_time / len(test_loader)

    # Log final test metrics
    logger.info("Test Loss (scaled): %.7f", avg_test_loss)
    logger.info(
        "Average inference time per sequence: %.6f seconds", avg_inf_time_per_seq
    )
    logger.info(
        "Average inference time per batch: %.6f seconds", avg_inf_time_per_batch
    )

    # Concatenate all batches
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_trues, dim=0)

    log_metrics_for_features(
        y_true, y_pred, scaler_y, AGENTS, FEATURES_PER_AGENT, logger
    )

    # Save config / hyperparameters
    config = {
        "device": str(device),
        "model_module": model.__class__.__module__,
        "model_class": model.__class__.__name__,
        "model_params": model_params,
        "DATA_TYPE": DATA_TYPE,
        "AGENTS": AGENTS,
        "FEATURES_PER_AGENT": FEATURES_PER_AGENT,
        "LOOK_BACK": LOOK_BACK,
        "FORWARD_LEN": FORWARD_LEN,
        "SEQUENTIAL_PREDICTION": SEQUENTIAL_PREDICTION,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
    }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    logger.info("Config saved to %s", config_path)

    # Group results back by trajectory_index
    traj_test = traj_test[: len(y_true)]  # align just in case
    unique_trajs = np.unique(traj_test)

    # Randomly select trajectories to plot
    plot_trajs = np.random.choice(
        unique_trajs, size=min(NUM_PLOTS, len(unique_trajs)), replace=False
    ).tolist()

    # Plot trajectories using the helper function
    plot_multiagent_trajectories(
        y_true=y_true.numpy(),
        y_pred=y_pred.numpy(),
        traj_ids=traj_test,
        plot_trajs=plot_trajs,
        scaler=scaler_y,
        features_per_agent=FEATURES_PER_AGENT,
        save_dir=exp_dir,
        velocity_scale=0.5,
        acceleration_scale=0.3,
    )

    if not EMBEDDING_EXTRACTION:
        # Stack all past inputs (for context)
        past_inputs = torch.cat(
            [b for b, _ in test_loader], dim=0
        ).numpy()  # (num_sequences, LOOK_BACK, num_features)

        # Ensure NUM_PLOTS does not exceed number of sequences
        num_sequences = y_true.shape[0]
        NUM_SUBPLOTS = min(NUM_SUBPLOTS, num_sequences)

        # Randomly select sequence indices for plotting
        plot_indices = np.random.choice(num_sequences, size=NUM_SUBPLOTS, replace=False)

        trajectory_sets = []

        for seq_idx in plot_indices:
            past = past_inputs[seq_idx]  # shape: (LOOK_BACK, features)
            true_future = y_true[seq_idx]  # shape: (seq_len, features)
            pred_future = y_pred[seq_idx]  # shape: (seq_len, features)

            # Inverse scale past and future
            past_orig = scale_per_agent(
                past, scaler_X, FEATURES_PER_AGENT, inverse=True
            )
            true_future_orig = scale_per_agent(
                true_future, scaler_y, FEATURES_PER_AGENT, inverse=True
            )
            pred_future_orig = scale_per_agent(
                pred_future, scaler_y, FEATURES_PER_AGENT, inverse=True
            )

            # Concatenate last past point with future to make continuous lines
            true_line = np.vstack([past_orig[-1:], true_future_orig])
            pred_line = np.vstack([past_orig[-1:], pred_future_orig])

            trajectory_sets.append((past_orig, true_line, pred_line))

        # Create subplots for selected sequences
        plot_path = Path(exp_dir) / "training_subplots.png"
        plot_3d_trajectories_subplots(
            trajectory_sets,
            per_agent=True,
            num_features=FEATURES_PER_AGENT,  # Enable velocity + acceleration
            velocity_scale=0.5,  # Adjust arrow length
            acceleration_scale=0.3,  # Adjust arrow length
            save_path=str(plot_path),
        )


if __name__ == "__main__":
    main()
