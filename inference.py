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
import time
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.plot_generator import plot_inference_trajectory
from utils.model_evaluator import evaluate_metrics_multi_agent as evaluate
from utils.logger import get_inference_logger
from data.trajectory_loader import load_dataset

# Set number of drones (agents) for inference
AGENTS = 3

# Paths & Config
experiment_dir = Path("experiments/20250925_124248")
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
FORWARD_LEN = config["FORWARD_LEN"]

# Log config info
logger.info("Dataset used: %s", DATA_TYPE)
logger.info("Number of drones: %d", AGENTS)

# Dynamically load model class
model_module_name = config[
    "model_module"
]  # e.g., "models.modified_attention_bi_gru_predictor"
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
df = load_dataset(DATA_TYPE, min_rows=200, num_flights=AGENTS)

# Pick a random trajectory for inference
traj_idx = np.random.choice(df["trajectory_index"].unique())
traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)
traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
total_len = traj_data.shape[0]

# Scale input sequence
scaler_X = MinMaxScaler((0, 1))
scaler_y = MinMaxScaler((0, 1))
traj_scaled_X = scaler_X.fit_transform(traj_data)
traj_scaled_y = scaler_y.fit_transform(traj_data)

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
        pred = model(X_tensor).cpu().numpy()  # (1, 1, features)

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
y_true_scaled = traj_scaled_y[LOOK_BACK + FORWARD_LEN - 1 :]

# inverse scale for plotting
y_true = scaler_y.inverse_transform(y_true_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate metrics
mse, rmse, mae, ede, axis_mse, axis_rmse, axis_mae = evaluate(
    torch.tensor(y_true), torch.tensor(y_pred), scaler_y, num_agents=AGENTS
)

mse_x, mse_y, mse_z = axis_mse
rmse_x, rmse_y, rmse_z = axis_rmse
mae_x, mae_y, mae_z = axis_mae

# Log metrics per axis and overall
logger.info(
    "Test Mean Squared Error (MSE) per axis (averaged over %d agents): x=%.6f, y=%.6f, z=%.6f meters^2",
    AGENTS,
    mse_x,
    mse_y,
    mse_z,
)
logger.info("Test Mean Squared Error (MSE) overall: %.6f meters^2", mse)

logger.info(
    "Test Root Mean Squared Error (RMSE) per axis (averaged over %d agents): x=%.6f, y=%.6f, z=%.6f meters",
    AGENTS,
    rmse_x,
    rmse_y,
    rmse_z,
)
logger.info("Test Root Mean Squared Error (RMSE) overall: %.6f meters", rmse)

logger.info(
    "Test Mean Absolute Error (MAE) per axis (averaged over %d agents): x=%.6f, y=%.6f, z=%.6f meters",
    AGENTS,
    mae_x,
    mae_y,
    mae_z,
)
logger.info("Test Mean Absolute Error (MAE) overall: %.6f meters", mae)

logger.info(
    "Test Euclidean Distance Error (EDE) (averaged over all agents): %.6f meters", ede
)

# Plot full trajectory
plot_inference_trajectory(
    y_true=y_true,
    y_pred=y_pred,
    agents=AGENTS,
    save_dir=str(experiment_dir),
    filename=f"inference_trajectory_{traj_idx}.png",
)
