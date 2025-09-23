from pathlib import Path
import json
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

from data.trajectory_loader import load_dataset
from models.attention_bi_gru_predictor import TrajPredictor

# ----------------------------
# Helper: plot full trajectory
# ----------------------------
def plot_full_trajectory(y_true, y_pred, scaler, agents, save_dir, filename="trajectory.png"):
    """
    Plot full multi-agent 3D trajectory (ground-truth vs predicted).

    Args:
        y_true (np.ndarray): Ground-truth scaled values, shape (timesteps, features)
        y_pred (np.ndarray): Predicted scaled values, same shape as y_true
        scaler (MinMaxScaler): Fitted scaler to inverse-transform data
        agents (int): Number of agents
        save_dir (str): Directory to save plot
        filename (str): Name of the PNG file
    """
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[-1]))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))

    dim = 3
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(agents)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for agent in range(agents):
        start = agent * dim
        # True trajectory
        ax.plot(
            y_true_inv[:, start],
            y_true_inv[:, start + 1],
            y_true_inv[:, start + 2],
            label=f"Agent {agent + 1} True",
            color=colors[agent],
        )
        # Predicted trajectory
        ax.plot(
            y_pred_inv[:, start],
            y_pred_inv[:, start + 1],
            y_pred_inv[:, start + 2],
            label=f"Agent {agent + 1} Pred",
            color=colors[agent],
            linestyle="--",
        )

    ax.set_title("Full Trajectory (True vs Predicted)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_label("Z")
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.show()
    plt.close()


# ----------------------------
# Paths & Config
# ----------------------------
experiment_dir = Path("experiments/20250923_094137")
CONFIG_PATH = experiment_dir / "config.json"
MODEL_PATH = experiment_dir / "best_model.pt"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_TYPE = config["DATA_TYPE"]
AGENTS = config["AGENTS"]
LOOK_BACK = config["LOOK_BACK"]
FORWARD_LEN = config["FORWARD_LEN"]  # steps predicted at each iteration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model
# ----------------------------
model = TrajPredictor(**config["model_params"]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Load dataset
# ----------------------------
df = load_dataset(DATA_TYPE, min_rows=100, num_flights=AGENTS)

# Pick a random trajectory
traj_idx = np.random.choice(df["trajectory_index"].unique())
traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)
traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
total_len = traj_data.shape[0]

# ----------------------------
# Scale full trajectory
# ----------------------------
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(traj_data)
scaler_y.fit(traj_data)

traj_scaled = scaler_X.transform(traj_data)

# ----------------------------
# Predict full trajectory iteratively
# ----------------------------
y_pred_scaled = []
input_seq = traj_scaled[:LOOK_BACK].copy()  # (LOOK_BACK, features)

with torch.no_grad():
    for i in range(total_len - LOOK_BACK):
        X_tensor = torch.from_numpy(input_seq[-LOOK_BACK:].reshape(1, LOOK_BACK, -1)).float().to(device)
        pred = model(X_tensor, pred_len=FORWARD_LEN).cpu().numpy()  # (1, num_features * FORWARD_LEN)
        
        # Take only the first step predicted
        first_step = pred[0, :input_seq.shape[1]]  # shape = (num_features,)
        
        input_seq = np.vstack([input_seq, first_step])
        y_pred_scaled.append(first_step)

y_pred_scaled = np.array(y_pred_scaled)  # shape = (total_len - LOOK_BACK, num_features)

# ----------------------------
# Ground truth aligned for plotting
# ----------------------------
y_true_scaled = traj_scaled[LOOK_BACK:]  # exclude initial LOOK_BACK

# ----------------------------
# Plot full trajectory
# ----------------------------
plot_full_trajectory(
    y_true=y_true_scaled,
    y_pred=y_pred_scaled,
    scaler=scaler_y,
    agents=AGENTS,
    save_dir=str(experiment_dir),
    filename=f"full_trajectory_{traj_idx}.png"
)
