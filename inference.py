from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from data.trajectory_loader import load_dataset
from models.modified_attention_bi_gru_predictor import TrajPredictor
from utils.scaler import scale_per_agent

# ----------------------------
# Helper: plot full trajectory
# ----------------------------
def plot_full_trajectory(y_true, y_pred, agents, save_dir, filename="trajectory.png"):
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

    dim = 3
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(agents)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for agent in range(agents):
        start = agent * dim
        # True trajectory
        ax.plot(
            y_true[:, start],
            y_true[:, start + 1],
            y_true[:, start + 2],
            label=f"Agent {agent + 1} True",
            color=colors[agent],
        )
        # Predicted trajectory
        ax.plot(
            y_pred[:, start],
            y_pred[:, start + 1],
            y_pred[:, start + 2],
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
experiment_dir = Path("experiments/20250924_153814")
CONFIG_PATH = experiment_dir / "config.json"
MODEL_PATH = experiment_dir / "best_model.pt"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_TYPE = config["DATA_TYPE"]
# AGENTS = config["AGENTS"]
AGENTS = 3
FEATURES_PER_AGENT = 3  # x,y,z
LOOK_BACK = config["LOOK_BACK"]
FORWARD_LEN = config["FORWARD_LEN"]  # steps predicted at each iteration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model
# ----------------------------
model_params = {
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "num_layers": 1,
}
model = TrajPredictor(**model_params).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Load dataset
# ----------------------------
df = load_dataset(DATA_TYPE, min_rows=200, num_flights=AGENTS)

# Pick a random trajectory
traj_idx = np.random.choice(df["trajectory_index"].unique())
traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)
traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
total_len = traj_data.shape[0]

# ----------------------------
# Scale full trajectory
# ----------------------------
scaler_X_path = experiment_dir / "scaler_X.pkl"
scaler_y_path = experiment_dir / "scaler_y.pkl"
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# ----------------------------
# Scale input sequence
# ----------------------------
traj_scaled_X = scale_per_agent(traj_data, scaler_X, FEATURES_PER_AGENT)   # for feeding into model
traj_scaled_y = scale_per_agent(traj_data, scaler_y, FEATURES_PER_AGENT)   # for prediction comparison

# ----------------------------
# Predict full trajectory iteratively
# ----------------------------
y_pred_scaled = []
input_seq = traj_scaled_X[:LOOK_BACK].copy()  # use X-scaler for model inputs

with torch.no_grad():
    for i in range(total_len - LOOK_BACK - FORWARD_LEN + 1):
        X_tensor = torch.from_numpy(input_seq[-LOOK_BACK:].reshape(1, LOOK_BACK, -1)).float().to(device)
        pred = model(X_tensor).cpu().numpy()
        
        y_pred_scaled.append(pred)

        # Teacher forcing: append ground truth NEXT step (scaled with X for inputs)
        next_step = traj_scaled_X[i + LOOK_BACK]
        input_seq = np.vstack([input_seq, next_step])

y_pred_scaled = np.array(y_pred_scaled).squeeze(1)  # (timesteps, features)

# ----------------------------
# Ground truth aligned (scaled with y-scaler)
# ----------------------------
y_true_scaled = traj_scaled_y[LOOK_BACK + FORWARD_LEN - 1:]

# inverse scale for plotting
y_true = scale_per_agent(y_true_scaled, scaler_y, FEATURES_PER_AGENT, inverse=True)
y_pred = scale_per_agent(y_pred_scaled, scaler_y, FEATURES_PER_AGENT, inverse=True)

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)
print("Expected features:", AGENTS * 3)

# ----------------------------
# Plot full trajectory
# ----------------------------
plot_full_trajectory(
    y_true=y_true,
    y_pred=y_pred,
    agents=AGENTS,
    save_dir=str(experiment_dir),
    filename=f"inference_trajectory_{traj_idx}.png"
)
