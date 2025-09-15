import os
import time
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data.trajectory_loader import load_and_concat_flights
from models.attention_bi_gru_predictor import TrajPredictor
from utils.logger import get_logger

# --- Parameters ---
LOOK_BACK = 50
FORWARD_LEN = 5
CSV_PATH = "data/flights.csv"
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-3

# --- Setup logger and experiment folder ---
logger, exp_dir = get_logger()
os.makedirs(exp_dir, exist_ok=True)

logger.info("Experiment started")
logger.info("Experiment folder: %s", exp_dir)

# --- Load DataFrame ---
df = load_and_concat_flights(CSV_PATH, min_rows=1000, num_flights=3)

# --- Prepare sequences ---
X, y, trajectory_ids = [], [], []

for traj_idx in df["trajectory_index"].unique():
    traj_df = df[df["trajectory_index"] == traj_idx].reset_index(drop=True)

    # Drop trajectory_index for features
    traj_data = traj_df.drop(columns=["trajectory_index"]).values.astype(np.float32)
    n_rows = traj_data.shape[0]

    seq_count = n_rows - LOOK_BACK - FORWARD_LEN + 1
    for i in range(seq_count):
        seq_X = traj_data[i : i + LOOK_BACK]              # shape (LOOK_BACK, features)
        seq_y = traj_data[i + LOOK_BACK + FORWARD_LEN - 1]  # shape (features,)

        X.append(seq_X)
        y.append(seq_y)
        trajectory_ids.append(traj_idx)

# --- Convert to NumPy arrays ---
X = np.array(X, dtype=np.float32)  # (num_sequences, LOOK_BACK, features)
y = np.array(y, dtype=np.float32)  # (num_sequences, features)
trajectory_ids = np.array(trajectory_ids)

# --- Split train/test ---
X_train, X_test, y_train, y_test, traj_train, traj_test = train_test_split(
    X, y, trajectory_ids, test_size=0.2, shuffle=False
)

num_features_X = X_train.shape[-1]

# --- Scale column by column (concise approach) ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X_train.reshape(-1, num_features_X))
X_train_scaled = scaler_X.transform(X_train.reshape(-1, num_features_X)).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features_X)).reshape(X_test.shape)

scaler_y.fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# --- Convert to tensors ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# --- Create DataLoaders ---
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                        batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                        batch_size=BATCH_SIZE, shuffle=False)

# --- Log dataset sizes ---
total_sequences = X_train_tensor.shape[0] + X_test_tensor.shape[0]
logger.info("Total sequences: %d", total_sequences)
logger.info("Train sequences: %s", X_train_tensor.shape)
logger.info("Test sequences: %s", X_test_tensor.shape)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# --- Model, criterion, optimizer ---
model_params = {
    "input_size": X_train_tensor.shape[-1],   # features (e.g., 3 for x,y,z)
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "output_size": y_train_tensor.shape[-1],  # same as features
    "num_layers": 1,
}

model = TrajPredictor(**model_params).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

logger.info("Model initialized:\n%s", model)

# --- Training ---
patience = 15
best_loss = float("inf")
epochs_no_improve = 0
training_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
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

    if epochs_no_improve >= patience:
        logger.info("Early stopping triggered after %d epochs", epoch + 1)
        break

# --- Save last-epoch model ---
torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pt"))
logger.info("Training completed in %.2f seconds", time.time() - training_start)

# --- Evaluation ---
model.eval()
all_preds, all_trues = [], []
inference_times = []
total_sequences = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        total_sequences += batch_x.size(0)

        start_time = time.time()
        outputs = model(batch_x)
        inference_times.append(time.time() - start_time)

        all_preds.append(outputs.cpu())
        all_trues.append(batch_y.cpu())

# Concatenate all batches
y_pred = torch.cat(all_preds, dim=0)
y_true = torch.cat(all_trues, dim=0)

# Inference time
total_inf_time = sum(inference_times)
logger.info("Average inference time per sequence: %.6f s", total_inf_time / total_sequences)
logger.info("Average inference time per batch: %.6f s", total_inf_time / len(test_loader))

# Save config / hyperparameters
config = {
    "device": str(device),
    "model_params": model_params,
    "LOOK_BACK": LOOK_BACK,
    "FORWARD_LEN": FORWARD_LEN,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
}

config_path = os.path.join(exp_dir, "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

logger.info("Config saved to %s", config_path)

# --- Group results back by trajectory_index ---
traj_test = traj_test[:len(y_true)]  # align just in case

NUM_PLOTS = 3  # number of trajectories to plot
unique_trajs = np.unique(traj_test)

for traj_idx in unique_trajs[:NUM_PLOTS]:
    mask = traj_test == traj_idx

    true_traj = y_true[mask].numpy()
    pred_traj = y_pred[mask].numpy()

    # --- 3D Plot (x, y, z path) ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
        label="True Path", color="blue"
    )
    ax.plot(
        pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
        label="Predicted Path", color="red", linestyle="--"
    )

    ax.set_title(f"Trajectory {traj_idx} (True vs Predicted)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # For 3D axes, set_label is used for Z axis if set_zlabel is unavailable
    ax.set_label("Z")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(exp_dir, f"trajectory_{traj_idx}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info("Saved trajectory plot for %s to %s", traj_idx, plot_path)