# Multi-Agent Trajectory Prediction Training (`train.py`)

This repository contains a **sequence-to-coordinate trajectory prediction training script** written in PyTorch. It predicts **3D trajectories for multiple drones (or agents)** by taking sequences of past coordinates for each agent and predicting their future positions simultaneously.

---

## 🔁 Workflow (what `train.py` does)

1. Load or generate multi-agent 3D trajectory data
2. Convert trajectories into sequences of past frames (`LOOK_BACK`) and future frames (`FORWARD_LEN`) for all agents
3. Split into training and testing sets
4. Normalize features and convert to PyTorch tensors
5. Train a sequence-to-sequence model (MSE loss, optional early stopping)
6. Evaluate performance (MSE, RMSE, MAE, EDE) for all agents
7. Visualize predictions vs ground truth in 3D plots
8. Save model, config, and plots

---

## 📦 Requirements (install first)

```bash
pip install -r requirements.txt
```

Minimum recommended environment:

* Python 3.8+
* PyTorch
* NumPy
* scikit-learn
* matplotlib

---

## ⚙️ Quick start — choose a model

Change the model import at the top of `train.py` to select the architecture you want to test:

```python
# Example: bi-directional GRU with attention
from models.attention_bi_gru_predictor import TrajPredictor
```

Available model modules:

1. `attention_bi_gru_predictor` — Bidirectional GRU with attention
2. `attention_bi_lstm_predictor` — Bidirectional LSTM with attention
3. `attention_gru_predictor` — Uni-directional GRU with attention
4. `attention_lstm_predictor` — Uni-directional LSTM with attention
5. `gru_predictor` — Plain GRU
6. `lstm_predictor` — Plain LSTM
7. `rnn_predictor` — Plain RNN

> ⚠️ **Important:** Bidirectional architectures require separate encoder/decoder hidden sizes (`enc_hidden_size`, `dec_hidden_size`). Uni-directional architectures typically use a single `hidden_size`.

---

## 🛠 Model parameters (set after choosing model)

**For bidirectional models:**

```python
model_params = {
    "input_size": X_train_tensor.shape[-1],  # features (e.g., 3 for x,y,z per agent)
    "enc_hidden_size": 64, # encoder hidden size
    "dec_hidden_size": 64, # decoder hidden size
    "output_size": y_train_tensor.shape[-1], # same as input features
    "num_layers": 1,
}
```

**For uni-directional models:**

```python
model_params = {
    "input_size": X_train_tensor.shape[-1],  # features (e.g., 3 for x,y,z per agent)
    "hidden_size": 64, # hidden size for encoder/decoder
    "output_size": y_train_tensor.shape[-1], # same as input features
    "num_layers": 1,
}
```

---

## 📁 Dataset configuration

Set the dataset type in `train.py`:

```python
DATA_TYPE = "mixed"   # Options: "zurich", "quadcopter", "mixed"
AGENTS = 3            # Number of agents/drones
```

* `zurich` — cleaned MAV dataset, 10 Hz
* `quadcopter` — real quadcopter trajectories
* `mixed` — combination of Zurich + quadcopter datasets for multi-agent scenarios

`AGENTS` allows you to specify the number of drones in the simulation or dataset.

---

## ⏱️ Data & training parameters

```python
# Data parameters
LOOK_BACK  = 50  # number of past frames used as input
FORWARD_LEN = 5  # number of future frames to predict

# Training parameters
BATCH_SIZE = 70
EPOCHS = 500
LEARNING_RATE = 1e-3
```

Adjust `LOOK_BACK` and `FORWARD_LEN` to experiment with short-term vs long-term forecasting.

---

## 📊 Plotting & output

```python
NUM_PLOTS = 3  # number of plots generated after training
```

After training, the script will:

* Save the trained model files (`best_model.pt` and `last_model.pt`)
* Save the training configuration (`config.json`)
* Save 3D plots of predicted vs ground-truth trajectories for selected examples
* Save detailed training logs, including evaluation metrics (MSE, RMSE, MAE, EDE)

---

## ✅ Quick usage example

Set dataset, number of agents, and model in `train.py`:

```python
from models.attention_bi_gru_predictor import TrajPredictor

DATA_TYPE = "mixed"
AGENTS = 3

LOOK_BACK  = 50
FORWARD_LEN = 5
BATCH_SIZE = 70
EPOCHS = 500
LEARNING_RATE = 1e-3
NUM_PLOTS = 4

model_params = {
    "input_size": X_train_tensor.shape[-1],
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "output_size": y_train_tensor.shape[-1],
    "num_layers": 1,
}
```

---

## ▶️ Run the training script

```bash
python train.py
# or
python3 train.py
```

---

## 📚 Dataset Credits

* **Zurich MAV Dataset** – [https://rpg.ifi.uzh.ch/zurichmavdataset.html](https://rpg.ifi.uzh.ch/zurichmavdataset.html)
* **Quadcopter Delivery Dataset (CMU)** – [https://kilthub.cmu.edu/articles/dataset/Data\_Collected\_with\_Package\_Delivery\_Quadcopter\_Drone/12683453](https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453)

---

Happy experimenting — try different numbers of drones, architectures, and hyperparameters to optimize multi-agent trajectory prediction!
