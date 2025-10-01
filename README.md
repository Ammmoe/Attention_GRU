# Multi-Agent Trajectory Prediction (`train.py` & `inference.py`)

This repository contains **training and inference scripts** for **multi-agent 3D trajectory prediction** using PyTorch. The system predicts drone (or agent) trajectories by taking past coordinate sequences and forecasting their future positions in 3D space.

> During inference, this model supports predictions for a variable number of agents (drones), making it adaptable to different multi-agent scenarios.

---

## 📂 Directory Structure

```bash
.
├── train.py              # Training script
├── inference.py          # Inference script
├── models/               # Model definitions (GRU, LSTM, RNN variants)
├── data/                 # Datasets and dataset loaders
├── utils/                # Utility/helper functions
├── experiments/          # Saved runs (models, configs, logs, plots, scalers)
│   ├── 20251001_094916/  # Example experiment folder
│   │   ├── last_model.pt
│   │   ├── best_model.pt
│   │   ├── scaler_X.pkl
│   │   ├── scaler_y.pkl
│   │   ├── config.json
│   │   ├── training.log
│   │   ├── inference.log
│   │   └── plots/
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## 📦 Requirements

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

## 🔁 Training (`train.py`)

### Training Workflow

1. Load or generate multi-agent 3D trajectory data
2. Convert trajectories into past (`LOOK_BACK`) and future (`FORWARD_LEN`) frames
3. Split into training and testing sets
4. Normalize features and convert to PyTorch tensors
5. Train a sequence-to-sequence model (MSE loss, optional early stopping)
6. Save trained weights (`last_model.pt`, `best_model.pt`) and scalers (`scaler_X.pkl`, `scaler_y.pkl`)
7. Evaluate performance (MSE, RMSE, MAE, EDE) including **per-timestep metrics**
8. Generate 3D trajectory plots

### Model Selection

Choose a model by changing the import at the top of `train.py`:

```python
# Example: bidirectional GRU with attention
from models.attention_bi_gru_predictor import TrajPredictor
```

Available models:

1. `attention_bi_gru_predictor` — Bi-GRU with attention
2. `attention_bi_lstm_predictor` — Bi-LSTM with attention
3. `attention_gru_predictor` — GRU with attention
4. `attention_lstm_predictor` — LSTM with attention
5. `gru_predictor` — plain GRU
6. `lstm_predictor` — plain LSTM
7. `rnn_predictor` — plain RNN

### Model Parameters

**Bidirectional models:**

```python
model_params = {
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "num_layers": 1,
}
```

**Uni-directional models:**

```python
model_params = {
    "hidden_size": 64,
    "num_layers": 1,
}
```

### Dataset Configuration

```python
DATA_TYPE = "mixed"   # "zurich", "quadcopter", or "mixed"
AGENTS = 3            # number of drones/agents
```

* **zurich** — cleaned MAV dataset, 10 Hz
* **quadcopter** — real quadcopter delivery dataset
* **mixed** — combination of both

### Data & Training Parameters

```python
SEQUENTIAL_PREDICTION = True  # If False, predict only the last point
LOOK_BACK = 50                # number of past frames as input
FORWARD_LEN = 5               # number of future frames to predict

# Training
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-3
```

### Plotting Parameters

```python
NUM_PLOTS = 2     # number of full test trajectories to plot
NUM_SUBPLOTS = 2  # number of detailed sequence plots per trajectory
```

* **NUM_PLOTS**: number of full prediction vs ground truth plots.
* **NUM_SUBPLOTS**: number of detailed sequence comparisons (each with its own `LOOK_BACK` and `FORWARD_LEN`).

### Training Example

```bash
python train.py
```

Example configuration inside `train.py`:

```python
from models.attention_bi_gru_predictor import TrajPredictor

DATA_TYPE = "mixed"
AGENTS = 3
SEQUENTIAL_PREDICTION = True

LOOK_BACK  = 50
FORWARD_LEN = 5
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-3
NUM_PLOTS = 2
NUM_SUBPLOTS = 2

model_params = {
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "num_layers": 1,
}
```

---

## ▶️ Inference (`inference.py`)

### Inference Workflow

1. Load saved model and scalers from an experiment directory
2. Perform trajectory prediction for one or more agents
3. Choose between **sequential prediction** (step-by-step) or **last-point-only prediction**
4. Visualize results with detailed subplots

### Inference Example

```bash
python inference.py
```

Key parameters inside `inference.py`:

```python
# Number of agents for inference
AGENTS = 1

# Sequential or last-point prediction
SEQUENTIAL_PREDICTION = True

# Number of detailed subplots
NUM_SUBPLOTS = 1

# Path to trained experiment folder
experiment_dir = Path("experiments/20251001_094916")
MODEL_PATH = experiment_dir / "last_model.pt"
```

> ⚠️ **Important:** Set the correct `experiment_dir` path to load model weights and scalers.
> You can choose `last_model.pt` or `best_model.pt` depending on your needs.

---

## 📚 Dataset Credits

* **Zurich MAV Dataset** – [https://rpg.ifi.uzh.ch/zurichmavdataset.html](https://rpg.ifi.uzh.ch/zurichmavdataset.html)
* **Quadcopter Delivery Dataset (CMU)** – [https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453](https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453)

---

Happy experimenting — train on multiple agents, predict trajectories, and visualize them with detailed sequential plots 🚀
