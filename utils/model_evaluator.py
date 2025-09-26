"""
model_evaluator.py

Evaluation utilities for trajectory prediction models, including multi-agent scenarios.

This module provides functions to compute common regression metrics for predicted
3D trajectories, supporting both single-agent and multi-agent cases. Metrics include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Euclidean Distance Error (EDE)
- Per-axis metrics (x, y, z), averaged over agents if multi-agent

Features are assumed to be flattened per agent (i.e., [x1, y1, z1, x2, y2, z2, ...])
for multi-agent datasets. The provided scaler is used to inverse-transform predictions
back to original units (e.g., meters) before computing metrics.

Functions:
-----------
- evaluate_metrics_multi_agent(y_true, y_pred, scaler, num_agents)
    Compute regression and Euclidean distance metrics for multi-agent 3D trajectories.

Example:
--------
from sklearn.preprocessing import MinMaxScaler
import torch
from metrics import evaluate_metrics_multi_agent

mse, rmse, mae, ede, axis_mse, axis_rmse, axis_mae = evaluate_metrics_multi_agent(
    y_true_tensor, y_pred_tensor, fitted_scaler, num_agents=3
)
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.scaler import scale_per_agent


def evaluate_metrics_multi_agent(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler, num_agents: int
):
    """
    Compute regression metrics for multi-agent 3D trajectories.

    Args:
        y_true (torch.Tensor): shape (num_sequences, features), features = num_agents*3
        y_pred (torch.Tensor): same shape as y_true
        scaler (MinMaxScaler): fitted scaler to inverse-transform predictions
        num_agents (int): number of agents in trajectory

    Returns:
        Overall MSE, RMSE, MAE, EDE
        Per-axis metrics (averaged over all agents)
    """

    num_features_per_agent = 3  # x, y, z
    total_features = num_agents * num_features_per_agent

    assert y_true.shape[1] == total_features, (
        "Mismatch in number of features and agents"
    )

    # --- Inverse scaling ---
    y_true_np = scale_per_agent(
        y_true.cpu().numpy(), scaler, num_features_per_agent=3, inverse=True
    )  # shape (num_points, num_agents*3)
    y_pred_np = scale_per_agent(
        y_pred.cpu().numpy(), scaler, num_features_per_agent=3, inverse=True
    )

    # --- Reshape to (num_points, num_agents, 3) ---
    y_true_reshaped = y_true_np.reshape(-1, num_agents, num_features_per_agent)
    y_pred_reshaped = y_pred_np.reshape(-1, num_agents, num_features_per_agent)

    # --- Compute per-axis metrics averaged across agents ---
    axis_errors = (
        y_true_reshaped - y_pred_reshaped
    )  # shape: (num_points, num_agents, 3)
    axis_mse = (axis_errors**2).reshape(-1, 3).mean(axis=0)
    axis_rmse = np.sqrt(axis_mse)
    axis_mae = np.abs(axis_errors).reshape(-1, 3).mean(axis=0)

    # --- Overall metrics ---
    mse = ((y_true_reshaped - y_pred_reshaped) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(y_true_reshaped - y_pred_reshaped).mean()

    # --- Euclidean distance error (EDE) per agent, averaged ---
    ede = np.linalg.norm(y_true_reshaped - y_pred_reshaped, axis=2).mean()

    return mse, rmse, mae, ede, axis_mse, axis_rmse, axis_mae


def evaluate_metrics_multi_agent_per_timestep(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler, num_agents: int
):
    """
    Compute regression metrics for multi-agent 3D trajectories per timestep.

    Args:
        y_true (torch.Tensor): shape (num_sequences, pred_len, features), features = num_agents*3
        y_pred (torch.Tensor): same shape as y_true
        scaler (MinMaxScaler): fitted scaler to inverse-transform predictions
        num_agents (int): number of agents in trajectory

    Returns:
        mse_t: array of shape (pred_len,) - MSE per timestep
        rmse_t: array of shape (pred_len,) - RMSE per timestep
        mae_t: array of shape (pred_len,) - MAE per timestep
        ede_t: array of shape (pred_len,) - Euclidean distance error per timestep
        axis_mse_t: array of shape (pred_len, 3) - per-axis MSE per timestep
        axis_rmse_t: array of shape (pred_len, 3) - per-axis RMSE per timestep
        axis_mae_t: array of shape (pred_len, 3) - per-axis MAE per timestep
    """
    num_features_per_agent = 3
    total_features = num_agents * num_features_per_agent
    assert y_true.shape[2] == total_features, (
        "Mismatch in number of features and agents"
    )

    # --- Inverse scaling ---
    y_true_np = scale_per_agent(
        y_true.cpu().numpy(), scaler, num_features_per_agent=3, inverse=True
    )
    y_pred_np = scale_per_agent(
        y_pred.cpu().numpy(), scaler, num_features_per_agent=3, inverse=True
    )

    # --- Reshape to (num_sequences, pred_len, num_agents, 3) ---
    y_true_reshaped = y_true_np.reshape(
        y_true_np.shape[0], y_true_np.shape[1], num_agents, num_features_per_agent
    )
    y_pred_reshaped = y_pred_np.reshape(
        y_pred_np.shape[0], y_pred_np.shape[1], num_agents, num_features_per_agent
    )

    # --- Initialize arrays ---
    pred_len = y_true_reshaped.shape[1]
    mse_t = np.zeros(pred_len)
    rmse_t = np.zeros(pred_len)
    mae_t = np.zeros(pred_len)
    ede_t = np.zeros(pred_len)
    axis_mse_t = np.zeros((pred_len, 3))
    axis_rmse_t = np.zeros((pred_len, 3))
    axis_mae_t = np.zeros((pred_len, 3))

    # --- Compute metrics per timestep ---
    for t in range(pred_len):
        errors_t = (
            y_true_reshaped[:, t] - y_pred_reshaped[:, t]
        )  # shape (num_sequences, num_agents, 3)
        axis_mse_t[t] = (errors_t**2).reshape(-1, 3).mean(axis=0)
        axis_rmse_t[t] = np.sqrt(axis_mse_t[t])
        axis_mae_t[t] = np.abs(errors_t).reshape(-1, 3).mean(axis=0)

        mse_t[t] = (errors_t**2).mean()
        rmse_t[t] = np.sqrt(mse_t[t])
        mae_t[t] = np.abs(errors_t).mean()
        ede_t[t] = np.linalg.norm(errors_t, axis=2).mean()

    return mse_t, rmse_t, mae_t, ede_t, axis_mse_t, axis_rmse_t, axis_mae_t
