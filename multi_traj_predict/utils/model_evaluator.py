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
from multi_traj_predict.utils.scaler import scale_per_agent


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
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    scaler: MinMaxScaler,
    num_agents: int,
    num_features_per_agent: int,
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


def evaluate_metrics_multi_agent_pos_vel(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler, num_agents: int
):
    """
    Compute regression metrics for multi-agent trajectories with positions + velocities.

    Args:
        y_true (torch.Tensor): shape (num_sequences, features), features = num_agents*6
        y_pred (torch.Tensor): same shape as y_true
        scaler (MinMaxScaler): fitted scaler to inverse-transform predictions
        num_agents (int): number of agents

    Returns:
        pos_mse, pos_rmse, pos_mae, ede        # position metrics
        vel_mse, vel_rmse, vel_mae            # velocity metrics
        axis_pos_mse, axis_pos_rmse, axis_pos_mae
        axis_vel_mse, axis_vel_rmse, axis_vel_mae
    """

    num_features_per_agent = 6  # x,y,z,vx,vy,vz
    total_features = num_agents * num_features_per_agent
    assert y_true.shape[1] == total_features, "Mismatch in features and agents"

    # --- Inverse scaling ---
    y_true_np = scale_per_agent(
        y_true.cpu().numpy(), scaler, num_features_per_agent=6, inverse=True
    )
    y_pred_np = scale_per_agent(
        y_pred.cpu().numpy(), scaler, num_features_per_agent=6, inverse=True
    )

    # --- Reshape to (num_points, num_agents, 6) ---
    y_true_reshaped = y_true_np.reshape(-1, num_agents, num_features_per_agent)
    y_pred_reshaped = y_pred_np.reshape(-1, num_agents, num_features_per_agent)

    # --- Split positions and velocities ---
    y_true_pos = y_true_reshaped[:, :, :3]
    y_pred_pos = y_pred_reshaped[:, :, :3]
    y_true_vel = y_true_reshaped[:, :, 3:]
    y_pred_vel = y_pred_reshaped[:, :, 3:]

    # --- Position metrics ---
    pos_errors = y_true_pos - y_pred_pos
    axis_pos_mse = (pos_errors**2).reshape(-1, 3).mean(axis=0)
    axis_pos_rmse = np.sqrt(axis_pos_mse)
    axis_pos_mae = np.abs(pos_errors).reshape(-1, 3).mean(axis=0)

    pos_mse = (pos_errors**2).mean()
    pos_rmse = np.sqrt(pos_mse)
    pos_mae = np.abs(pos_errors).mean()
    ede = np.linalg.norm(
        pos_errors, axis=2
    ).mean()  # Euclidean distance only for positions

    # --- Velocity metrics ---
    vel_errors = y_true_vel - y_pred_vel
    axis_vel_mse = (vel_errors**2).reshape(-1, 3).mean(axis=0)
    axis_vel_rmse = np.sqrt(axis_vel_mse)
    axis_vel_mae = np.abs(vel_errors).reshape(-1, 3).mean(axis=0)

    vel_mse = (vel_errors**2).mean()
    vel_rmse = np.sqrt(vel_mse)
    vel_mae = np.abs(vel_errors).mean()

    return (
        pos_mse,
        pos_rmse,
        pos_mae,
        ede,
        vel_mse,
        vel_rmse,
        vel_mae,
        axis_pos_mse,
        axis_pos_rmse,
        axis_pos_mae,
        axis_vel_mse,
        axis_vel_rmse,
        axis_vel_mae,
    )


def evaluate_metrics_multi_agent_pos_vel_per_timestep(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler, num_agents: int
):
    """
    Compute per-timestep regression metrics for multi-agent trajectories with positions + velocities.

    Args:
        y_true (torch.Tensor): shape (num_sequences, pred_len, features), features = num_agents*6
        y_pred (torch.Tensor): same shape as y_true
        scaler (MinMaxScaler): fitted scaler to inverse-transform predictions
        num_agents (int): number of agents

    Returns:
        pos_mse_t, pos_rmse_t, pos_mae_t, ede_t        # positions
        vel_mse_t, vel_rmse_t, vel_mae_t              # velocities
        axis_pos_mse_t, axis_pos_rmse_t, axis_pos_mae_t
        axis_vel_mse_t, axis_vel_rmse_t, axis_vel_mae_t
    """
    num_features_per_agent = 6
    total_features = num_agents * num_features_per_agent
    assert y_true.shape[2] == total_features, (
        "Mismatch in number of features and agents"
    )

    # --- Inverse scaling ---
    y_true_np = scale_per_agent(
        y_true.cpu().numpy(), scaler, num_features_per_agent=6, inverse=True
    )
    y_pred_np = scale_per_agent(
        y_pred.cpu().numpy(), scaler, num_features_per_agent=6, inverse=True
    )

    # --- Reshape to (num_sequences, pred_len, num_agents, 6) ---
    y_true_reshaped = y_true_np.reshape(
        y_true_np.shape[0], y_true_np.shape[1], num_agents, num_features_per_agent
    )
    y_pred_reshaped = y_pred_np.reshape(
        y_pred_np.shape[0], y_pred_np.shape[1], num_agents, num_features_per_agent
    )

    # --- Split positions and velocities ---
    y_true_pos = y_true_reshaped[:, :, :, :3]
    y_pred_pos = y_pred_reshaped[:, :, :, :3]
    y_true_vel = y_true_reshaped[:, :, :, 3:]
    y_pred_vel = y_pred_reshaped[:, :, :, 3:]

    pred_len = y_true_reshaped.shape[1]

    # --- Initialize arrays ---
    pos_mse_t = np.zeros(pred_len)
    pos_rmse_t = np.zeros(pred_len)
    pos_mae_t = np.zeros(pred_len)
    ede_t = np.zeros(pred_len)
    axis_pos_mse_t = np.zeros((pred_len, 3))
    axis_pos_rmse_t = np.zeros((pred_len, 3))
    axis_pos_mae_t = np.zeros((pred_len, 3))

    vel_mse_t = np.zeros(pred_len)
    vel_rmse_t = np.zeros(pred_len)
    vel_mae_t = np.zeros(pred_len)
    axis_vel_mse_t = np.zeros((pred_len, 3))
    axis_vel_rmse_t = np.zeros((pred_len, 3))
    axis_vel_mae_t = np.zeros((pred_len, 3))

    # --- Compute metrics per timestep ---
    for t in range(pred_len):
        # --- Positions ---
        pos_errors = (
            y_true_pos[:, t] - y_pred_pos[:, t]
        )  # shape (num_sequences, num_agents, 3)
        axis_pos_mse_t[t] = (pos_errors**2).reshape(-1, 3).mean(axis=0)
        axis_pos_rmse_t[t] = np.sqrt(axis_pos_mse_t[t])
        axis_pos_mae_t[t] = np.abs(pos_errors).reshape(-1, 3).mean(axis=0)

        pos_mse_t[t] = (pos_errors**2).mean()
        pos_rmse_t[t] = np.sqrt(pos_mse_t[t])
        pos_mae_t[t] = np.abs(pos_errors).mean()
        ede_t[t] = np.linalg.norm(pos_errors, axis=2).mean()  # Euclidean distance

        # --- Velocities ---
        vel_errors = (
            y_true_vel[:, t] - y_pred_vel[:, t]
        )  # shape (num_sequences, num_agents, 3)
        axis_vel_mse_t[t] = (vel_errors**2).reshape(-1, 3).mean(axis=0)
        axis_vel_rmse_t[t] = np.sqrt(axis_vel_mse_t[t])
        axis_vel_mae_t[t] = np.abs(vel_errors).reshape(-1, 3).mean(axis=0)

        vel_mse_t[t] = (vel_errors**2).mean()
        vel_rmse_t[t] = np.sqrt(vel_mse_t[t])
        vel_mae_t[t] = np.abs(vel_errors).mean()

    return (
        pos_mse_t,
        pos_rmse_t,
        pos_mae_t,
        ede_t,
        vel_mse_t,
        vel_rmse_t,
        vel_mae_t,
        axis_pos_mse_t,
        axis_pos_rmse_t,
        axis_pos_mae_t,
        axis_vel_mse_t,
        axis_vel_rmse_t,
        axis_vel_mae_t,
    )


def evaluate_metrics_multi_agent_pos_vel_acc_per_timestep(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler, num_agents: int
):
    """
    Compute per-timestep regression metrics for multi-agent trajectories with positions + velocities + accelerations.

    Args:
        y_true (torch.Tensor): shape (num_sequences, pred_len, features), features = num_agents*9
        y_pred (torch.Tensor): same shape as y_true
        scaler (MinMaxScaler): fitted scaler to inverse-transform predictions
        num_agents (int): number of agents

    Returns:
        pos_mse_t, pos_rmse_t, pos_mae_t, ede_t,
        vel_mse_t, vel_rmse_t, vel_mae_t,
        acc_mse_t, acc_rmse_t, acc_mae_t,
        axis_pos_mse_t, axis_pos_rmse_t, axis_pos_mae_t,
        axis_vel_mse_t, axis_vel_rmse_t, axis_vel_mae_t,
        axis_acc_mse_t, axis_acc_rmse_t, axis_acc_mae_t
    """
    num_features_per_agent = 9
    total_features = num_agents * num_features_per_agent
    assert y_true.shape[2] == total_features, (
        "Mismatch in number of features and agents"
    )

    # --- Inverse scaling ---
    y_true_np = scale_per_agent(
        y_true.cpu().numpy(),
        scaler,
        num_features_per_agent=num_features_per_agent,
        inverse=True,
    )
    y_pred_np = scale_per_agent(
        y_pred.cpu().numpy(),
        scaler,
        num_features_per_agent=num_features_per_agent,
        inverse=True,
    )

    # --- Reshape to (num_sequences, pred_len, num_agents, 9) ---
    y_true_reshaped = y_true_np.reshape(
        y_true_np.shape[0], y_true_np.shape[1], num_agents, num_features_per_agent
    )
    y_pred_reshaped = y_pred_np.reshape(
        y_pred_np.shape[0], y_pred_np.shape[1], num_agents, num_features_per_agent
    )

    # --- Split positions, velocities, accelerations ---
    y_true_pos = y_true_reshaped[:, :, :, 0:3]
    y_pred_pos = y_pred_reshaped[:, :, :, 0:3]

    y_true_vel = y_true_reshaped[:, :, :, 3:6]
    y_pred_vel = y_pred_reshaped[:, :, :, 3:6]

    y_true_acc = y_true_reshaped[:, :, :, 6:9]
    y_pred_acc = y_pred_reshaped[:, :, :, 6:9]

    pred_len = y_true_reshaped.shape[1]

    # --- Initialize arrays ---
    pos_mse_t = np.zeros(pred_len)
    pos_rmse_t = np.zeros(pred_len)
    pos_mae_t = np.zeros(pred_len)
    ede_t = np.zeros(pred_len)
    axis_pos_mse_t = np.zeros((pred_len, 3))
    axis_pos_rmse_t = np.zeros((pred_len, 3))
    axis_pos_mae_t = np.zeros((pred_len, 3))

    vel_mse_t = np.zeros(pred_len)
    vel_rmse_t = np.zeros(pred_len)
    vel_mae_t = np.zeros(pred_len)
    axis_vel_mse_t = np.zeros((pred_len, 3))
    axis_vel_rmse_t = np.zeros((pred_len, 3))
    axis_vel_mae_t = np.zeros((pred_len, 3))

    acc_mse_t = np.zeros(pred_len)
    acc_rmse_t = np.zeros(pred_len)
    acc_mae_t = np.zeros(pred_len)
    axis_acc_mse_t = np.zeros((pred_len, 3))
    axis_acc_rmse_t = np.zeros((pred_len, 3))
    axis_acc_mae_t = np.zeros((pred_len, 3))

    # --- Compute metrics per timestep ---
    for t in range(pred_len):
        # Positions
        pos_errors = y_true_pos[:, t] - y_pred_pos[:, t]
        axis_pos_mse_t[t] = (pos_errors**2).reshape(-1, 3).mean(axis=0)
        axis_pos_rmse_t[t] = np.sqrt(axis_pos_mse_t[t])
        axis_pos_mae_t[t] = np.abs(pos_errors).reshape(-1, 3).mean(axis=0)

        pos_mse_t[t] = (pos_errors**2).mean()
        pos_rmse_t[t] = np.sqrt(pos_mse_t[t])
        pos_mae_t[t] = np.abs(pos_errors).mean()
        ede_t[t] = np.linalg.norm(pos_errors, axis=2).mean()

        # Velocities
        vel_errors = y_true_vel[:, t] - y_pred_vel[:, t]
        axis_vel_mse_t[t] = (vel_errors**2).reshape(-1, 3).mean(axis=0)
        axis_vel_rmse_t[t] = np.sqrt(axis_vel_mse_t[t])
        axis_vel_mae_t[t] = np.abs(vel_errors).reshape(-1, 3).mean(axis=0)

        vel_mse_t[t] = (vel_errors**2).mean()
        vel_rmse_t[t] = np.sqrt(vel_mse_t[t])
        vel_mae_t[t] = np.abs(vel_errors).mean()

        # Accelerations
        acc_errors = y_true_acc[:, t] - y_pred_acc[:, t]
        axis_acc_mse_t[t] = (acc_errors**2).reshape(-1, 3).mean(axis=0)
        axis_acc_rmse_t[t] = np.sqrt(axis_acc_mse_t[t])
        axis_acc_mae_t[t] = np.abs(acc_errors).reshape(-1, 3).mean(axis=0)

        acc_mse_t[t] = (acc_errors**2).mean()
        acc_rmse_t[t] = np.sqrt(acc_mse_t[t])
        acc_mae_t[t] = np.abs(acc_errors).mean()

    return (
        pos_mse_t,
        pos_rmse_t,
        pos_mae_t,
        ede_t,
        vel_mse_t,
        vel_rmse_t,
        vel_mae_t,
        acc_mse_t,
        acc_rmse_t,
        acc_mae_t,
        axis_pos_mse_t,
        axis_pos_rmse_t,
        axis_pos_mae_t,
        axis_vel_mse_t,
        axis_vel_rmse_t,
        axis_vel_mae_t,
        axis_acc_mse_t,
        axis_acc_rmse_t,
        axis_acc_mae_t,
    )
