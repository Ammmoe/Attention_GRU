"""
plot_generator.py

Provides utilities to visualize multi-agent 3D trajectory predictions.

This module includes a function to plot and save trajectories for
multiple agents, comparing ground-truth vs predicted positions in 3D.
It supports inverse scaling using a fitted MinMaxScaler, and can
save plots as PNG files while optionally displaying them interactively.
"""

import os
import math
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from multi_traj_predict.utils.scaler import scale_per_agent


def plot_multiagent_trajectories(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    traj_ids: np.ndarray,
    plot_trajs: list,
    scaler: MinMaxScaler,
    features_per_agent: int,
    save_dir: str,
    velocity_scale: float = 0.5,
    acceleration_scale: float = 0.3,
):
    """
    Plot 3D trajectories (True vs Predicted) for multiple agents, optionally with velocity
    and acceleration vectors depending on features_per_agent (3, 6, or 9).

    Args:
        y_true (np.ndarray): Ground-truth trajectories.
        y_pred (np.ndarray): Predicted trajectories.
        traj_ids (np.ndarray): Array of trajectory indices.
        plot_trajs (list): List of trajectory indices to plot.
        scaler (MinMaxScaler): Fitted scaler for inverse transform.
        agents (int): Number of agents per trajectory.
        features_per_agent (int): 3 (pos), 6 (pos+vel), or 9 (pos+vel+acc).
        save_dir (str): Directory to save output PNGs.
        velocity_scale (float): Scale of velocity vectors.
        acceleration_scale (float): Scale of acceleration vectors.
    """

    # Determine plotting configuration
    dim = features_per_agent
    plot_velocity = features_per_agent >= 6
    plot_acceleration = features_per_agent >= 9

    for traj_idx in plot_trajs:
        mask = traj_ids == traj_idx
        if not np.any(mask):
            continue

        # Inverse transform each trajectory
        true_traj = scale_per_agent(y_true[mask], scaler, dim, inverse=True)
        pred_traj = scale_per_agent(y_pred[mask], scaler, dim, inverse=True)

        # Determine available agents dynamically
        current_agents = min(true_traj.shape[1], pred_traj.shape[1]) // dim
        if current_agents == 0:
            continue

        colors = [plt.get_cmap("tab10")(i % 10) for i in range(current_agents)]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for agent in range(current_agents):
            start = agent * dim
            end = start + dim
            if end > true_traj.shape[1] or end > pred_traj.shape[1]:
                continue

            # Plot positions (x, y, z)
            ax.plot(
                true_traj[:, start],
                true_traj[:, start + 1],
                true_traj[:, start + 2],
                label=f"Agent {agent + 1} True",
                color=colors[agent],
                linewidth=1,
            )
            ax.plot(
                pred_traj[:, start],
                pred_traj[:, start + 1],
                pred_traj[:, start + 2],
                label=f"Agent {agent + 1} Pred",
                color=colors[agent],
                linestyle="--",
                linewidth=1,
            )

            # Plot velocity vectors if available
            if plot_velocity and end >= start + 6:
                ax.quiver(
                    true_traj[:, start],
                    true_traj[:, start + 1],
                    true_traj[:, start + 2],
                    true_traj[:, start + 3],
                    true_traj[:, start + 4],
                    true_traj[:, start + 5],
                    length=velocity_scale,
                    color=colors[agent],
                    alpha=0.5,
                    normalize=True,
                )
                ax.quiver(
                    pred_traj[:, start],
                    pred_traj[:, start + 1],
                    pred_traj[:, start + 2],
                    pred_traj[:, start + 3],
                    pred_traj[:, start + 4],
                    pred_traj[:, start + 5],
                    length=velocity_scale,
                    color=colors[agent],
                    linestyle="dashed",
                    alpha=0.5,
                    normalize=True,
                )

            # Plot acceleration vectors if available
            if plot_acceleration and end >= start + 9:
                ax.quiver(
                    true_traj[:, start],
                    true_traj[:, start + 1],
                    true_traj[:, start + 2],
                    true_traj[:, start + 6],
                    true_traj[:, start + 7],
                    true_traj[:, start + 8],
                    length=acceleration_scale,
                    color=colors[agent],
                    alpha=0.3,
                    linewidth=0.8,
                    normalize=True,
                )
                ax.quiver(
                    pred_traj[:, start],
                    pred_traj[:, start + 1],
                    pred_traj[:, start + 2],
                    pred_traj[:, start + 6],
                    pred_traj[:, start + 7],
                    pred_traj[:, start + 8],
                    length=acceleration_scale,
                    color=colors[agent],
                    linestyle="dashed",
                    alpha=0.3,
                    linewidth=0.8,
                    normalize=True,
                )

        # Dynamic title based on features used
        title_parts = ["Positions"]
        if plot_velocity:
            title_parts.append("Velocities")
        if plot_acceleration:
            title_parts.append("Accelerations")

        ax.set_title(f"Trajectory {traj_idx} ({' + '.join(title_parts)})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"trajectory_{traj_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()


def plot_attention_heatmap(attention_weights, save_dir):
    """
    Plots and saves an agent-level attention heatmap.

    Args:
        attention_weights (np.array): Shape (T, A) - Attention weights for one sample.
            - T: Number of timesteps (LOOK_BACK)
            - A: Number of agents
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(8, 12))
    ax = sns.heatmap(
        attention_weights,
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Attention Weight"},
    )
    ax.set_title("Agent Attention Weights Over Time")
    ax.set_xlabel("Agent Index")
    ax.set_ylabel("Input Timestep")
    ax.set_xticks(np.arange(attention_weights.shape[1]) + 0.5)
    ax.set_xticklabels([str(i) for i in range(attention_weights.shape[1])])

    # Save the figure
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plot_path = os.path.join(save_dir, "attention_heatmap.png")
    plt.savefig(plot_path, dpi=150)

    # Show interactively
    plt.show()
    plt.close()


def plot_inference_trajectory(
    y_true, y_pred, agents, save_dir, filename="trajectory.png"
):
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
            y_true[:, -1, start],
            y_true[:, -1, start + 1],
            y_true[:, -1, start + 2],
            label=f"Agent {agent + 1} True",
            color=colors[agent],
            linewidth=1,
        )
        # Predicted trajectory
        ax.plot(
            y_pred[:, -1, start],
            y_pred[:, -1, start + 1],
            y_pred[:, -1, start + 2],
            label=f"Agent {agent + 1} Pred",
            color=colors[agent],
            linestyle="--",
            linewidth=1,
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


def plot_3d_trajectories_subplots(
    trajectory_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    title: str = "3D Trajectory Predictions (Random Examples)",
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
    per_agent: bool = False,
    num_features: int = 3,  # can be 3, 6, or 9
    velocity_scale: float = 0.5,
    acceleration_scale: float = 0.3,
) -> None:
    """
    Plot multiple 3D trajectory sets as subplots with optional velocity and acceleration vectors.

    Each element in `trajectory_sets` should be a tuple of three numpy arrays:
        - past: Past trajectory points (shape [seq_len, num_agents * num_features])
        - true_line: True future trajectory points
        - pred_line: Predicted future trajectory points

    Parameters
    ----------
    num_features : int
        Number of features per agent:
            3 -> position only (x, y, z)
            6 -> position + velocity (vx, vy, vz)
            9 -> position + velocity + acceleration (ax, ay, az)
    """

    num_plots = len(trajectory_sets)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    fig = plt.figure(figsize=figsize)

    for i, (past, true_line, pred_line) in enumerate(trajectory_sets, 1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")

        num_agents = past.shape[1] // num_features
        agents_to_plot = range(num_agents) if per_agent else [0]

        for agent_idx in agents_to_plot:
            base = agent_idx * num_features

            past_agent = past[:, base : base + 3]
            true_agent = true_line[:, base : base + 3]
            pred_agent = pred_line[:, base : base + 3]

            past_color = colors[0] if colors else "b"
            true_color = colors[1] if colors else "g"
            pred_color = colors[2] if colors else "r"
            past_label = labels[0] if labels else "Past"
            true_label = labels[1] if labels else "True"
            pred_label = labels[2] if labels else "Predicted"

            # --- Position Trajectories ---
            ax.plot(
                past_agent[:, 0],
                past_agent[:, 1],
                past_agent[:, 2],
                f"{past_color}-",
                marker="o",
                markersize=1.5,
                linewidth=1,
                label=past_label if agent_idx == 0 else None,
            )

            ax.plot(
                true_agent[:, 0],
                true_agent[:, 1],
                true_agent[:, 2],
                f"{true_color}-",
                marker="o",
                markersize=1.5,
                linewidth=1,
                label=true_label if agent_idx == 0 else None,
            )

            ax.plot(
                pred_agent[:, 0],
                pred_agent[:, 1],
                pred_agent[:, 2],
                f"{pred_color}--",
                marker="o",
                markersize=1.5,
                linewidth=1,
                label=pred_label if agent_idx == 0 else None,
            )

            # --- Velocity Arrows ---
            if num_features >= 6:
                true_vel = true_line[:, base + 3 : base + 6]
                pred_vel = pred_line[:, base + 3 : base + 6]
                ax.quiver(
                    true_agent[:, 0],
                    true_agent[:, 1],
                    true_agent[:, 2],
                    true_vel[:, 0],
                    true_vel[:, 1],
                    true_vel[:, 2],
                    length=velocity_scale,
                    normalize=True,
                    color=true_color,
                    alpha=0.4,
                )
                ax.quiver(
                    pred_agent[:, 0],
                    pred_agent[:, 1],
                    pred_agent[:, 2],
                    pred_vel[:, 0],
                    pred_vel[:, 1],
                    pred_vel[:, 2],
                    length=velocity_scale,
                    normalize=True,
                    color=pred_color,
                    alpha=0.4,
                    linestyle="dashed",
                )

            # --- Acceleration Arrows ---
            if num_features == 9:
                true_acc = true_line[:, base + 6 : base + 9]
                pred_acc = pred_line[:, base + 6 : base + 9]
                ax.quiver(
                    true_agent[:, 0],
                    true_agent[:, 1],
                    true_agent[:, 2],
                    true_acc[:, 0],
                    true_acc[:, 1],
                    true_acc[:, 2],
                    length=acceleration_scale,
                    normalize=True,
                    color=true_color,
                    alpha=0.3,
                )
                ax.quiver(
                    pred_agent[:, 0],
                    pred_agent[:, 1],
                    pred_agent[:, 2],
                    pred_acc[:, 0],
                    pred_acc[:, 1],
                    pred_acc[:, 2],
                    length=acceleration_scale,
                    normalize=True,
                    color=pred_color,
                    alpha=0.3,
                    linestyle="dashed",
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Sequence {i}")
        if i == 1:  # avoid repeated legends
            ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
