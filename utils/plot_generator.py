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
from utils.scaler import scale_per_agent


def plot_trajectories(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    traj_ids: np.ndarray,
    plot_trajs: list,
    scaler: MinMaxScaler,
    agents: int,
    save_dir: str,
):
    """
    Plot 3D trajectories for selected trajectory indices.

    Args:
        y_true (np.ndarray): Ground-truth trajectories, shape (num_sequences, num_features).
        y_pred (np.ndarray): Predicted trajectories, same shape as y_true.
        traj_ids (np.ndarray): Array of trajectory indices corresponding to each sequence.
        plot_trajs (list): List of trajectory indices to plot.
        scaler (MinMaxScaler): Fitted scaler to inverse-transform trajectories.
        AGENTS (int): Number of agents in the trajectory.
        COLORS (list): List of colors for plotting agents.
        save_dir (str): Directory to save plot PNGs.

    Notes:
        - Assumes 3 features per agent (x, y, z) arranged consecutively in columns.
        - Each trajectory in `plot_trajs` will generate a separate PNG file.
        - If `COLORS` has fewer entries than `AGENTS`, colors will be reused from the 'tab10' colormap.
    """

    dim = 3  # x, y, z per agent
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(agents)]

    for traj_idx in plot_trajs:
        mask = traj_ids == traj_idx
        true_traj = scale_per_agent(y_true[mask], scaler, dim, inverse=True)
        pred_traj = scale_per_agent(y_pred[mask], scaler, dim, inverse=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for agent in range(agents):
            start = agent * dim

            # True trajectory
            ax.plot(
                true_traj[:, -1, start],
                true_traj[:, -1, start + 1],
                true_traj[:, -1, start + 2],
                label=f"Agent {agent + 1} True",
                color=colors[agent],
                linewidth=1,
            )

            # Predicted trajectory
            ax.plot(
                pred_traj[:, -1, start],
                pred_traj[:, -1, start + 1],
                pred_traj[:, -1, start + 2],
                label=f"Agent {agent + 1} Pred",
                color=colors[agent],
                linestyle="--",
                linewidth=1,
            )

        ax.set_title(f"Trajectory {traj_idx} (True vs Predicted)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.legend()

        # Save PNG
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"training_trajectory_{traj_idx}.png")
        plt.savefig(plot_path, dpi=150)

        # Show interactively
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
    per_agent: bool = False,  # plot all agents or just first agent
    num_features: int = 3,  # usually 3 for x,y,z
) -> None:
    """
    Plot multiple 3D trajectory sets as subplots.

    Each element in `trajectory_sets` should be a tuple of three numpy arrays:
        - past: Past trajectory points (shape [seq_len, num_agents * num_features])
        - true_line: True future trajectory points (shape [pred_len, num_agents * num_features])
        - pred_line: Predicted future trajectory points (shape [pred_len, num_agents * num_features])

    Parameters
    ----------
    trajectory_sets : list of tuple[np.ndarray, np.ndarray, np.ndarray]
        List of trajectory sets to plot.
    labels : list[str], optional
        Labels for past, true, and predicted trajectories, by default ["Past", "True", "Predicted"]
    colors : list[str], optional
        Colors for past, true, and predicted trajectories, by default ["b", "g", "r"]
    title : str, optional
        Overall title for the figure, by default "3D Trajectory Predictions (Random Examples)"
    figsize : tuple, optional
        Figure size, by default (15, 10)
    save_path : str, optional
        File path to save the figure. If None, the figure is not saved.
    per_agent : bool, optional
        Whether to plot trajectories for all agents or only the first agent, by default False
    num_features : int, optional
        Number of features per agent (usually 3 for x, y, z), by default 3

    Notes
    -----
    - Each row in the input arrays should concatenate all agent features.
    - The function automatically connects the last past point to the first future point.
    - Legend labels are only applied to the first agent to avoid clutter.
    """

    num_plots = len(trajectory_sets)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    fig = plt.figure(figsize=figsize)

    for i, (past, true_line, pred_line) in enumerate(trajectory_sets, 1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")

        # Determine number of agents
        num_agents = past.shape[1] // num_features

        # Decide which agents to plot
        agents_to_plot = range(num_agents) if per_agent else [0]

        for agent_idx in agents_to_plot:
            # reshape trajectories to [seq_len, num_agents, 3] then select agent
            past_agent = past[
                :, agent_idx * num_features : (agent_idx + 1) * num_features
            ]
            true_agent = true_line[
                :, agent_idx * num_features : (agent_idx + 1) * num_features
            ]
            pred_agent = pred_line[
                :, agent_idx * num_features : (agent_idx + 1) * num_features
            ]

            # Colors and labels
            past_color = colors[0] if colors else "b"
            true_color = colors[1] if colors else "g"
            pred_color = colors[2] if colors else "r"
            past_label = labels[0] if labels else "Past"
            true_label = labels[1] if labels else "True"
            pred_label = labels[2] if labels else "Predicted"

            # Plot past trajectory
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

            # True future
            ax.plot(
                [past_agent[-1, 0], true_agent[0, 0]],
                [past_agent[-1, 1], true_agent[0, 1]],
                [past_agent[-1, 2], true_agent[0, 2]],
                color=past_color,
                linewidth=1,
            )
            ax.scatter(
                true_agent[0, 0],
                true_agent[0, 1],
                true_agent[0, 2],
                c=past_color,
                marker="o",
                s=10,  # type: ignore[reportCallIssue]
            )
            if true_agent.shape[0] > 1:
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

            # Predicted future
            ax.plot(
                [past_agent[-1, 0], pred_agent[0, 0]],
                [past_agent[-1, 1], pred_agent[0, 1]],
                [past_agent[-1, 2], pred_agent[0, 2]],
                color=past_color,
                linewidth=1,
            )
            ax.scatter(
                pred_agent[0, 0],
                pred_agent[0, 1],
                pred_agent[0, 2],
                c=past_color,
                marker="o",
                s=10,  # type: ignore[reportCallIssue]
            )
            if pred_agent.shape[0] > 1:
                ax.plot(
                    pred_agent[:, 0],
                    pred_agent[:, 1],
                    pred_agent[:, 2],
                    f"{pred_color}-",
                    marker="o",
                    markersize=1.5,
                    linewidth=1,
                    label=pred_label if agent_idx == 0 else None,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.set_title(f"Sequence {i}")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_trajectories_with_velocity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    traj_ids: np.ndarray,
    plot_trajs: list,
    scaler: MinMaxScaler,
    agents: int,
    save_dir: str,
    plot_velocity: bool = True,
    velocity_scale: float = 0.5,
):
    """
    Plot 3D trajectories (true vs predicted) with optional velocity vectors.
    Handles cases where some trajectories or timesteps have fewer agents.
    """
    dim = 6  # x, y, z, vx, vy, vz per agent

    for traj_idx in plot_trajs:
        mask = traj_ids == traj_idx
        if not np.any(mask):
            continue  # skip if no data for this traj_idx

        # inverse scale
        true_traj = scale_per_agent(y_true[mask], scaler, dim, inverse=True)
        pred_traj = scale_per_agent(y_pred[mask], scaler, dim, inverse=True)

        # infer actual number of agents available for this trajectory
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
                continue  # skip incomplete agent data

            # True positions
            ax.plot(
                true_traj[:, start],
                true_traj[:, start + 1],
                true_traj[:, start + 2],
                label=f"Agent {agent + 1} True",
                color=colors[agent],
                linewidth=1,
            )

            # Predicted positions
            ax.plot(
                pred_traj[:, start],
                pred_traj[:, start + 1],
                pred_traj[:, start + 2],
                label=f"Agent {agent + 1} Pred",
                color=colors[agent],
                linestyle="--",
                linewidth=1,
            )

            if plot_velocity:
                # Handle cases where velocities might be missing (less than 6 features)
                if true_traj.shape[1] >= end and pred_traj.shape[1] >= end:
                    ax.quiver(
                        true_traj[:, start],
                        true_traj[:, start + 1],
                        true_traj[:, start + 2],
                        true_traj[:, start + 3],
                        true_traj[:, start + 4],
                        true_traj[:, start + 5],
                        length=velocity_scale,
                        color=colors[agent],
                        normalize=True,
                        alpha=0.5,
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
                        normalize=True,
                        alpha=0.5,
                    )

        ax.set_title(f"Trajectory {traj_idx} (Positions + Velocities)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"trajectory_{traj_idx}.png"), dpi=150)
        plt.show()


def plot_trajectories_with_velocity_acceleration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    traj_ids: np.ndarray,
    plot_trajs: list,
    scaler: MinMaxScaler,
    agents: int,
    save_dir: str,
    plot_velocity: bool = True,
    plot_acceleration: bool = True,
    velocity_scale: float = 0.5,
    acceleration_scale: float = 0.3,
):
    """
    Plot 3D trajectories (true vs predicted) with optional velocity and acceleration vectors.
    Handles cases where some trajectories or timesteps have fewer agents.
    Assumes 9 features per agent: pos(3), vel(3), acc(3).
    """
    dim = 9  # x,y,z, vx,vy,vz, ax,ay,az per agent

    for traj_idx in plot_trajs:
        mask = traj_ids == traj_idx
        if not np.any(mask):
            continue  # skip if no data for this traj_idx

        # inverse scale
        true_traj = scale_per_agent(y_true[mask], scaler, dim, inverse=True)
        pred_traj = scale_per_agent(y_pred[mask], scaler, dim, inverse=True)

        # infer actual number of agents available for this trajectory
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
                continue  # skip incomplete agent data

            # True positions
            ax.plot(
                true_traj[:, start],
                true_traj[:, start + 1],
                true_traj[:, start + 2],
                label=f"Agent {agent + 1} True",
                color=colors[agent],
                linewidth=1,
            )

            # Predicted positions
            ax.plot(
                pred_traj[:, start],
                pred_traj[:, start + 1],
                pred_traj[:, start + 2],
                label=f"Agent {agent + 1} Pred",
                color=colors[agent],
                linestyle="--",
                linewidth=1,
            )

            if plot_velocity and true_traj.shape[1] >= end and pred_traj.shape[1] >= end:
                # Velocity vectors (positions + 3:6)
                ax.quiver(
                    true_traj[:, start],
                    true_traj[:, start + 1],
                    true_traj[:, start + 2],
                    true_traj[:, start + 3],
                    true_traj[:, start + 4],
                    true_traj[:, start + 5],
                    length=velocity_scale,
                    color=colors[agent],
                    normalize=True,
                    alpha=0.5,
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
                    normalize=True,
                    alpha=0.5,
                )
            
            if plot_acceleration and true_traj.shape[1] >= end and pred_traj.shape[1] >= end:
                # Acceleration vectors (positions + 6:9)
                ax.quiver(
                    true_traj[:, start],
                    true_traj[:, start + 1],
                    true_traj[:, start + 2],
                    true_traj[:, start + 6],
                    true_traj[:, start + 7],
                    true_traj[:, start + 8],
                    length=acceleration_scale,
                    color=colors[agent],
                    normalize=True,
                    alpha=0.3,
                    linewidth=0.8,
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
                    normalize=True,
                    alpha=0.3,
                    linewidth=0.8,
                )

        ax.set_title(f"Trajectory {traj_idx} (Positions + Velocities + Accelerations)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"trajectory_{traj_idx}.png"), dpi=150)
        plt.show()
