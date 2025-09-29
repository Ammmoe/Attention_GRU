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
from typing import List, Tuple, Optional
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
            )

            # Predicted trajectory
            ax.plot(
                pred_traj[:, -1, start],
                pred_traj[:, -1, start + 1],
                pred_traj[:, -1, start + 2],
                label=f"Agent {agent + 1} Pred",
                color=colors[agent],
                linestyle="--",
            )

        ax.set_title(f"Trajectory {traj_idx} (True vs Predicted)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.legend()

        # Save PNG
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"trajectory_{traj_idx}.png")
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
        )
        # Predicted trajectory
        ax.plot(
            y_pred[:, -1, start],
            y_pred[:, -1, start + 1],
            y_pred[:, -1, start + 2],
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
    Each row in past/true/pred should have concatenated agent features.
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
                markersize=2,
                label=past_label if agent_idx == 0 else None,
            )

            # True future
            ax.plot(
                [past_agent[-1, 0], true_agent[0, 0]],
                [past_agent[-1, 1], true_agent[0, 1]],
                [past_agent[-1, 2], true_agent[0, 2]],
                color=past_color,
            )
            ax.scatter(
                true_agent[0, 0],
                true_agent[0, 1],
                true_agent[0, 2],
                c=past_color,
                marker="o",
                s=10,
            )
            if true_agent.shape[0] > 1:
                ax.plot(
                    true_agent[:, 0],
                    true_agent[:, 1],
                    true_agent[:, 2],
                    f"{true_color}-",
                    marker="o",
                    markersize=2,
                    label=true_label if agent_idx == 0 else None,
                )

            # Predicted future
            ax.plot(
                [past_agent[-1, 0], pred_agent[0, 0]],
                [past_agent[-1, 1], pred_agent[0, 1]],
                [past_agent[-1, 2], pred_agent[0, 2]],
                color=past_color,
            )
            ax.scatter(
                pred_agent[0, 0],
                pred_agent[0, 1],
                pred_agent[0, 2],
                c=past_color,
                marker="o",
                s=10,
            )
            if pred_agent.shape[0] > 1:
                ax.plot(
                    pred_agent[:, 0],
                    pred_agent[:, 1],
                    pred_agent[:, 2],
                    f"{pred_color}-",
                    marker="o",
                    markersize=2,
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
