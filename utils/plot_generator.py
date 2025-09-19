"""
plot_generator.py

Provides utilities to visualize multi-agent 3D trajectory predictions.

This module includes a function to plot and save trajectories for
multiple agents, comparing ground-truth vs predicted positions in 3D.
It supports inverse scaling using a fitted MinMaxScaler, and can
save plots as PNG files while optionally displaying them interactively.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
        true_traj = scaler.inverse_transform(y_true[mask])
        pred_traj = scaler.inverse_transform(y_pred[mask])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for agent in range(agents):
            start = agent * dim

            # True trajectory
            ax.plot(
                true_traj[:, start],
                true_traj[:, start + 1],
                true_traj[:, start + 2],
                label=f"Agent {agent + 1} True",
                color=colors[agent],
            )

            # Predicted trajectory
            ax.plot(
                pred_traj[:, start],
                pred_traj[:, start + 1],
                pred_traj[:, start + 2],
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
