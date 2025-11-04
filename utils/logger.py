"""
logger.py

Provides utilities to create a timestamped experiment logger.

This module simplifies logging for experiments or scripts by automatically
creating a timestamped experiment folder and configuring a logger that writes
to both a log file and optionally to the console. Useful for machine learning
experiments, data processing scripts, or any project where organized logging
is desired.

Functions:
----------
get_logger(exp_root="experiments", log_name="train.log")
    Creates a logger and a timestamped experiment folder, returning both
    for use in scripts.
"""

import logging
import os
from datetime import datetime
from utils.model_evaluator import (
    evaluate_metrics_multi_agent_per_timestep,
    evaluate_metrics_multi_agent_pos_vel_per_timestep,
    evaluate_metrics_multi_agent_pos_vel_acc_per_timestep,
)


def get_logger(exp_root="experiments", log_name="train.log"):
    """
    Set up a logger that writes logs to a timestamped experiment folder.

    Args:
        exp_root (str, optional): Root directory to store experiment folders. Defaults to "experiments".
        log_name (str, optional): Name of the log file. Defaults to "train.log".

    Returns:
        tuple:
            logger (logging.Logger): Configured logger object.
            exp_dir (str): Path to the created timestamped experiment folder.

    Notes:
        - A new folder is created inside `exp_root` with the current timestamp (YYYYMMDD_HHMMSS).
        - Logger writes to both the log file and the console.
        - Calling this function multiple times avoids adding duplicate handlers.
    """

    # Ensure root experiments folder exists
    os.makedirs(exp_root, exist_ok=True)

    # Create a timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(exp_root, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if this function is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(exp_dir, log_name))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, exp_dir


def get_inference_logger(exp_dir, log_name="inference.log"):
    """
    Set up an inference logger that appends logs into a fixed file
    inside an existing experiment folder.

    Args:
        exp_dir (str): Path to the existing experiment folder.
        log_name (str, optional): Log file name. Defaults to "inference.log".

    Returns:
        logging.Logger: Configured logger object.
    """

    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, log_name)

    # Use experiment folder name to make logger unique
    logger = logging.getLogger(f"inference_{os.path.basename(exp_dir)}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicates
        fh = logging.FileHandler(log_path, mode="a")  # append mode
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # --- Write a session header ---
        logger.info("\n")
        logger.info("%s", "=" * 80)
        logger.info(
            "NEW INFERENCE SESSION - %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info("%s", "=" * 80)

    return logger


def log_metrics_for_features(
    y_true, y_pred, scaler_y, num_agents, features_per_agent, logger
):
    """
    Log evaluation metrics per timestep for multi-agent trajectories.

    Selects the appropriate evaluation function based on features_per_agent:
    - 3 features: position only
    - 6 features: position + velocity
    - 9 features: position + velocity + acceleration

    Args:
        y_true (torch.Tensor): Ground truth tensor
        y_pred (torch.Tensor): Predicted tensor
        scaler_y (MinMaxScaler): Scaler for inverse transform
        num_agents (int): Number of agents
        features_per_agent (int): Features per agent (3, 6, or 9)
        logger: Logger instance
    """

    if features_per_agent == 3:
        metrics = evaluate_metrics_multi_agent_per_timestep(
            y_true, y_pred, scaler_y, num_agents
        )
        mse_t, rmse_t, mae_t, ede_t, axis_mse_t, axis_rmse_t, axis_mae_t = metrics

        header = (
            f"{'Timestep':>8} | {'EDE':>10} | {'MSE':>10} | {'RMSE':>10} | {'MAE':>10} | "
            + " ".join(
                [
                    f"{m}_x".rjust(10) + f" {m}_y".rjust(10) + f" {m}_z".rjust(10)
                    for m in ["MSE", "RMSE", "MAE"]
                ]
            )
        )
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        for t, (ede, mse, rmse, mae, axis_mse, axis_rmse, axis_mae) in enumerate(
            zip(ede_t, mse_t, rmse_t, mae_t, axis_mse_t, axis_rmse_t, axis_mae_t)
        ):
            logger.info(
                "%8d | %10.6f | %10.6f | %10.6f | %10.6f | "
                "%10.6f %10.6f %10.6f | "
                "%10.6f %10.6f %10.6f | "
                "%10.6f %10.6f %10.6f",
                t,
                ede,
                mse,
                rmse,
                mae,
                axis_mse[0],
                axis_mse[1],
                axis_mse[2],
                axis_rmse[0],
                axis_rmse[1],
                axis_rmse[2],
                axis_mae[0],
                axis_mae[1],
                axis_mae[2],
            )

        logger.info("-" * len(header))
        logger.info(
            "%8s | %10.6f | %10.6f | %10.6f | %10.6f | "
            "%10.6f %10.6f %10.6f | "
            "%10.6f %10.6f %10.6f | "
            "%10.6f %10.6f %10.6f",
            "Average",
            ede_t.mean(),
            mse_t.mean(),
            rmse_t.mean(),
            mae_t.mean(),
            axis_mse_t.mean(axis=0)[0],
            axis_mse_t.mean(axis=0)[1],
            axis_mse_t.mean(axis=0)[2],
            axis_rmse_t.mean(axis=0)[0],
            axis_rmse_t.mean(axis=0)[1],
            axis_rmse_t.mean(axis=0)[2],
            axis_mae_t.mean(axis=0)[0],
            axis_mae_t.mean(axis=0)[1],
            axis_mae_t.mean(axis=0)[2],
        )
        logger.info("-" * len(header))

    elif features_per_agent == 6:
        metrics = evaluate_metrics_multi_agent_pos_vel_per_timestep(
            y_true, y_pred, scaler_y, num_agents
        )
        (
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
        ) = metrics

        header = (
            f"{'Timestep':>8} | {'EDE':>10} | "
            f"{'Pos_MSE':>10} | {'Pos_RMSE':>10} | {'Pos_MAE':>10} | "
            f"{'Vel_MSE':>10} | {'Vel_RMSE':>10} | {'Vel_MAE':>10}"
        )
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        for t in range(len(ede_t)):
            logger.info(
                "%8d | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f",
                t,
                ede_t[t],
                pos_mse_t[t],
                pos_rmse_t[t],
                pos_mae_t[t],
                vel_mse_t[t],
                vel_rmse_t[t],
                vel_mae_t[t],
            )

        logger.info("-" * len(header))
        logger.info(
            "%8s | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f",
            "Average",
            ede_t.mean(),
            pos_mse_t.mean(),
            pos_rmse_t.mean(),
            pos_mae_t.mean(),
            vel_mse_t.mean(),
            vel_rmse_t.mean(),
            vel_mae_t.mean(),
        )
        logger.info("-" * len(header))

        # Optional: Log per-axis position and velocity metrics here

    elif features_per_agent == 9:
        metrics = evaluate_metrics_multi_agent_pos_vel_acc_per_timestep(
            y_true, y_pred, scaler_y, num_agents
        )
        (
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
        ) = metrics

        header = (
            f"{'Timestep':>8} | {'EDE':>10} | "
            f"{'Pos_MSE':>10} | {'Pos_RMSE':>10} | {'Pos_MAE':>10} | "
            f"{'Vel_MSE':>10} | {'Vel_RMSE':>10} | {'Vel_MAE':>10} | "
            f"{'Acc_MSE':>10} | {'Acc_RMSE':>10} | {'Acc_MAE':>10}"
        )
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        for t in range(len(ede_t)):
            logger.info(
                "%8d | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f",
                t,
                ede_t[t],
                pos_mse_t[t],
                pos_rmse_t[t],
                pos_mae_t[t],
                vel_mse_t[t],
                vel_rmse_t[t],
                vel_mae_t[t],
                acc_mse_t[t],
                acc_rmse_t[t],
                acc_mae_t[t],
            )

        logger.info("-" * len(header))
        logger.info(
            "%8s | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f | %10.6f",
            "Average",
            ede_t.mean(),
            pos_mse_t.mean(),
            pos_rmse_t.mean(),
            pos_mae_t.mean(),
            vel_mse_t.mean(),
            vel_rmse_t.mean(),
            vel_mae_t.mean(),
            acc_mse_t.mean(),
            acc_rmse_t.mean(),
            acc_mae_t.mean(),
        )
        logger.info("-" * len(header))

        # Optional: Log per-axis position, velocity, and acceleration metrics here

    else:
        logger.warning(
            f"Unsupported FEATURES_PER_AGENT={features_per_agent} for evaluation logging."
        )
