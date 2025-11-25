"""
trajectory_loss.py

Defines a custom loss function for trajectory prediction tasks.

Combines:
1. Mean Squared Error (MSE) loss for positional accuracy.
2. Directional loss based on cosine similarity between consecutive displacement vectors
    to encourage the predicted trajectory to follow the correct direction.

The total loss is a weighted sum of MSE and directional loss, controlled by `lambda_dir`.
"""
from torch import nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """
    Combines MSE loss with directional loss using cosine similarity between
    consecutive displacement vectors.
    """

    def __init__(self, lambda_dir=5e-6):
        """
        Args:
            lambda_dir (float): Weight for the directional loss component.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_dir = lambda_dir

    def forward(self, pred, target):
        """
        Args:
            pred: [batch, seq_len, num_features] predicted trajectory
            target: [batch, seq_len, num_features] ground truth trajectory
        Returns:
            loss: scalar tensor
        """
        # 1. MSE loss on positions
        mse_loss = self.mse(pred, target)

        # 2. Directional loss (cosine similarity between displacement vectors)
        pred_vecs = pred[:, 1:, :] - pred[:, :-1, :]
        target_vecs = target[:, 1:, :] - target[:, :-1, :]
        cosine_sim = F.cosine_similarity(pred_vecs, target_vecs, dim=-1)  # pylint: disable=E1102
        dir_loss = 1.0 - cosine_sim.mean()

        # 3. Combine
        loss = mse_loss + self.lambda_dir * dir_loss
        return loss
