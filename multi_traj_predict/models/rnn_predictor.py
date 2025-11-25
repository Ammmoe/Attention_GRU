"""
rnn_predictor.py

Sequence-to-sequence vanilla RNN model for multi-agent trajectory prediction in 2D or 3D space.

Features:
- Supports variable number of agents (drones).
- Autoregressive decoding with optional teacher forcing.
- Fully connected layer maps RNN hidden states to output coordinates.

Example usage:
    model = TrajPredictor(input_size=3, hidden_size=128, output_size=3)
    preds = model(src_tensor, tgt=tgt_tensor, pred_len=10)
"""

import torch
from torch import nn


class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence vanilla RNN model for multi-agent trajectory prediction.

    Architecture:
        - Encoder RNN: Encodes past trajectory for each agent.
        - Decoder RNN: Autoregressively generates future trajectory.
        - Fully connected layer: Maps RNN hidden states to output coordinates.

    Args:
        input_size (int): Number of input features per timestep (e.g., 2 for x,y or 3 for x,y,z).
        hidden_size (int): Number of hidden units in the RNN layers.
        output_size (int): Number of output features per timestep.
        num_layers (int): Number of stacked RNN layers for encoder and decoder.
    """

    def __init__(self, input_size=3, hidden_size=128, output_size=3, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.input_size = input_size
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.RNN(output_size, hidden_size, num_layers, batch_first=True)
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Apply Xavier initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize GRU weights
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Initialize Linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, src, tgt=None, pred_len=1):
        """
        Forward pass for multi-agent trajectory prediction.

        Args:
            src (torch.Tensor): Past trajectories for all agents,
                shape (batch, seq_len, num_agents * input_size).
            tgt (torch.Tensor, optional): Ground truth future trajectories for
                teacher forcing, shape (batch, pred_len, num_agents * output_size).
            pred_len (int): Number of steps to predict if `tgt` is not provided.

        Returns:
            torch.Tensor: Predicted future trajectories for all agents,
                shape (batch, pred_len, num_agents * output_size).

        Notes:
            - Supports variable number of agents.
            - Teacher forcing can be used if `tgt` is provided.
        """
        _, _, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)

        tgt_agents = None
        if tgt is not None:
            tgt_agents = torch.split(tgt, self.output_size, dim=2)

        outputs_per_agent = []

        for agent_idx in range(num_agents):
            # ---- Encoder ----
            _, h = self.encoder(
                src_agents[agent_idx]
            )  # h: (num_layers, batch, hidden_size)

            # Decoder input = last observed point
            dec_input = src_agents[agent_idx][:, -1:, :]
            agent_outputs = []

            for t in range(pred_len):
                # Decoder step
                out, h = self.decoder(dec_input, h)
                pred = self.fc(out.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))

                # Teacher forcing / autoregressive
                if tgt_agents is not None:
                    dec_input = tgt_agents[agent_idx][:, t : t + 1, :]
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))

        # Concatenate agent outputs along feature dimension
        outputs = torch.cat(outputs_per_agent, dim=2)
        return outputs
