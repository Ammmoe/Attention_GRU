"""
rnn_predictor.py

Defines a sequence-to-sequence vanilla RNN model for predicting future 2D or 3D trajectories.

The model uses an encoder-decoder architecture:
- The encoder RNN processes past trajectory points (LOOK_BACK timesteps).
- The decoder RNN predicts future trajectory points autoregressively.
- A fully connected layer maps RNN hidden states to output coordinates.

This model is suitable for trajectory prediction tasks in robotics, UAVs, and motion modeling.
"""

import torch
from torch import nn


class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence vanilla RNN model for trajectory prediction.

    Architecture:
        - Encoder RNN: Encodes past trajectory of length LOOK_BACK.
        - Decoder RNN: Autoregressively generates future trajectory of length FORWARD_LEN.
        - Fully connected layer: Maps RNN hidden states to output coordinates.

    Args:
        input_size (int): Number of input features per timestep (e.g., 2 for x,y or 3 for x,y,z).
        hidden_size (int): Number of hidden units in the RNN layers.
        output_size (int): Number of output features per timestep.
        num_layers (int): Number of stacked RNN layers for both encoder and decoder.

    Forward Pass:
        - Accepts a batch of past trajectories.
        - Encodes past trajectory using the encoder RNN.
        - Decodes future trajectory step by step, feeding each prediction as the next decoder input.
        - Returns predicted future sequence.
    """

    def __init__(self, input_size=3, hidden_size=128, output_size=3, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.RNN(output_size, hidden_size, num_layers, batch_first=True)
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

    def forward(self, x, pred_len=1):
        """
        Perform a forward pass to predict future trajectory points.

        Args:
            x (torch.Tensor): Input past trajectory of shape (batch_size, LOOK_BACK, input_size).
            pred_len (int): Number of future timesteps to predict (default=1).

        Returns:
            torch.Tensor: Predicted future trajectory.
                - Shape (batch_size, pred_len, output_size) if pred_len > 1
                - Shape (batch_size, output_size) if pred_len == 1

        Notes:
            - The model uses autoregressive decoding: each predicted point is fed back
            as input to the decoder for predicting the next timestep.
            - Suitable for predicting sequences of arbitrary length.
        """

        # Encode past trajectory
        _, h = self.encoder(x)  # h shape: (num_layers, batch, hidden_size)

        # First decoder input = last input point
        decoder_input = x[:, -1:, :]  # (batch, 1, input_size)
        outputs = []

        # Autoregressive decoding
        for _ in range(pred_len):
            out, h = self.decoder(decoder_input, h)
            pred = self.fc(out)  # (batch, 1, output_size)
            outputs.append(pred)
            decoder_input = pred  # feed prediction back

        outputs = torch.cat(outputs, dim=1)  # (batch, future_len, output_size)

        # Return squeezed version if only one step is predicted
        if pred_len == 1:
            return outputs.squeeze(1)  # (batch, output_size)
        return outputs
