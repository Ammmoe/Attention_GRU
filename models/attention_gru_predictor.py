"""
_attention_gru_predictor.py

Defines a sequence-to-sequence GRU model with attention for predicting future 3D or 2D trajectories.

The model uses an encoder-decoder architecture with attention:
- The encoder GRU processes past trajectory points and outputs hidden states.
- The attention mechanism computes context vectors from encoder hidden states at each step.
- The decoder GRU predicts future trajectory points autoregressively with attention.
- A fully connected layer maps hidden states to output coordinates.

This model is suitable for trajectory prediction tasks in robotics, UAVs, and motion modeling.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch, hidden_size) current decoder hidden state
            encoder_outputs: (batch, seq_len, hidden_size) all encoder outputs

        Returns:
            context: (batch, hidden_size) weighted sum of encoder outputs
            attn_weights: (batch, seq_len) attention weights
        """
        _, seq_len, _ = encoder_outputs.size()

        # Repeat decoder hidden across sequence length
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate and compute scores
        energy = torch.tanh(
            self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2))
        )
        attn_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum (context vector)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        )  # (batch, 1, hidden_size)
        context = context.squeeze(1)

        return context, attn_weights


# Define Seq2Seq GRU Model with Attention
class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence GRU model with attention for trajectory prediction.

    Architecture:
        - Encoder GRU: Encodes past trajectory of length LOOK_BACK into hidden states.
        - Attention: Computes weighted context vectors over encoder outputs.
        - Decoder GRU: Autoregressively generates future trajectory of length FORWARD_LEN using context.
        - Fully connected layer: Maps decoder hidden states to output coordinates.

    Args:
        input_size (int): Number of input features per timestep (e.g., 2 for x,y or 3 for x,y,z).
        hidden_size (int): Number of hidden units in the GRU layers.
        output_size (int): Number of output features per timestep.
        num_layers (int): Number of stacked GRU layers for both encoder and decoder.
    """

    def __init__(self, input_size=3, hidden_size=128, output_size=3, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(
            output_size + hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, pred_len=1):
        """
        Args:
            x (torch.Tensor): Input past trajectory of shape (batch_size, LOOK_BACK, input_size).
            pred_len (int): Number of future timesteps to predict.

        Returns:
            torch.Tensor: Predicted future trajectory.
                - Shape (batch_size, pred_len, output_size) if pred_len > 1
                - Shape (batch_size, output_size) if pred_len == 1
        """
        # Encode past trajectory
        encoder_outputs, hidden = self.encoder(x)

        # First decoder input = last input point
        decoder_input = x[:, -1:, :]

        outputs = []
        for _ in range(pred_len):
            # Attention
            dec_hidden_t = hidden[-1]  # (batch, hidden_size)
            context, _ = self.attention(dec_hidden_t, encoder_outputs)

            # Combine decoder input + context
            dec_input = torch.cat([decoder_input.squeeze(1), context], dim=1)
            dec_input = dec_input.unsqueeze(1)  # (batch, 1, input_size+hidden_size)

            # Decode
            out, hidden = self.decoder(dec_input, hidden)
            pred = self.fc(out)  # (batch, 1, output_size)

            outputs.append(pred)
            decoder_input = pred  # autoregressive feedback

        outputs = torch.cat(outputs, dim=1)
        if pred_len == 1:
            return outputs.squeeze(1)
        return outputs
