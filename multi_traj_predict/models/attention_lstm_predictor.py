"""
attention_lstm_predictor.py

Sequence-to-sequence LSTM model with Bahdanau-style attention for multi-agent
trajectory prediction in 2D or 3D space.

This module provides:

- Attention: Additive attention mechanism to compute context vectors from encoder outputs.
- TrajPredictor: LSTM-based encoder-decoder model with attention, supporting
    variable numbers of agents and optional teacher forcing during decoding.

Example usage:
    model = TrajPredictor(input_size=3, hidden_size=64, output_size=3)
    preds = model(src_tensor, tgt=tgt_tensor, pred_len=10)
"""

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau-style additive attention mechanism.

    Computes a context vector as a weighted sum of encoder outputs, where
    weights are computed from the current decoder hidden state.

    Args:
        hidden_size (int): Dimensionality of encoder/decoder hidden states.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Forward pass for computing attention.

        Args:
            decoder_hidden (torch.Tensor): Current decoder hidden state,
                shape (batch, hidden_size).
            encoder_outputs (torch.Tensor): Encoder outputs for all time steps,
                shape (batch, seq_len, hidden_size).

        Returns:
            context (torch.Tensor): Context vector computed as weighted sum of
                encoder outputs, shape (batch, hidden_size).
            attn_weights (torch.Tensor): Attention weights over encoder outputs,
                shape (batch, seq_len).
        """
        _, seq_len, _ = encoder_outputs.size()

        # Repeat decoder hidden across sequence length
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate and compute energy scores
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


class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence LSTM model with attention for multi-agent trajectory prediction.

    Supports:
        - Variable number of agents (multi-agent predictions).
        - Autoregressive decoding.
        - Optional teacher forcing using target future trajectories.

    Args:
        input_size (int): Number of input features per timestep (e.g., 2 for x,y or 3 for x,y,z).
        hidden_size (int): Number of hidden units in the LSTM layers.
        output_size (int): Number of output features per timestep.
        num_layers (int): Number of stacked LSTM layers for encoder and decoder.
    """

    def __init__(self, input_size=3, hidden_size=64, output_size=3, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(
            output_size + hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.output_size = output_size
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            torch.Tensor: Predicted trajectories for all agents,
                shape (batch, pred_len, num_agents * output_size).
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
            enc_outputs, (h, c) = self.encoder(src_agents[agent_idx])

            # expand to match decoder num_layers
            h = h.repeat(self.num_layers, 1, 1)
            c = c.repeat(self.num_layers, 1, 1)

            dec_input = src_agents[agent_idx][:, -1:, :]  # last input step
            agent_outputs = []

            for t in range(pred_len):
                # Attention
                context, _ = self.attention(h[-1], enc_outputs)

                # Decoder step
                rnn_input = torch.cat((dec_input, context.unsqueeze(1)), dim=2)
                out, (h, c) = self.decoder(rnn_input, (h, c))

                pred = self.fc(out.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))

                # Teacher forcing / autoregressive
                if tgt_agents is not None:
                    dec_input = tgt_agents[agent_idx][:, t : t + 1, :]
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))

        outputs = torch.cat(outputs_per_agent, dim=2)
        return outputs
