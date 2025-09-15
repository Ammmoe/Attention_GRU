"""
attention_bi_gru_predictor.py

Defines a sequence-to-sequence GRU model with attention for predicting future 3D or 2D trajectories.
Supports bidirectional encoder and flexible autoregressive decoding.
"""

import torch
from torch import nn


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.
    Computes a context vector as a weighted sum of encoder outputs for each decoder step.
    """

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch_size, dec_hidden_size] current decoder hidden state
            encoder_outputs: [batch_size, seq_len, enc_hidden_size] encoder outputs
        Returns:
            attn_weights: [batch_size, seq_len] attention weights over encoder outputs
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence GRU model with attention for trajectory prediction.

    Features:
        - Bidirectional encoder GRU
        - Attention over encoder outputs
        - Unidirectional decoder GRU
        - Flexible decoding: teacher forcing or autoregressive

    Args:
        input_size (int): Number of input features per timestep (2 or 3 for x,y,z).
        enc_hidden_size (int): Number of hidden units in the encoder GRU.
        dec_hidden_size (int): Number of hidden units in the decoder GRU.
        output_size (int): Number of output features per timestep.
        num_layers (int): Number of stacked GRU layers.
    """

    def __init__(
        self,
        input_size=3,
        enc_hidden_size=64,
        dec_hidden_size=64,
        output_size=3,
        num_layers=1,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            input_size,
            enc_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(enc_hidden_size * 2, dec_hidden_size)
        self.decoder = nn.GRU(
            input_size + enc_hidden_size * 2,
            dec_hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(dec_hidden_size, output_size)
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

    def forward(self, src, tgt=None, pred_len=1):
        """
        Args:
            src: [batch_size, src_seq_len, input_size] input past trajectory
            tgt: [batch_size, tgt_seq_len, input_size] target future trajectory (optional, for teacher forcing)
            pred_len: int, number of steps to predict if tgt is None

        Returns:
            outputs: [batch_size, pred_len, output_size] predicted future trajectory
        """

        # ---- Encoder ----
        enc_outputs, hidden = self.encoder(src)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(0)

        if hidden.size(2) != self.decoder.hidden_size:
            hidden = nn.Linear(hidden.size(2), self.decoder.hidden_size).to(
                hidden.device
            )(hidden)

        # Determine prediction length
        if tgt is not None:
            pred_len = tgt.size(1)
        elif pred_len is None:
            raise ValueError("Either tgt or pred_len must be provided")

        # ---- Decoder ----
        outputs = []
        # Initial decoder input: last src point
        dec_input = src[:, -1:, :]

        for t in range(pred_len):
            # Attention
            attn_weights = self.attention(hidden[-1], enc_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)

            # Decoder GRU input: previous output + context
            rnn_input = torch.cat((dec_input, context), dim=2)
            output, hidden = self.decoder(rnn_input, hidden)
            pred = self.fc_out(output.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # Next decoder input
            if tgt is not None:
                # teacher forcing
                dec_input = tgt[:, t : t + 1, :]
            else:
                # autoregressive
                dec_input = pred.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)

        # Return squeezed version if only one step is predicted
        if pred_len == 1:
            return outputs.squeeze(1)  # (batch, output_size)
        return outputs
