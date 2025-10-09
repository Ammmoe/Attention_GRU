"""
attention_bi_lstm_predictor.py

Defines a sequence-to-sequence bi-directional LSTM model with attention for predicting
future 2D/3D trajectories of a variable number of agents.
Supports flexible autoregressive decoding with or without teacher forcing.
"""

import torch
from torch import nn


class Attention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch, dec_hidden_size] (decoder hidden state h_t)
            encoder_outputs: [batch, seq_len, enc_hidden_size]
        Returns:
            attn_weights: [batch, seq_len]
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class TrajPredictor(nn.Module):
    """
    Seq2Seq bi-directional LSTM with attention for multi-agent trajectory prediction.

    Args:
        input_size (int): number of features per agent (2 or 3 for x,y,z).
        enc_hidden_size (int): hidden size of encoder LSTM.
        dec_hidden_size (int): hidden size of decoder LSTM.
        num_layers (int): number of stacked LSTM layers.
    """

    def __init__(
        self, input_size=3, enc_hidden_size=64, dec_hidden_size=64, num_layers=1
    ):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        # Shared encoder/decoder across agents
        self.encoder = nn.LSTM(
            input_size,
            enc_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(enc_hidden_size * 2, dec_hidden_size)
        self.decoder = nn.LSTM(
            input_size + enc_hidden_size * 2,
            dec_hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(dec_hidden_size, input_size)

        # Projection if encoder hidden size != decoder hidden size
        self.hidden_proj = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, src, tgt=None, pred_len=1):
        """
        Args:
            src: [batch, look_back, num_agents * input_size]
            tgt: [batch, pred_len, num_agents * input_size] (optional, for teacher forcing)
            pred_len: number of steps if tgt is None
        Returns:
            outputs: [batch, pred_len, num_agents * input_size]
        """
        _, _, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)

        tgt_agents = None
        if tgt is not None:
            tgt_agents = torch.split(tgt, self.input_size, dim=2)

        outputs_per_agent = []

        # Loop over agents
        for agent_idx in range(num_agents):
            # ---- Encoder ----
            enc_outputs, (h, c) = self.encoder(src_agents[agent_idx])

            # concat last forward and backward states
            h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, enc_hidden*2)
            c = torch.cat([c[-2], c[-1]], dim=1)

            # project to decoder size
            h = self.hidden_proj(h).unsqueeze(0)  # (1, batch, dec_hidden)
            c = self.hidden_proj(c).unsqueeze(0)  # (1, batch, dec_hidden)

            # expand to match decoder num_layers
            h = h.repeat(self.num_layers, 1, 1)
            c = c.repeat(self.num_layers, 1, 1)

            dec_input = src_agents[agent_idx][:, -1:, :]  # last observed point
            agent_outputs = []

            for t in range(pred_len):
                # Attention
                attn_weights = self.attention(h[-1], enc_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)

                # Decoder step
                rnn_input = torch.cat((dec_input, context), dim=2)
                output, (h, c) = self.decoder(rnn_input, (h, c))

                pred = self.fc_out(output.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))

                # Next input
                if tgt_agents is not None:
                    dec_input = tgt_agents[agent_idx][:, t : t + 1, :]
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))

        # Concatenate agent outputs along feature dimension
        outputs = torch.cat(
            outputs_per_agent, dim=2
        )  # (batch, pred_len, num_agents * input_size)

        return outputs
