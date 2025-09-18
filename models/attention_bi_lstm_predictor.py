"""
attention_bi_lstm_predictor.py

Defines a sequence-to-sequence LSTM model with attention for predicting future 3D or 2D trajectories.
Supports bidirectional encoder and flexible autoregressive decoding.
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
    Seq2Seq LSTM with attention for trajectory prediction.
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
        self.fc_out = nn.Linear(dec_hidden_size, output_size)

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        # projection if encoder hidden size != decoder hidden size
        self.hidden_proj = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, src, tgt=None, pred_len=1):
        """
        Args:
            src: [batch, src_len, input_size]
            tgt: [batch, tgt_len, input_size] (optional, for teacher forcing)
            pred_len: number of steps if tgt is None
        Returns:
            outputs: [batch, pred_len, output_size]
        """
        # ---- Encoder ----
        enc_outputs, (h, c) = self.encoder(src)

        # concat last forward and backward states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, enc_hidden*2)
        c = torch.cat([c[-2], c[-1]], dim=1)

        # project to decoder size
        h = self.hidden_proj(h).unsqueeze(0)  # (1, batch, dec_hidden)
        c = self.hidden_proj(c).unsqueeze(0)  # (1, batch, dec_hidden)

        # expand to match decoder num_layers
        h = h.repeat(self.num_layers, 1, 1)  # (num_layers, batch, dec_hidden)
        c = c.repeat(self.num_layers, 1, 1)  # (num_layers, batch, dec_hidden)

        if tgt is not None:
            pred_len = tgt.size(1)
        elif pred_len is None:
            raise ValueError("Either tgt or pred_len must be provided")

        outputs = []
        dec_input = src[:, -1:, :]  # last src point

        for t in range(pred_len):
            # Attention (only on hidden state h[-1])
            attn_weights = self.attention(h[-1], enc_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)

            # LSTM decoder step
            rnn_input = torch.cat((dec_input, context), dim=2)
            output, (h, c) = self.decoder(rnn_input, (h, c))

            pred = self.fc_out(output.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # Next input
            if tgt is not None:
                dec_input = tgt[:, t : t + 1, :]
            else:
                dec_input = pred.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)
        if pred_len == 1:
            return outputs.squeeze(1)
        return outputs
