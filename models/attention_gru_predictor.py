"""
attention_gru_predictor.py

Seq2Seq GRU with attention for 2D/3D multi-agent trajectory prediction.

- Encoder GRU encodes past positions.
- Attention selects relevant encoder outputs.
- Decoder GRU autoregressively generates future steps.
- FC layer maps hidden states to coordinates.

Supports flexible autoregressive decoding, and variable number of agents.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau-style additive attention.

    Computes a context vector as a weighted sum of encoder outputs,
    where weights are learned based on the current decoder hidden state.
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
        )  # (batch, seq_len, hidden_size)
        attn_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum (context vector)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        )  # (batch, 1, hidden_size)
        context = context.squeeze(1)

        return context, attn_weights


# Define Seq2Seq GRU Model with Attention
class TrajPredictor(nn.Module):
    """
    Seq2Seq GRU model with attention for multi-agent trajectory prediction.

    - Encoder GRU processes past trajectories.
    - Attention highlights relevant encoder outputs for each step.
    - Decoder GRU generates future trajectories autoregressively.
    - Fully connected layer maps decoder states to coordinates.
    """

    def __init__(self, input_size=3, hidden_size=64, output_size=3, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(
            output_size + hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

    def forward(self, src, tgt=None, pred_len=1):
        """
        Run forward pass.

        Args:
            src (Tensor): Past trajectories [batch, seq_len, num_agents * input_size].
            tgt (Tensor, optional): Ground-truth futures [batch, pred_len, num_agents * input_size].
            pred_len (int): Number of steps to predict if tgt is not given.

        Returns:
            Tensor: Predicted trajectories [batch, pred_len, num_agents * input_size].
        """
        _, _, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)

        tgt_agents = None
        if tgt is not None:
            tgt_agents = torch.split(tgt, self.input_size, dim=2)

        outputs_per_agent = []
        for agent_idx in range(num_agents):
            # --- Encoder ---
            enc_output, hidden = self.encoder(src_agents[agent_idx])
            # hidden: [num_layers, batch, hidden_size]
            hidden_dec = hidden  # no bidirectional merge

            # --- Decoder ---
            dec_input = src_agents[agent_idx][:, -1:, :]  # last input step
            agent_outputs = []
            for t in range(pred_len):
                # Attention: [batch, 1, enc_hidden_size]
                attn_weights = self.attention(hidden_dec[-1], enc_output)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_output)

                # GRU input = [batch, 1, input_size + enc_hidden_size]
                rnn_input = torch.cat((dec_input, context), dim=2)
                dec_out, hidden_dec = self.decoder(rnn_input, hidden_dec)

                # Project to output
                pred = self.fc(dec_out.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))

                # Teacher forcing vs autoregressive
                if tgt_agents is not None:
                    dec_input = tgt_agents[agent_idx][:, t : t + 1, :]
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))

        # --- Concatenate all agents ---
        outputs = torch.cat(
            outputs_per_agent, dim=2
        )  # [batch, pred_len, num_agents * input_size]
        return outputs
