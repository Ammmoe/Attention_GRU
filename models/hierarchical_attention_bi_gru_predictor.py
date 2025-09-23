# traj_predictor_bi_gru.py

import torch
from torch import nn
import torch.nn.functional as F


# Agent-level attention
class AgentAttention(nn.Module):
    """
    Agent-level attention mechanism.

    At each timestep, this module learns to focus on the most relevant
    agent(s) among a variable number of agents, producing a weighted
    summary representation of all agents.

    Args:
        agent_feat_dim (int): Dimensionality of features per agent (default: 3 for x,y,z).
        hidden_size (int): Size of the hidden layer used to compute attention scores.

    Input:
        agent_features (Tensor): Shape (B, A, F)
            - B: Batch size
            - A: Number of agents
            - F: Feature dimension per agent

    Output:
        context (Tensor): Shape (B, F), weighted agent feature representation.
        attn_weights (Tensor): Shape (B, A), attention weights per agent.
    """

    def __init__(self, agent_feat_dim=3, hidden_size=64):
        super().__init__()
        self.attn = nn.Linear(agent_feat_dim, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, agent_features):
        """
        agent_features: (B, A, F)
            - A = number of agents
            - F = features per agent (3 for x,y,z)

        Returns:
            timestep_repr: (B, F) weighted agent representation
            attn_weights: (B, A) agent attention weights
        """
        energy = torch.tanh(self.attn(agent_features))  # (B, A, H)
        attn_scores = self.v(energy).squeeze(-1)  # (B, A)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, A)

        # Weighted sum of agents
        context = torch.bmm(attn_weights.unsqueeze(1), agent_features).squeeze(
            1
        )  # (B, F)
        return context, attn_weights


# Temporal attention (Bahdanau)
# Supports different encoder/decoder sizes (e.g., Bi-GRU encoder)
class Attention(nn.Module):
    """
    Temporal attention mechanism (Bahdanau-style).

    Computes attention over encoder outputs at all timesteps, conditioned
    on the current decoder hidden state.

    Args:
        encoder_hidden_size (int): Size of encoder hidden states.
        decoder_hidden_size (int): Size of decoder hidden states.
        attn_hidden_size (int, optional): Hidden size for attention energy computation.
                                        Defaults to max(encoder_hidden_size, decoder_hidden_size).

    Input:
        decoder_hidden (Tensor): Shape (B, dec_H), current decoder hidden state.
        encoder_outputs (Tensor): Shape (B, T, enc_H), encoder hidden states over all timesteps.

    Output:
        context (Tensor): Shape (B, enc_H), context vector as weighted sum of encoder outputs.
        attn_weights (Tensor): Shape (B, T), attention weights over timesteps.
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size, attn_hidden_size=None):
        super().__init__()
        if attn_hidden_size is None:
            attn_hidden_size = max(encoder_hidden_size, decoder_hidden_size)
        self.attn = nn.Linear(
            encoder_hidden_size + decoder_hidden_size, attn_hidden_size
        )
        self.v = nn.Linear(attn_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (B, decoder_hidden_size)
            encoder_outputs: (B, T, encoder_hidden_size)
        Returns:
            context: (B, encoder_hidden_size)
            attn_weights: (B, T)
        """
        _, T, _ = encoder_outputs.size()
        # expand decoder hidden across time steps
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(
            1, T, 1
        )  # (B, T, dec_H)
        energy = torch.tanh(
            self.attn(torch.cat([decoder_hidden_exp, encoder_outputs], dim=2))
        )  # (B,T,attn_H)
        attn_scores = self.v(energy).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(
            1
        )  # (B, enc_H)
        return context, attn_weights


# Hierarchical Seq2Seq Bi-GRU Predictor
class TrajPredictor(nn.Module):
    """
    Hierarchical Seq2Seq trajectory predictor with agent-level and temporal attention.

    Architecture:
        1. Agent-level attention: Computes weighted agent representation per timestep.
        2. Bidirectional GRU encoder: Encodes temporal sequence of timestep representations.
        3. Temporal attention: Allows decoder to focus on relevant timesteps.
        4. GRU decoder: Generates future trajectories autoregressively.
        5. Fully connected output layer: Maps decoder hidden states to output predictions.

    Args:
        agent_feat_dim (int): Number of features per agent (default: 3 for x,y,z).
        hidden_size (int): Hidden size for GRU layers.
        output_size (int): Number of output features (e.g., 9 for 3 agents × 3 features).

    Input:
        x (Tensor): Shape (B, T, A*F)
            - B: Batch size
            - T: Input sequence length
            - A: Number of agents
            - F: Features per agent
        pred_len (int): Number of future timesteps to predict.

    Output:
        outputs_2d (Tensor): Shape (B * pred_len, output_size)
            Flattened predictions for compatibility with training pipelines.
    """

    def __init__(self, agent_feat_dim=3, hidden_size=64, output_size=9):
        super().__init__()
        self.agent_attn = AgentAttention(agent_feat_dim, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            agent_feat_dim, hidden_size, batch_first=True, bidirectional=True
        )

        # Linear layer to project Bi-GRU hidden to decoder size
        self.enc2dec = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder GRU
        self.decoder = nn.GRU(output_size + hidden_size, hidden_size, batch_first=True)

        # Attention
        self.temporal_attn = Attention(
            encoder_hidden_size=hidden_size * 2, decoder_hidden_size=hidden_size
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pred_len=1, return_attn=False):  # Add return_attn flag
        """
        Args:
            x: (B, T, A*F)
            return_attn: (bool) If True, returns attention weights.
        Returns:
            outputs_2d: (B*pred_len, output_size)
            attn_weights: (B, T, A) - Optional, returned if return_attn is True
        """
        B, T, feat_total = x.size()
        agent_feat_dim = 3
        A = feat_total // agent_feat_dim
        x_agents = x.view(B, T, A, agent_feat_dim)

        # ---- Agent-level attention per timestep ----
        timestep_reprs = []
        agent_attn_weights_list = []  # Create a list to store weights
        for t in range(T):
            context, agent_attn_weights = self.agent_attn(
                x_agents[:, t, :, :]
            )  # (B, F), (B, A)
            timestep_reprs.append(context.unsqueeze(1))
            agent_attn_weights_list.append(
                agent_attn_weights.unsqueeze(1)
            )  # Append weights

        timestep_reprs = torch.cat(timestep_reprs, dim=1)  # (B, T, F)
        # Stack all attention weights into a single tensor
        all_agent_attn_weights = torch.cat(agent_attn_weights_list, dim=1)  # (B, T, A)

        # ---- Temporal encoding ----
        encoder_outputs, hidden = self.encoder(timestep_reprs) # (B, T, 2*H), (2, B, H)
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1) # (B, 2*H)
        hidden_dec = torch.tanh(self.enc2dec(hidden_cat)).unsqueeze(0) # (1, B, H)

        # ---- Autoregressive decoding ----
        decoder_input = x[:, -1:, :] # (B, 1, A*F)
        outputs = []

        for _ in range(pred_len):
            context, _ = self.temporal_attn(hidden_dec.squeeze(0), encoder_outputs)
            context_proj = self.enc2dec(context)
            dec_input = torch.cat(
                [decoder_input.squeeze(1), context_proj], dim=1
            ).unsqueeze(1)
            out, hidden_dec = self.decoder(dec_input, hidden_dec)
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred

        outputs = torch.cat(outputs, dim=1)

        # Conditionally return the attention weights along with the output
        if return_attn:
            return outputs.reshape(-1, self.output_size), all_agent_attn_weights

        return outputs.reshape(-1, self.output_size)
