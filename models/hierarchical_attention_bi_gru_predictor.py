# traj_predictor_bi_gru.py

import torch
from torch import nn
import torch.nn.functional as F


# ===========================
# Agent-level attention
# ===========================
class AgentAttention(nn.Module):
    """
    Agent-level attention: focuses on which agent matters at a given timestep.
    Input: (B, num_agents, agent_feat_dim)
    Output: (B, agent_feat_dim)
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


# ===========================
# Temporal attention (Bahdanau)
# Supports different encoder/decoder sizes (e.g., Bi-GRU encoder)
# ===========================
class Attention(nn.Module):
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
        B, T, _ = encoder_outputs.size()
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


# ===========================
# Hierarchical Seq2Seq Bi-GRU Predictor
# ===========================
class TrajPredictor(nn.Module):
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

    def forward(self, x, pred_len=1):
        """
        Args:
            x: (B, T, A*F)
        Returns:
            outputs_2d: (B*pred_len, output_size)
        """
        B, T, _ = x.size()
        A, F = 3, 3  # agents and features
        x_agents = x.view(B, T, A, F)

        # ---- Agent-level attention per timestep ----
        timestep_reprs = []
        for t in range(T):
            context, _ = self.agent_attn(x_agents[:, t, :, :])  # (B, F)
            timestep_reprs.append(context.unsqueeze(1))
        timestep_reprs = torch.cat(timestep_reprs, dim=1)  # (B, T, F)

        # ---- Temporal encoding ----
        encoder_outputs, hidden = self.encoder(timestep_reprs)  # (B, T, H*2), (2,B,H)
        # Concatenate forward/backward hidden and project to decoder size
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)  # (B, H*2)
        hidden_dec = torch.tanh(self.enc2dec(hidden_cat)).unsqueeze(0)  # (1, B, H)

        # ---- Autoregressive decoding ----
        decoder_input = x[:, -1:, :]  # last timestep features
        outputs = []

        for _ in range(pred_len):
            # Attention
            context, _ = self.temporal_attn(
                hidden_dec.squeeze(0), encoder_outputs
            )  # (B,H*2)
            context_proj = self.enc2dec(context)  # project to decoder size (B,H)
            dec_input = torch.cat(
                [decoder_input.squeeze(1), context_proj], dim=1
            ).unsqueeze(1)
            out, hidden_dec = self.decoder(dec_input, hidden_dec)  # (B,1,H), (1,B,H)
            pred = self.fc(out)  # (B,1,output_size)
            outputs.append(pred)
            decoder_input = pred

        outputs = torch.cat(outputs, dim=1)  # (B,pred_len,output_size)
        return outputs.reshape(-1, self.output_size)
