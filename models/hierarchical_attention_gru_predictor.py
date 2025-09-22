import torch
from torch import nn
import torch.nn.functional as F


class AgentAttention(nn.Module):
    """
    Agent-level attention: focuses on which agent matters at a given timestep.
    Input: (B, num_agents, agent_feat_dim)
    Output: (B, agent_hidden_dim)
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
        context = torch.bmm(attn_weights.unsqueeze(1), agent_features)  # (B, 1, F)
        context = context.squeeze(1)  # (B, F)

        return context, attn_weights


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


class TrajPredictor(nn.Module):
    """
    Hierarchical Seq2Seq GRU with:
        1) Agent-level attention per timestep
        2) Temporal attention across timesteps
    """

    def __init__(self, agent_feat_dim=3, hidden_size=64, output_size=9):
        super().__init__()
        self.agent_attn = AgentAttention(agent_feat_dim, hidden_size)

        # Encoder processes per-timestep agent summaries
        self.encoder = nn.GRU(agent_feat_dim, hidden_size, batch_first=True)

        # Decoder (same as before, but conditioned on context)
        self.decoder = nn.GRU(output_size + hidden_size, hidden_size, batch_first=True)

        self.temporal_attn = Attention(
            hidden_size
        )  # reuse your earlier Attention class
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pred_len=1):
        """
        Args:
            x: (B, T, A*F)  -- e.g. A=3, F=3 → 9 features per timestep
        """
        B, T, _ = x.size()
        A, agent_feat = 3, 3  # (can be params too)

        # Reshape to per-agent structure: (B, T, A, agent_feat)
        x_agents = x.view(B, T, A, agent_feat)

        # ---- Agent-level attention at each timestep ----
        timestep_reprs = []
        agent_attn_all = []
        for t in range(T):
            context, attn_weights = self.agent_attn(
                x_agents[:, t, :, :]
            )  # (B, agent_feat), (B, A)
            timestep_reprs.append(context.unsqueeze(1))  # keep time dim
            agent_attn_all.append(attn_weights.unsqueeze(1))

        timestep_reprs = torch.cat(timestep_reprs, dim=1)  # (B, T, agent_feat)
        agent_attn_all = torch.cat(agent_attn_all, dim=1)  # (B, T, A)

        # ---- Temporal encoding ----
        encoder_outputs, hidden = self.encoder(timestep_reprs)  # (B, T, H), (1,B,H)

        # ---- Autoregressive decoding ----
        decoder_input = x[:, -1:, :]  # last timestep raw features (B,1,9)
        outputs = []

        for _ in range(pred_len):
            dec_hidden_t = hidden[-1]  # (B, H)
            context, _ = self.temporal_attn(dec_hidden_t, encoder_outputs)

            dec_input = torch.cat(
                [decoder_input.squeeze(1), context], dim=1
            )  # (B, 9+H)
            dec_input = dec_input.unsqueeze(1)

            out, hidden = self.decoder(dec_input, hidden)
            pred = self.fc(out)  # (B,1,9)

            outputs.append(pred)
            decoder_input = pred  # feedback

        outputs = torch.cat(outputs, dim=1)  # (B, pred_len, 9)

        # Flatten to 2D for scaler compatibility: (B*pred_len, output_size)
        outputs_2d = outputs.reshape(-1, outputs.size(-1))

        return outputs_2d
