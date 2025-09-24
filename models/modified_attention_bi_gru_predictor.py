import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class TrajPredictor(nn.Module):
    def __init__(self, input_size=3, enc_hidden_size=64, dec_hidden_size=64, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        # Shared modules for all agents
        self.encoder = nn.GRU(input_size, enc_hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(enc_hidden_size*2, dec_hidden_size)
        self.enc_to_dec = nn.Linear(enc_hidden_size*2, dec_hidden_size)
        self.decoder = nn.GRU(input_size + enc_hidden_size*2, dec_hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_size, input_size)

    def forward(self, src, tgt=None, pred_len=1):
        """
        src: [batch, seq_len, num_agents * input_size]
        tgt: optional, [batch, pred_len, num_agents * input_size]
        """
        batch_size, seq_len, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)
        if tgt is not None:
            tgt_agents = torch.split(tgt, self.input_size, dim=2)

        outputs_per_agent = []
        for agent_idx in range(num_agents):
            # Encoder
            enc_output, hidden = self.encoder(src_agents[agent_idx])
            num_directions = 2
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1) if num_directions==2 else hidden[-1]
            hidden_dec = self.enc_to_dec(hidden_cat).unsqueeze(0).repeat(self.num_layers, 1, 1)

            # Decoder
            dec_input = src_agents[agent_idx][:, -1:, :]
            agent_outputs = []
            for t in range(pred_len):
                attn_weights = self.attention(hidden_dec[-1], enc_output)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_output)
                rnn_input = torch.cat((dec_input, context), dim=2)
                dec_out, hidden_dec = self.decoder(rnn_input, hidden_dec)
                pred = self.fc_out(dec_out.squeeze(1))
                agent_outputs.append(pred.unsqueeze(1))
                if tgt is not None:
                    dec_input = tgt_agents[agent_idx][:, t:t+1, :]  # type: ignore
                else:
                    dec_input = pred.unsqueeze(1)

            outputs_per_agent.append(torch.cat(agent_outputs, dim=1))
            
        # --- Concatenate all agents ---
        outputs = torch.cat(outputs_per_agent, dim=2)  # [batch, pred_len, num_agents * input_size]

        # If pred_len == 1, remove the middle dimension
        if outputs.size(1) == 1:
            outputs = outputs.squeeze(1)  # [batch, num_agents * input_size]

        return outputs
