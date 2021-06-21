import torch
import torch.nn as nn


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/unit/attention/multihead_attention.py#L7
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # q -> [b, n_head, h * w, k_channels] in 2d map or [b, n_head, seq_len, k_channels] in a sequence.
        # k -> [b, n_head, h * w, k_channels] in 2d map or [b, n_head, seq_len, k_channels] in a sequence.
        # v -> [b, n_head, h * w, v_channels] in 2d map or [b, n_head, seq_len, v_channels] in a sequence.
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature  # [b, n_head, h * w, h * w] or [b, n_head, seq_len, seq_len]

        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [b, n_head, h * w, v_channels] or [b, n_head,  seq_len, v_channels]

        return out, attn


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/unit/attention/multihead_attention.py#L29
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, k_channels, v_channels, n_head=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.in_channels = in_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.n_head = n_head  # num of attention head.

        self.q_linear = nn.Linear(in_channels, n_head * k_channels)
        self.k_linear = nn.Linear(in_channels, n_head * k_channels)
        self.v_linear = nn.Linear(in_channels, n_head * v_channels)
        self.attention = ScaledDotProductAttention(temperature=k_channels ** 0.5, dropout=dropout)
        self.out_linear = nn.Linear(n_head * v_channels, in_channels)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v -> [b, h * w, in_channels] in 2d map or [b, seq_len, in_channels] in a sequence.
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.q_linear(q).view(b, q_len, self.n_head, self.k_channels).transpose(1, 2)
        k = self.k_linear(k).view(b, k_len, self.n_head, self.k_channels).transpose(1, 2)
        v = self.v_linear(v).view(b, v_len, self.n_head, self.v_channels).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().view(b, q_len, self.n_head * self.v_channels)
        out = self.out_linear(out)
        out = self.dropout(out)

        # Finally,
        # out -> [b, h * w, in_features] in 2d map or [b, seq_len, in_features] in a sequence.
        # attn -> [b, n_head, h * w, h * w] in 2d map or [b, n_head, seq_len, seq_len] in a sequence.
        return out, attn
