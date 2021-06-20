import math
import random

import torch
import torch.nn as nn

from .backbones import Shallow_cnn, DeepCNN300, efficientnet_backbone
from .Attention import MultiHeadAttention
from .FeedForward import FeedForward1D, FeedForward2D


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/position_encoder/adaptive_2d_encoder.py#L8
class Adaptive2DPositionEncoder(nn.Module):
    def __init__(self, in_channels, max_h=200, max_w=200, dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()

        h_position_encoder = self.generate_encoder(in_channels, max_h)  # [max_h, in_channels]
        h_position_encoder = h_position_encoder.transpose(0, 1)  # [in_channels, max_h]
        h_position_encoder = h_position_encoder.view(1, in_channels, max_h, 1)  # [1, in_channels, max_h, 1] -> braodcase for batch feature map.

        w_position_encoder = self.generate_encoder(in_channels, max_w)  # [max_w, in_channels]
        w_position_encoder = w_position_encoder.transpose(0, 1)  # [in_channels, max_w]
        w_position_encoder = w_position_encoder.view(1, in_channels, 1, max_w)  # [1, in_channels, 1, max_w] -> broadcast for batched feature map.

        self.register_buffer("h_position_encoder", h_position_encoder)
        self.register_buffer("w_position_encoder", w_position_encoder)

        self.h_scale = self.scale_factor_generate(in_channels)
        self.w_scale = self.scale_factor_generate(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)  # [max_len, 1]

        i = torch.arange(in_channels).float().unsqueeze(0)  # [1, in_channels]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates  # [max_len, in_channels]
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def scale_factor_generate(self, in_channels):
        scale_factor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid(),
        )

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)  # [b, c, 1, 1]

        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/position_encoder/encoder.py#L8
class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=2000, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        position_encoder = self.generate_encoder(in_channels, max_len)
        position_encoder = position_encoder.unsqueeze(0)  # [1, max_len, in_channels]

        self.register_buffer("position_encoder", position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)  # [max_len, 1]

        i = torch.arange(in_channels).float().unsqueeze(0)  # [1, in_channels]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates  # [max_len, in_channels]
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def forward(self, x):
        out = x + self.position_encoder[:, x.size(1), :]
        out = self.dropout(out)

        return out


# Reference: https://github.com/Media-Smart/vedastr/blob/f7c1a928e88dd1fb41107103afb548cbf87855d7/vedastr/models/bodies/sequences/transformer/unit/encoder.py#L10
class TransformerEncoderLayer1D(nn.Module):
    def __init__(self, in_channels, k_channels, v_channels, n_head=8, dropout=0.1):
        super(TransformerEncoderLayer1D, self).__init__()

        self.attention = MultiHeadAttention(in_channels=in_channels, k_channels=k_channels, v_channels=v_channels, n_head=n_head, dropout=dropout)
        self.attention_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.feedforward = FeedForward1D(hidden_dim=in_channels, dropout=dropout)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, src, src_mask=None):
        attn_out, _ = self.attention(q=src, k=src, v=src, mask=src_mask)
        out_1 = self.attention_norm(src + attn_out)

        ffn_out = self.feedforward(out_1)
        out_2 = self.feedforward_norm(out_1 + ffn_out)

        return out_2


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/unit/encoder.py#L31
class TransformerEncoderLayer2D(nn.Module):
    def __init__(self, in_channels, filter_dim, k_channels, v_channels, n_head=8, dropout=0.1):
        super(TransformerEncoderLayer2D, self).__init__()

        self.attention = MultiHeadAttention(in_channels=in_channels, k_channels=k_channels, v_channels=v_channels, n_head=n_head, dropout=dropout)
        self.attention_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.feedforward = FeedForward2D(hidden_dim=in_channels, filter_dim=filter_dim, dropout=dropout)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, src, src_mask=None):
        b, c, h, w = src.size()

        src = src.view(b, c, h * w).transpose(1, 2)  # [b, h * w, c]
        if src_mask is not None:
            src_mask = src_mask.view(b, 1, h * w)

        attn_out, _ = self.attention(q=src, k=src, v=src, mask=src_mask)  # [b, h * w, c]
        out_1 = self.attention_norm(src + attn_out)
        out_1_2d = out_1.transpose(1, 2).contiguous().view(b, c, h, w)

        ffn_out = self.feedforward(out_1_2d).view(b, c, h * w).transpose(1, 2)
        out_2 = self.feedforward_norm(out_1 + ffn_out)
        out_2_2d = out_2.transpose(1, 2).contiguous().view(b, c, h, w)

        return out_2_2d


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/unit/decoder.py#L10
class TransformerDecoderLayer1D(nn.Module):
    def __init__(self, in_channels, filter_dim, k_channels, v_channels, n_head=8, dropout=0.1):
        super(TransformerDecoderLayer1D, self).__init__()

        self.self_attention = MultiHeadAttention(in_channels=in_channels, k_channels=k_channels, v_channels=v_channels, n_head=n_head, dropout=dropout)
        self.self_attention_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.attention = MultiHeadAttention(in_channels=in_channels, k_channels=k_channels, v_channels=v_channels, n_head=n_head, dropout=dropout)
        self.attention_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.feedforward = FeedForward1D(hidden_dim=in_channels, filter_dim=filter_dim, dropout=dropout)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        attn_1, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        out_1 = self.self_attention_norm(attn_1)

        size = src.size()
        if len(size) == 4:
            b, c, h, w = size
            src = src.view(b, c, h * w).transpose(1, 2)
            if src_mask is not None:
                src_mask = src_mask.view(b, 1, h * w)

        attn_2, _ = self.attention(out_1, src, src, src_mask)
        out_2 = self.attention_norm(out_1 + attn_2)

        ffn_out = self.feedforward(out_2)
        out_3 = self.feedforward_norm(out_2 + ffn_out)

        return out_3


# Reference: https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/vedastr/models/bodies/sequences/transformer/encoder.py#L15
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, filter_dim, hidden_dim, block_num, max_h=200, max_w=200, n_head=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # self.shallow_cnn = Shallow_cnn(in_channels=in_channels, hidden_dim=hidden_dim)
        # self.shallow_cnn = DeepCNN300(in_channels, num_in_features=48, hidden_dim=hidden_dim, dropout=dropout,)
        self.shallow_cnn = efficientnet_backbone(in_channels, hidden_dim, dropout)
        self.pos_encoder = Adaptive2DPositionEncoder(in_channels=hidden_dim, max_h=max_h, max_w=max_w, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer2D(in_channels=hidden_dim, filter_dim=filter_dim, k_channels=hidden_dim, v_channels=hidden_dim, n_head=n_head, dropout=dropout)
                for _ in range(block_num)
            ]
        )

    def forward(self, src, src_mask=None):
        src = self.shallow_cnn(src)
        src = self.pos_encoder(src)

        for block in self.blocks:
            src = block(src, src_mask)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, num_classes, hidden_dim, filter_dim, block_num, pad_id, st_id, n_head=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
        self.pos_encoder = PositionEncoder1D(in_channels=hidden_dim, max_len=500, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderLayer1D(in_channels=hidden_dim, filter_dim=filter_dim, k_channels=hidden_dim, v_channels=hidden_dim, n_head=n_head, dropout=dropout)
                for _ in range(block_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id

    def pad_mask(self, text):
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, text):
        seq_len = text.size(1)
        order_mask = torch.triu(torch.ones(seq_len, seq_len, device=text.device), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0)

        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(self, feats, texts, is_train=True, max_length=50, teacher_forcing_ratio=1.0):
        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(texts)
            tgt_mask = self.pad_mask(texts) | self.order_mask(texts)

            tgt = self.pos_encoder(tgt)
            for block in self.blocks:
                tgt = block(tgt, feats, tgt_mask, None)
            out = self.generator(tgt)
        else:
            out = None
            texts = torch.full(size=(feats.size(0), 1), dtype=torch.long, fill_value=self.st_id, device=feats.device)

            for _ in range(max_length):
                tgt = self.text_embedding(texts)
                tgt_mask = self.order_mask(texts)

                tgt = self.pos_encoder(tgt)
                for block in self.blocks:
                    tgt = block(tgt, feats, tgt_mask, None)
                out = self.generator(tgt)

                next_text = torch.argmax(out[:, -1:, :], dim=-1)
                texts = torch.cat([texts, next_text], dim=-1)

        return out


class SATRN(nn.Module):
    def __init__(self, config, tokenizer):
        super(SATRN, self).__init__()

        self.encoder = TransformerEncoder(
            in_channels=3 if config.data.rgb else 1,
            hidden_dim=config.model.hidden_dim,
            filter_dim=config.model.filter_dim,
            block_num=config.model.n_e,
            n_head=config.model.num_head,
            dropout=config.model.dropout_rate,
        )
        self.decoder = TransformerDecoder(
            num_classes=len(tokenizer.token_to_id),
            hidden_dim=config.model.hidden_dim,
            filter_dim=config.model.filter_dim,
            block_num=config.model.n_d,
            pad_id=tokenizer.token_to_id[tokenizer.PAD_TOKEN],
            st_id=tokenizer.token_to_id[tokenizer.START_TOKEN],
            n_head=config.model.num_head,
            dropout=config.model.dropout_rate,
        )

    def forward(self, images, texts=None, is_train=True, teacher_forcing_ratio=1.0):
        enc_result = self.encoder(images)
        dec_result = self.decoder(enc_result, texts[:, :-1], is_train, texts[:, :-1].size(1), teacher_forcing_ratio,)

        return dec_result
