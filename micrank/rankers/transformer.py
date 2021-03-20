from torch import nn
from asteroid.masknn import norms, activations
import torch
import math


class FF(nn.Module):
    def __init__(
        self, in_chan, inner_size, norm_type="gLN", activation="swish", dropout=0.0
    ):
        super().__init__()

        self.norm = norms.get(norm_type)(in_chan)
        self.ff = nn.Sequential(
            nn.Conv1d(in_chan, inner_size, 1),
            activations.get(activation)(),
            nn.Dropout(dropout),
            nn.Conv1d(inner_size, in_chan, 1),
        )

    def forward(self, x):
        # x is batch, channels, frames

        normed = self.norm(x)
        return self.ff(normed) + x


class MHALayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        norm_type="gLN",
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim
        )
        self.norm = norms.get(norm_type)(embed_dim)

    def forward(self, x):

        normed = self.norm(x).permute(2, 0, 1)
        out, _ = self.mha(normed, normed, normed)
        return out.permute(1, 2, 0) + x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[..., : x.size(-1)]
        return self.dropout(x)


class Transformer_SC(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan=1,
        chunk_size=7,
        subsample=10,
        embed_dim=128,
        n_heads=4,
        ff_inner=1024,
        n_layers=4,
        dropout=0.2,
        norm_type="gLN",
        max_seq_len=5000,
    ):
        super(Transformer_SC, self).__init__()

        self.chunk_size = chunk_size
        self.subsample = subsample
        in_bottleneck = (2 * chunk_size + 1) * in_chan if chunk_size > 1 else in_chan

        self.norm_sc = norms.get(norm_type)(in_bottleneck)
        self.bottleneck = nn.Sequential(nn.Conv1d(in_bottleneck, embed_dim, 1))
        self.pos_encs = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        self.output = nn.Conv1d(embed_dim, out_chan, 1)

        self.layers = nn.ModuleList([])
        for l in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MHALayer(embed_dim, n_heads, dropout, norm_type=norm_type),
                        FF(embed_dim, ff_inner, norm_type, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):

        batch, mics, chans, frames = x.shape
        x = x.reshape(batch*mics, chans, frames)

        # cat subsample
        if self.chunk_size > 1:
            x = torch.nn.functional.unfold(
                x.unsqueeze(-1),
                kernel_size=(self.chunk_size * 2 + 1, 1),
                padding=(self.chunk_size, 0),
                stride=(1, 1),
            )

        if self.subsample > 1:
            x = x[..., :: self.subsample]

        x = self.norm_sc(x)
        x = self.bottleneck(x)
        x = self.pos_encs(x)

        for i in range(len(self.layers)):
            mha, ff = self.layers[i]
            x = ff(mha(x))

        return self.output(x).mean(-1).reshape(batch, mics)
