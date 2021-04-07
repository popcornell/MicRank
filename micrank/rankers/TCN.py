from torch import nn
from asteroid.masknn import norms
import torch
import math
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import take_mag
import torchaudio
from asteroid

class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].
    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from
            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm
    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """
    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, norm_type="bN", dropout=0.0):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)


        in_bottle = in_chan
        in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), nn.Dropout(dropout), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""

        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
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

class TCN(nn.Module):
    """ Temporal Convolutional network used in ConvTasnet.
    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
    """
    def __init__(self, in_chan=256, out_chan_tcn=1, n_blocks=5, n_repeats=3,
                 bn_chan=64, hid_chan=128,  kernel_size=3,
                 norm_type="gLN",
                 dropout=0.0,
                 chunk=200,
                 stride=40,
                 e2e=False,
                 fb_ksz=64,
                 fb_stride=32,
                 samplerate=16000):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.out_chan_tcn = out_chan_tcn
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.chunk = chunk
        self.stride = stride
        self.e2e = e2e

        if self.e2e:
            self.fbank, _ = make_enc_dec("analytic_free", in_chan, fb_ksz, fb_stride, samplerate)
            in_chan = in_chan // 2
        self.in_norm = norms.get(norm_type)(in_chan)
        self.bottleneck = nn.Sequential(nn.Conv1d(in_chan, bn_chan, 1))
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            res_blocks = nn.ModuleList()
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                res_blocks.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type, dropout=dropout))

            # here TAC
            #res_blocks.append(TAC(bn_chan))
            self.TCN.append(res_blocks)

        #self.dropout = nn.Dropout(0.5)
       # self.out_lstm = torch.nn.LSTM(bn_chan, bn_chan, dropout=dropout,
        #@                              batch_first=True, bidirectional=True)
        #out_act = nn.Conv1d(bn_chan, bn, 1)
        self.out = nn.Sequential(nn.PReLU(), nn.Conv1d(bn_chan, 1, 1))#nn.Linear(2*bn_chan, 1)
        # Get activation function.
        #self.blstm = torch.nn.LSTM(out_chan_tcn*2)

    def forward(self, x, lens=None):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        if self.e2e:
            b, mics, frames = x.size()
            x = x.reshape(b*mics, frames)

            x = take_mag(self.fbank(x)) ** (1 / 3)
            _, chans, frames = x.size()
        else:
            b, mics, chans, frames = x.size()
            x = x.reshape(b * mics, chans, frames)


        if not self.training:
            #if frames < self.chunk:
             #   x = torch.nn.functional.pad(x, ((0, self.chunk-frames)), mode="constant")



            x = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk, 1),
            padding=(self.chunk, 0),
            stride=(self.stride, 1),
        )

            n_chunks = x.shape[-1]
            x = x.reshape(b*mics, chans, self.chunk, n_chunks).permute(0, 3, 1, 2).reshape(b*mics*n_chunks, chans, self.chunk)

        #x = take_mag(self.fbank(x)) ** (1 / 3)
        x = self.in_norm(x)


        x = self.bottleneck(x)
        for i in range(len(self.TCN)):
            for convs_indx in range(len(self.TCN[i])):
                residual = self.TCN[i][convs_indx](x)
                x = x + residual

        if not self.training:
            logits = self.out(x).mean(-1).reshape(b, mics, n_chunks).mean(-1)
        else:
            logits = self.out(x).mean(-1).reshape(b, mics)

        return logits

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

