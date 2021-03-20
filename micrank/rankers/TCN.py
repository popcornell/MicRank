from torch import nn
from asteroid.masknn import norms
import torch


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
    def __init__(self, in_chan, out_chan=1, n_blocks=5, n_repeats=3,
                 bn_chan=64, hid_chan=128,  kernel_size=3,
                 norm_type="gLN", dropout=0.0):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
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

        out_conv = nn.Conv1d(bn_chan, out_chan, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        # Get activation function.

    def forward(self, mixture_w, lens=None):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        b, mics, chans, frames = mixture_w.size()
        mixture_w = mixture_w.reshape(b*mics, chans, frames)
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            for convs_indx in range(len(self.TCN[i]) - 1):
                residual = self.TCN[i][convs_indx](output)
                output = output + residual
            #tac = self.TCN[i][-1]
            # apply TAC
            #_, chans, frames = output.size()
            #output = output.reshape(b, mics, chans, frames)
            #output = tac(output)
            #output = output.reshape(b*mics, chans, frames)

        logits = self.out(output)

        # perform mean only on non padded frames
        if lens is not None:
            import ipdb
            ipdb.set_trace()
            # masked fill

        return logits.mean(-1).squeeze(-1).reshape(b, -1)

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