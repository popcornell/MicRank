import torchaudio
import torch

class EnvelopeVariance(torch.nn.Module):
   def __init__(self, n_mels=40, n_fft=400, hop_length=200, samplerate=16000, eps=1e-6):
       super(EnvelopeVariance, self).__init__()
       self.mels = torchaudio.transforms.MelSpectrogram(sample_rate=samplerate,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length, n_mels=n_mels, power=1)
       self.eps = eps
       self.subband_weights = torch.nn.Parameter(torch.randn(n_mels))

   def forward(self, channels):
       mels = self.mels(channels)
       logmels = torch.log(mels + self.eps)
       mels = torch.exp(logmels - torch.mean(logmels, -1, keepdim=True))
       var = torch.var(mels ** (1 / 3), dim=-1)
       # channels, subbands
       var = var / torch.amax(var, 1, keepdim=True)
       subband_weights = torch.abs(self.subband_weights)
       ranking = torch.sum(var*subband_weights, -1)
       return ranking


class CepstralDistance(torch.nn.Module):
    def __init__(self,  n_ceps=24, n_fft=400, hop_length=200, eps=1e-6):
        super(CepstralDistance, self).__init__()
        self.spectrum = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length, power=1)
        self.eps = eps
        dct_mat = torchaudio.functional.create_dct(n_ceps, n_fft // 2 +1, 'ortho')
        self.register_buffer("dct_mat", dct_mat)

        self.ceps_weights = torch.nn.Parameter(torch.rand(n_ceps -1))

    def from_oracle(self, channels, close_talk):
        # b, chans, frames = channels.shape
        magspec = self.spectrum(channels)
        reference = torch.log(self.spectrum(close_talk) + self.eps)

        cep_ref = torch.einsum("bft, fc->bct", reference, self.dct_mat)
        cep_all = torch.einsum("bmft, fc->bmct", torch.log(magspec + self.eps), self.dct_mat)

        pairwise_dists = cep_all[:, :, 1:] - cep_ref[:, 1:].unsqueeze(1)  # discard DC coefficient
        ceps_weights = torch.abs(self.ceps_weights)
        pairwise_dists = torch.sum(ceps_weights * torch.sum(pairwise_dists ** 2, -1), -1)

        return pairwise_dists  # batch, channels

    def forward(self, channels):
        #b, chans, frames = channels.shape
        magspec = self.spectrum(channels)
        reference = torch.mean(torch.log(magspec + self.eps), 1)

        cep_ref = torch.einsum("bft, fc->bct", reference, self.dct_mat)
        cep_all = torch.einsum("bmft, fc->bmct", torch.log(magspec + self.eps), self.dct_mat)

        pairwise_dists = cep_all[:, :, 1:] - cep_ref[:, 1:].unsqueeze(1) # discard DC coefficient
        ceps_weights = torch.abs(self.ceps_weights)
        pairwise_dists = (2*torch.sum(ceps_weights*torch.sum(pairwise_dists**2, -1), -1))**0.5

        return pairwise_dists # batch, channels






if __name__ == "__main__":
    d = EnvelopeVariance()
    d(torch.rand((2, 4, 16000)))