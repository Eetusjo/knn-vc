import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class BigVGANVocoder(nn.Module):
    """Wrapper that adapts BigVGAN to accept WavLM features like HiFi-GAN.

    BigVGAN expects mel-spectrogram input (batch, num_mels, time) at ~93.75Hz
    (24000Hz / 256 hop). WavLM produces features at 50Hz (20ms frames).

    This wrapper:
      1. Projects 1024-dim WavLM features to num_mels (100) via learned linear layer
      2. Interpolates from 50Hz to BigVGAN's expected frame rate (~93.75Hz)
      3. Runs BigVGAN synthesis at 24kHz
      4. Resamples output to target_sr (default 16kHz) if needed
    """

    def __init__(self, bigvgan_model, projection: nn.Linear, target_sr: int = 16000):
        """
        Arguments:
            - bigvgan_model: Pretrained BigVGAN model
            - projection: nn.Linear(1024, num_mels) mapping WavLM → pseudo-mel
            - target_sr: Output sample rate (default 16000 to match kNN-VC pipeline)
        """
        super().__init__()
        self.bigvgan = bigvgan_model
        self.projection = projection
        self.bigvgan_sr = bigvgan_model.h.sampling_rate   # 24000
        self.target_sr = target_sr
        self.bigvgan_hop = bigvgan_model.h.hop_size       # 256
        self.wavlm_frame_rate = 50  # Hz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            - x: (batch, seq_len, 1024) WavLM features
        Returns:
            - (batch, 1, num_samples) waveform at target_sr
        """
        # 1. Project WavLM features to pseudo-mel dimension
        x = self.projection(x)  # (batch, seq_len, num_mels)

        # 2. Transpose to Conv1d format expected by BigVGAN
        x = x.permute(0, 2, 1)  # (batch, num_mels, seq_len)

        # 3. Interpolate from 50Hz to BigVGAN's expected frame rate
        #    BigVGAN frame rate = sampling_rate / hop_size = 24000/256 = 93.75Hz
        target_len = int(x.shape[2] * (self.bigvgan_sr / self.bigvgan_hop) / self.wavlm_frame_rate)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        # 4. BigVGAN synthesis at 24kHz
        wav = self.bigvgan(x)  # (batch, 1, num_samples_24k)

        # 5. Resample to target SR if needed
        if self.target_sr != self.bigvgan_sr:
            wav = torchaudio.functional.resample(
                wav.squeeze(1), self.bigvgan_sr, self.target_sr
            ).unsqueeze(1)

        return wav

    def remove_weight_norm(self):
        self.bigvgan.remove_weight_norm()
